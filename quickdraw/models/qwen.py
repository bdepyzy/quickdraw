"""Single-token Qwen reference math with static FP8 and W4A16 projections."""

import torch
import torch.nn.functional as F

from ..kvcache.state import DecodeState
from .weights import (
    Checkpoint,
    FP8Linear,
    FullAttnWeights,
    GDNWeights,
    ModelConfig,
    MoEWeights,
    NVFP4Linear,
    unpack_fp4,
)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the last axis in FP32, then return the input dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)


def quantize_fp8(x: torch.Tensor, input_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (x.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn), input_scale


def fp8_gemv(x: torch.Tensor, proj: FP8Linear) -> torch.Tensor:
    assert proj.input_scale is not None, "static FP8 requires an input scale"
    xq, input_scale = quantize_fp8(x, proj.input_scale)

    return F.linear(xq.float() * input_scale, proj.weight.float() * proj.weight_scale).to(x.dtype)


def nvfp4_gemv(x: torch.Tensor, proj: NVFP4Linear, *, routing_weight=None,
               row_chunk: int = 1024) -> torch.Tensor:
    """W4A16 dot with late global scaling; routed down projections round twice."""
    assert proj.weight.ndim == 2 and x.ndim == 1 and x.numel() == proj.in_features
    assert row_chunk > 0
    result = torch.empty(proj.out_features, dtype=x.dtype, device=x.device)
    activation = x.float().unsqueeze(0)
    scale = proj.weight_scale_2.float()
    if routing_weight is not None:
        scale = (scale * routing_weight).to(x.dtype).float()
    for start in range(0, proj.out_features, row_chunk):
        stop = min(start + row_chunk, proj.out_features)
        base = unpack_fp4(proj.weight[start:stop]) * proj.weight_scale[start:stop].float().repeat_interleave(
            proj.group_size, dim=-1)
        accumulated = F.linear(activation, base).squeeze(0)
        if routing_weight is not None:
            accumulated = accumulated.to(x.dtype).float()
        result[start:stop] = accumulated * scale
    return result


def _expert_projection(proj: NVFP4Linear, expert: int) -> NVFP4Linear:
    """View one expert's packed weights and scales without copying."""
    global_scale = proj.weight_scale_2
    if global_scale.numel() != 1:
        global_scale = global_scale[expert]
    return NVFP4Linear(proj.weight[expert], proj.weight_scale[expert], global_scale,
                       group_size=proj.group_size)


def full_attention_block(w: FullAttnWeights, x: torch.Tensor,
                         kv_cache: tuple[torch.Tensor, torch.Tensor], position: int, *,
                         config: ModelConfig, project_fp8=fp8_gemv) -> torch.Tensor:
    """GQA with per-head query/gate packing, partial RoPE, and sigmoid gating.
    Q/K gamma is 1 + weight; cache normalized, rotated keys through position."""
    key_cache, value_cache = kv_cache
    assert 0 <= position < min(key_cache.shape[0], value_cache.shape[0])
    assert config.num_key_value_heads > 0
    assert config.num_attention_heads % config.num_key_value_heads == 0

    q_and_gate = project_fp8(x, w.q_proj)
    head_rows = q_and_gate.view(config.num_attention_heads, 2 * config.head_dim)
    q, gate = head_rows.chunk(2, dim=-1)
    k = project_fp8(x, w.k_proj).view(config.num_key_value_heads, config.head_dim)
    v = project_fp8(x, w.v_proj).view(config.num_key_value_heads, config.head_dim)

    q = rms_norm(q, 1.0 + w.q_norm.float(), config.rms_norm_eps)
    k = rms_norm(k, 1.0 + w.k_norm.float(), config.rms_norm_eps)

    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    assert rotary_dim % 2 == 0 and 0 <= rotary_dim <= config.head_dim
    if rotary_dim:
        # BF16 trig cache, FP32 rotation, then BF16 storage.
        indices = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=q.device)
        inv_freq = 1.0 / (config.rope_theta ** (indices / rotary_dim))
        angles = position * inv_freq
        cos = torch.cat((angles.cos(), angles.cos())).to(q.dtype).float()
        sin = torch.cat((angles.sin(), angles.sin())).to(q.dtype).float()
        rotated_heads = []
        for heads in (q, k):
            features = heads[..., :rotary_dim].float()
            first, second = features.chunk(2, dim=-1)
            rotated = features * cos + torch.cat((-second, first), dim=-1) * sin
            rotated_heads.append(torch.cat((rotated.to(heads.dtype), heads[..., rotary_dim:]), dim=-1))
        q, k = rotated_heads

    key_cache[position].copy_(k)
    value_cache[position].copy_(v)

    keys, values = (cache[:position + 1].repeat_interleave(
        config.num_attention_heads // config.num_key_value_heads, dim=1).transpose(0, 1)
        for cache in (key_cache, value_cache))

    scores = torch.matmul(q.float().unsqueeze(1), keys.float().transpose(1, 2)).squeeze(1)
    scores = scores * (config.head_dim ** -0.5)
    attention_weights = torch.softmax(scores, dim=-1)

    attended = torch.matmul(attention_weights.unsqueeze(1), values.float()).squeeze(1).to(q.dtype)
    gated = attended * torch.sigmoid(gate)

    return project_fp8(gated.reshape(-1), w.o_proj)


def gated_delta_net_block(w: GDNWeights, x: torch.Tensor, rec_state: torch.Tensor,
                          conv_state: torch.Tensor, position: int, *, config: ModelConfig,
                          project_fp8=fp8_gemv) -> torch.Tensor:
    """FP32 gated delta recurrence over convolved QKV, with direct-gamma RMSNorm.
    conv_state stores raw QKV; rec_state stores FP32 memory; z gates with SiLU."""
    mixed_qkv = project_fp8(x, w.in_proj_qkv)

    z = project_fp8(x, w.in_proj_z).view(
        config.linear_num_value_heads, config.linear_value_head_dim)

    a = F.linear(x.unsqueeze(0), w.in_proj_a).squeeze(0)
    b = F.linear(x.unsqueeze(0), w.in_proj_b).squeeze(0)

    assert config.linear_num_value_heads % config.linear_num_key_heads == 0

    window = torch.cat((conv_state, mixed_qkv.unsqueeze(-1)), dim=-1)
    conv_state.copy_(window[:, 1:])
    convolved = F.silu((window * w.conv1d.squeeze(1)).float().sum(-1)).to(x.dtype)

    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    value_width = config.linear_num_value_heads * config.linear_value_head_dim
    q, k, v = convolved.split((key_width, key_width, value_width))
    q = q.view(config.linear_num_key_heads, config.linear_key_head_dim).float()
    k = k.view(config.linear_num_key_heads, config.linear_key_head_dim).float()
    v = v.view(config.linear_num_value_heads, config.linear_value_head_dim).float()
    repeats = config.linear_num_value_heads // config.linear_num_key_heads
    q = q.repeat_interleave(repeats, dim=0)
    k = k.repeat_interleave(repeats, dim=0)

    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q / (config.linear_key_head_dim ** 0.5)

    beta = b.float().sigmoid().unsqueeze(-1)
    log_decay = -w.A_log.float().exp() * F.softplus(a.float() + w.dt_bias.float())
    decayed = rec_state * log_decay.exp()[:, None, None]
    prediction = (decayed * k.unsqueeze(-1)).sum(dim=-2)
    correction = beta * (v - prediction)
    updated = decayed + k.unsqueeze(-1) * correction.unsqueeze(-2)
    rec_state.copy_(updated)
    attended = (updated * q.unsqueeze(-1)).sum(dim=-2).to(x.dtype)

    attended_fp32 = attended.float()
    normalized = attended_fp32 * torch.rsqrt(
        attended_fp32.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    gated = (normalized * w.norm.float() * F.silu(z.float())).to(x.dtype)
    return project_fp8(gated.reshape(-1), w.out_proj)


def moe_block(w: MoEWeights, x: torch.Tensor, *, config: ModelConfig,
              project_fp4=nvfp4_gemv) -> torch.Tensor:
    """Top-k NVFP4 experts with renormalized routing and a gated shared expert."""
    assert 1 <= config.num_experts_per_tok <= config.num_experts

    router_logits = F.linear(x.unsqueeze(0), w.router).squeeze(0)
    probabilities = torch.softmax(router_logits.float(), dim=-1)
    selected_ids = probabilities.argsort(descending=True, stable=True)[:config.num_experts_per_tok]
    selected_weights = probabilities[selected_ids]
    selected_weights = selected_weights / selected_weights.sum()

    output = torch.zeros_like(x, dtype=torch.float32)
    order = selected_ids.argsort()
    for slot, expert in zip(order.tolist(), selected_ids[order].tolist()):
        gate = project_fp4(x, _expert_projection(w.experts_gate, expert))
        up = project_fp4(x, _expert_projection(w.experts_up, expert))
        hidden = F.silu(gate) * up
        contribution = project_fp4(hidden, _expert_projection(w.experts_down, expert),
                                   routing_weight=selected_weights[slot])
        output = output + contribution.float()

    shared_gate = project_fp4(x, w.shared_gate)
    shared_up = project_fp4(x, w.shared_up)
    shared = project_fp4(F.silu(shared_gate) * shared_up, w.shared_down)
    shared_strength = F.linear(x.unsqueeze(0), w.shared_expert_gate).sigmoid().reshape(())
    return output.to(x.dtype) + shared * shared_strength


def forward_step(ckpt: Checkpoint, token_id: torch.Tensor, position: int, state: DecodeState, *,
                 normalize=rms_norm, attention=full_attention_block,
                 delta_net=gated_delta_net_block, experts=moe_block,
                 project_fp4=nvfp4_gemv, prepared=None, logits=True) -> torch.Tensor | None:
    """Process a token or chunk; optionally return its final token's BF16 logits.
    The engine advances position. Norm gamma is 1 + stored weight."""
    config = ckpt.config
    assert len(ckpt.layers) == config.num_hidden_layers
    assert all(layer.idx == index and layer.layer_type == config.layer_types[index]
               for index, layer in enumerate(ckpt.layers)), "complete checkpoint required"
    assert token_id.ndim in (0, 1) and token_id.dtype == torch.long
    if isinstance(position, int):
        assert 0 <= position < state.kv_k.shape[1]

    x = F.embedding(token_id, ckpt.embed_tokens)
    residual = torch.zeros_like(x)
    full_index = linear_index = 0
    for layer in ckpt.layers:
        combined = x.float() + residual.float()
        gamma = prepared[0][layer.idx][0] if prepared else 1 + layer.input_layernorm.float()
        normalized = normalize(combined, gamma, config.rms_norm_eps).to(x.dtype)
        residual = combined.to(x.dtype)
        if layer.layer_type == "full_attention":
            mixed = attention(
                layer.attn, normalized, (state.kv_k[full_index], state.kv_v[full_index]),
                position, config=config)
            full_index += 1
        else:
            assert layer.layer_type == "linear_attention"
            mixed = delta_net(
                layer.attn, normalized, state.rec_state[linear_index],
                state.conv_state[linear_index], position, config=config)
            linear_index += 1
        combined = mixed.float() + residual.float()
        gamma = prepared[0][layer.idx][1] if prepared else 1 + layer.post_attention_layernorm.float()
        normalized = normalize(combined, gamma, config.rms_norm_eps).to(x.dtype)
        residual = combined.to(x.dtype)
        x = experts(layer.moe, normalized, config=config)

    if not logits:
        return None
    if x.ndim == 2:
        x, residual = x[-1], residual[-1]
    gamma = prepared[1] if prepared else 1 + ckpt.final_norm.float()
    normalized = normalize(x.float() + residual.float(), gamma, config.rms_norm_eps).to(x.dtype)
    return project_fp4(normalized, ckpt.lm_head)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits)
    assert 0.0 < temperature < float("inf"), "temperature must be finite and nonnegative"

    scores = logits.float()
    scores = (scores - scores.max()) / temperature
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
