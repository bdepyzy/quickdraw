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
)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the last axis in FP32, then return the input dtype."""
    xf = x.float()
    mean_square = xf.square().mean(dim=-1, keepdim=True)
    normalized = xf * torch.rsqrt(mean_square + eps)
    y = normalized * weight
    return y.to(x.dtype)


def quantize_fp8(x: torch.Tensor, input_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.float() / input_scale
    limit = torch.finfo(torch.float8_e4m3fn).max
    clamped = scale.clamp(-limit, limit)
    q = clamped.to(torch.float8_e4m3fn)
    return q, input_scale


def fp8_gemv(x: torch.Tensor, proj: FP8Linear) -> torch.Tensor:
    assert proj.input_scale is not None, "static FP8 requires an input scale"
    xq, input_scale = quantize_fp8(x, proj.input_scale)

    x_dequant = xq.float() * input_scale
    w_dequant = proj.weight.float() * proj.weight_scale

    y = F.linear(x_dequant.unsqueeze(0), w_dequant).squeeze(0)

    return y.to(x.dtype)


def nvfp4_gemv(x: torch.Tensor, proj: NVFP4Linear, *, row_chunk: int = 1024) -> torch.Tensor:
    """W4A16 projection, dequantizing at most row_chunk output rows at once.
    Calibration input_scale is unused in this weight-only format."""
    assert proj.weight.ndim == 2 and x.ndim == 1 and x.numel() == proj.in_features
    assert row_chunk > 0
    result = torch.empty(proj.out_features, dtype=x.dtype, device=x.device)
    activation = x.unsqueeze(0)
    for start in range(0, proj.out_features, row_chunk):
        stop = min(start + row_chunk, proj.out_features)
        chunk = NVFP4Linear(proj.weight[start:stop], proj.weight_scale[start:stop],
                            proj.weight_scale_2, group_size=proj.group_size)
        result[start:stop] = F.linear(activation, chunk.dequantize()).squeeze(0)
    return result


def _expert_projection(proj: NVFP4Linear, expert: int) -> NVFP4Linear:
    """View one expert's packed weights and scales without copying."""
    global_scale = proj.weight_scale_2
    if global_scale.numel() != 1:
        global_scale = global_scale[expert]
    return NVFP4Linear(proj.weight[expert], proj.weight_scale[expert], global_scale,
                       group_size=proj.group_size)


def full_attention_block(
    w: FullAttnWeights,
    x: torch.Tensor,          # [hidden] bf16 (already normed)
    kv_cache: tuple[torch.Tensor, torch.Tensor],  # K,V [ctx, kv_heads, head_dim] bf16
    position: int,
    *,
    config: ModelConfig,
    project_fp8=fp8_gemv,
) -> torch.Tensor:
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
        # FP32 frequencies; rotation in the activation dtype.
        indices = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=q.device)
        inv_freq = 1.0 / (config.rope_theta ** (indices / rotary_dim))
        angles = position * inv_freq
        cos = torch.cat((angles.cos(), angles.cos())).to(q.dtype)
        sin = torch.cat((angles.sin(), angles.sin())).to(q.dtype)
        rotated_heads = []
        for heads in (q, k):
            features = heads[..., :rotary_dim]
            first, second = features.chunk(2, dim=-1)
            rotated = features * cos + torch.cat((-second, first), dim=-1) * sin
            rotated_heads.append(torch.cat((rotated, heads[..., rotary_dim:]), dim=-1))
        q, k = rotated_heads

    key_cache[position].copy_(k)
    value_cache[position].copy_(v)

    active_length = position + 1
    keys = key_cache[:active_length]
    values = value_cache[:active_length]

    heads_per_kv = config.num_attention_heads // config.num_key_value_heads
    keys = keys.repeat_interleave(heads_per_kv, dim=1).transpose(0, 1)
    values = values.repeat_interleave(heads_per_kv, dim=1).transpose(0, 1)

    scores = torch.matmul(q.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
    scores = scores * (config.head_dim ** -0.5)
    attention_weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)

    attended = torch.matmul(attention_weights.unsqueeze(1), values).squeeze(1)
    gated = attended * torch.sigmoid(gate)

    return project_fp8(gated.reshape(-1), w.o_proj)


def gated_delta_net_block(
    w: GDNWeights,
    x: torch.Tensor,          # [hidden] bf16 (already normed)
    rec_state: torch.Tensor,  # [value_heads, key_dim, value_dim] fp32
    conv_state: torch.Tensor,  # [channels, kernel-1] bf16
    position: int,
    *,
    config: ModelConfig,
    project_fp8=fp8_gemv,
) -> torch.Tensor:
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
    convolved = F.conv1d(window.unsqueeze(0), w.conv1d,
                        groups=mixed_qkv.numel()).reshape(-1)
    convolved = F.silu(convolved)

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

    # BF16 sigmoid before the FP32 recurrence.
    beta = b.sigmoid().float().unsqueeze(-1)
    log_decay = -w.A_log.float().exp() * F.softplus(a.float() + w.dt_bias.float())
    decayed = rec_state * log_decay.exp()[:, None, None]
    prediction = (decayed * k.unsqueeze(-1)).sum(dim=-2)
    correction = beta * (v - prediction)
    updated = decayed + k.unsqueeze(-1) * correction.unsqueeze(-2)
    rec_state.copy_(updated)
    attended = (updated * q.unsqueeze(-1)).sum(dim=-2).to(x.dtype)

    # Cast normalized features before gamma; evaluate the gate in FP32.
    attended_fp32 = attended.float()
    normalized = attended_fp32 * torch.rsqrt(
        attended_fp32.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    normalized = normalized.to(x.dtype) * w.norm
    gated = (normalized * F.silu(z.float())).to(x.dtype)
    return project_fp8(gated.reshape(-1), w.out_proj)


def moe_block(
    w: MoEWeights,
    x: torch.Tensor,          # [hidden] bf16 (already normed)
    *,
    config: ModelConfig,
    project_fp4=nvfp4_gemv,
) -> torch.Tensor:
    """Top-k NVFP4 experts with renormalized routing and a gated shared expert."""
    assert 1 <= config.num_experts_per_tok <= config.num_experts

    router_logits = F.linear(x.unsqueeze(0), w.router).squeeze(0)
    probabilities = torch.softmax(router_logits.float(), dim=-1)
    selected_weights, selected_ids = probabilities.topk(config.num_experts_per_tok)
    selected_weights = (selected_weights / selected_weights.sum()).to(x.dtype)

    output = torch.zeros_like(x, dtype=torch.float32)
    # Accumulate BF16-weighted expert contributions in FP32.
    order = selected_ids.argsort()
    for slot, expert in zip(order.tolist(), selected_ids[order].tolist()):
        gate = project_fp4(x, _expert_projection(w.experts_gate, expert))
        up = project_fp4(x, _expert_projection(w.experts_up, expert))
        hidden = F.silu(gate) * up
        contribution = project_fp4(hidden, _expert_projection(w.experts_down, expert))
        output = output + (contribution * selected_weights[slot]).float()

    shared_gate = project_fp4(x, w.shared_gate)
    shared_up = project_fp4(x, w.shared_up)
    shared = project_fp4(F.silu(shared_gate) * shared_up, w.shared_down)
    shared_strength = F.linear(x.unsqueeze(0), w.shared_expert_gate).sigmoid().reshape(())
    return output.to(x.dtype) + shared * shared_strength


def forward_step(
    ckpt: Checkpoint,
    token_id: torch.Tensor,   # scalar long
    position: int,
    state: DecodeState,
    *,
    normalize=rms_norm,
    attention=full_attention_block,
    delta_net=gated_delta_net_block,
    experts=moe_block,
    project_fp4=nvfp4_gemv,
) -> torch.Tensor:
    """Update history and return BF16 logits [vocab]; the engine advances position.
    Layer/final norm gamma is 1 + stored weight."""
    config = ckpt.config
    assert len(ckpt.layers) == config.num_hidden_layers
    assert all(layer.idx == index and layer.layer_type == config.layer_types[index]
               for index, layer in enumerate(ckpt.layers)), "complete checkpoint required"
    assert token_id.ndim == 0 and token_id.dtype == torch.long
    assert 0 <= position < state.kv_k.shape[1]

    x = F.embedding(token_id, ckpt.embed_tokens)
    full_index = linear_index = 0
    for layer in ckpt.layers:
        normalized = normalize(x, 1.0 + layer.input_layernorm.float(), config.rms_norm_eps)
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
        x = x + mixed
        normalized = normalize(x, 1.0 + layer.post_attention_layernorm.float(), config.rms_norm_eps)
        x = x + experts(layer.moe, normalized, config=config)

    normalized = normalize(x, 1.0 + ckpt.final_norm.float(), config.rms_norm_eps)
    return project_fp4(normalized, ckpt.lm_head)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits)
    assert 0.0 < temperature < float("inf"), "temperature must be finite and nonnegative"

    scores = logits.float()
    scores = (scores - scores.max()) / temperature
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
