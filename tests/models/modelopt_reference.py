"""HF attention/recurrence with independent adapters for measured ModelOpt arithmetic."""

import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.models import referencegen as hf_reference
from tests.models.mathcheck import pack, save_reference, unpack


def fp4_base(proj):
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    codes = torch.stack((proj.weight.long() % 16, proj.weight.long() // 16), -1).flatten(-2)
    return lut[codes] * proj.weight_scale.float().repeat_interleave(proj.group_size, -1)


def linear(x, proj):
    return (F.linear(x.float(), fp4_base(proj)) * proj.weight_scale_2.float()).to(x.dtype)


def full_attention_block(w, x, kv_cache, position, *, config):
    _, hf = hf_reference.backend()
    rotary, attention = hf.apply_rotary_pos_emb, hf.eager_attention_forward

    def rotate(q, k, cos, sin, **kwargs):
        q_rot, k_rot = rotary(q.float(), k.float(), cos.to(q.dtype).float(), sin.to(k.dtype).float(), **kwargs)
        return q_rot.to(q.dtype), k_rot.to(k.dtype)

    def attend(module, q, k, v, *args, **kwargs):
        output, weights = attention(module, q.float(), k.float(), v.float(), *args, **kwargs)
        return output.to(q.dtype), weights

    with patch.object(hf, "apply_rotary_pos_emb", rotate), patch.object(hf, "eager_attention_forward", attend):
        return hf_reference.full_attention_block(w, x, kv_cache, position, config=config)


def moe_block(w, x, *, config):
    scores = F.linear(x, w.router).softmax(-1, dtype=torch.float32)
    ranked, indices = torch.sort(scores, descending=True, stable=True)
    weights, ids = ranked[:config.num_experts_per_tok], indices[:config.num_experts_per_tok]
    weights /= weights.sum()
    gates = torch.bmm(fp4_base(w.experts_gate)[ids], x.float().expand(ids.numel(), -1).unsqueeze(-1)).squeeze(-1)
    ups = torch.bmm(fp4_base(w.experts_up)[ids], x.float().expand(ids.numel(), -1).unsqueeze(-1)).squeeze(-1)
    gate_scales = w.experts_gate.weight_scale_2.expand(config.num_experts)[ids]
    up_scales = w.experts_up.weight_scale_2.expand(config.num_experts)[ids]
    hidden = F.silu((gates * gate_scales[:, None]).to(x.dtype)) * (ups * up_scales[:, None]).to(x.dtype)
    down = torch.bmm(fp4_base(w.experts_down)[ids], hidden.float().unsqueeze(-1)).squeeze(-1).to(x.dtype)
    down_scales = w.experts_down.weight_scale_2.expand(config.num_experts)[ids]
    routed = (down * (down_scales * weights).to(x.dtype)[:, None]).float().sum(0).to(x.dtype)
    shared = linear(F.silu(linear(x, w.shared_gate)) * linear(x, w.shared_up), w.shared_down)
    return routed + shared * F.linear(x, w.shared_expert_gate).sigmoid().reshape(())


def gated_delta_net_block(w, x, rec_state, conv_state, position, *, config):
    _, hf = hf_reference.backend()
    mixed = hf_reference.StaticFP8(w.in_proj_qkv)(x)
    history = torch.cat((conv_state, mixed[:, None]), -1)
    products = history * w.conv1d[:, 0]
    convolved = F.silu(products.float().sum(-1)).to(x.dtype)
    conv_state.copy_(history[:, 1:])
    key_width = config.linear_num_key_heads * config.linear_key_head_dim
    q, k, v = convolved.split((key_width, key_width, config.linear_num_value_heads * config.linear_value_head_dim))
    repeats = config.linear_num_value_heads // config.linear_num_key_heads
    q = q.reshape(1, 1, config.linear_num_key_heads, config.linear_key_head_dim).repeat_interleave(repeats, 2)
    k = k.reshape(1, 1, config.linear_num_key_heads, config.linear_key_head_dim).repeat_interleave(repeats, 2)
    v = v.reshape(1, 1, config.linear_num_value_heads, config.linear_value_head_dim)
    alpha = F.linear(x, w.in_proj_a).float() + w.dt_bias.float()
    decay = -w.A_log.float().exp() * F.softplus(alpha)
    beta = F.linear(x, w.in_proj_b).float().sigmoid()
    core, state = hf.torch_recurrent_gated_delta_rule(
        q, k, v, g=decay.reshape(1, 1, -1), beta=beta.reshape(1, 1, -1),
        initial_state=rec_state.unsqueeze(0), output_final_state=True, use_qk_l2norm_in_kernel=True)
    rec_state.copy_(state[0])
    z = hf_reference.StaticFP8(w.in_proj_z)(x).reshape(-1, config.linear_value_head_dim)
    normalized = F.rms_norm(core.reshape_as(z).float(), (config.linear_value_head_dim,),
                            w.norm.float(), config.rms_norm_eps)
    gated = (normalized * F.silu(z.float())).to(x.dtype)
    return hf_reference.StaticFP8(w.out_proj)(gated.reshape(-1))


def forward_step(ckpt, token_id, position, state):
    c = ckpt.config
    x = ckpt.embed_tokens[token_id]
    residual = torch.zeros_like(x)
    full = linear_index = 0
    for layer in ckpt.layers:
        combined = x.float() + residual.float()
        normalized = F.rms_norm(combined, (c.hidden_size,), 1 + layer.input_layernorm.float(), c.rms_norm_eps).to(x.dtype)
        residual = combined.to(x.dtype)
        if layer.layer_type == "full_attention":
            mixed = full_attention_block(layer.attn, normalized,
                (state.kv_k[full], state.kv_v[full]), position, config=c)
            full += 1
        else:
            mixed = gated_delta_net_block(layer.attn, normalized, state.rec_state[linear_index],
                state.conv_state[linear_index], position, config=c)
            linear_index += 1
        combined = residual.float() + mixed.float()
        normalized = F.rms_norm(combined, (c.hidden_size,), 1 + layer.post_attention_layernorm.float(), c.rms_norm_eps).to(x.dtype)
        residual = combined.to(x.dtype)
        x = moe_block(layer.moe, normalized, config=c)
    normalized = F.rms_norm(x.float() + residual.float(), (c.hidden_size,), 1 + ckpt.final_norm.float(), c.rms_norm_eps).to(x.dtype)
    return linear(normalized, ckpt.lm_head)


def capture(directory):
    transformers, hf = hf_reference.backend()
    source_hash = hashlib.sha256(Path(inspect.getfile(hf)).read_bytes()).hexdigest()
    source = f"Transformers {transformers.__version__} attention/recurrent core; measured ModelOpt W4A16 arithmetic adapters"
    functions = dict(full_attention_block=full_attention_block,
                     gated_delta_net_block=gated_delta_net_block, moe_block=moe_block, forward_step=forward_step)
    manifest = dict(transformers=transformers.__version__, torch=str(torch.__version__),
                    hf_source_sha256=source_hash, adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    arithmetic="late/joint NVFP4 scaling, BF16 conv products, FP32 beta/gate, FP32 residual add/norm, FP32 RoPE/attention",
                    cases=[])
    directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name, function, kwargs in hf_reference.cases(forward=forward_step):
            before = pack(kwargs)
            expected = functions[function](**kwargs)
            names = {"full_attention_block": ("kv_cache",), "gated_delta_net_block": ("rec_state", "conv_state"),
                     "forward_step": ("state",)}.get(function, ())
            after = {key: kwargs[key] for key in names}
            assert expected.dtype == torch.bfloat16 and torch.isfinite(expected).all()
            save_reference(directory / f"{name}.pt", function=function, kwargs=unpack(before, "cpu"),
                           expected=expected, after=after, source=source)
            manifest["cases"].append(dict(file=f"{name}.pt", function=function))
            print(f"Saved {name}: {function}", flush=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
