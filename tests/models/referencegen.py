"""Generate synthetic CPU fixtures from Transformers and independent format adapters."""

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from quickdraw.models.weights import (
    Checkpoint, FP8Linear, FullAttnWeights, GDNWeights, LayerWeights,
    ModelConfig, MoEWeights, NVFP4Linear,
)
from quickdraw.kvcache.state import DecodeState
from tests.models.mathcheck import pack, save_reference, unpack

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "references" / "model"


def backend():
    import transformers
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
    return transformers, hf


def hf_config(config):
    _, hf = backend()
    values = asdict(config)
    theta = values.pop("rope_theta")
    partial = values.pop("partial_rotary_factor")
    values.pop("full_attention_interval")
    rotary_dim = int(config.head_dim * partial)
    result = hf.Qwen3_5MoeTextConfig(
        **values, max_position_embeddings=64, partial_rotary_factor=partial,
        rope_parameters={"rope_type": "default", "rope_theta": theta,
                         "partial_rotary_factor": partial,
                         "mrope_section": [rotary_dim // 2, 0, 0],
                         "mrope_interleaved": True},
    )
    result._attn_implementation = "eager"
    result._experts_implementation = "grouped_mm"
    return result


def decode_nvfp4(proj):
    """Independent W4A16 decoder: low nibble is even input; one E4M3 scale per 16 inputs."""
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    codes = torch.stack((proj.weight.long() % 16, proj.weight.long() // 16), dim=-1)
    values = lut[codes].flatten(-2)
    scales = proj.weight_scale.float().repeat_interleave(proj.group_size, dim=-1)
    global_scale = proj.weight_scale_2.float()
    if global_scale.ndim:
        global_scale = global_scale.reshape(-1, 1, 1)
    return (values * scales * global_scale).to(torch.bfloat16)


class StaticFP8(nn.Module):
    """Use the existing FP8 format contract with independent Torch operations."""

    def __init__(self, proj):
        super().__init__()
        self.register_buffer("weight", proj.weight.float() * proj.weight_scale.float())
        self.register_buffer("input_scale", proj.input_scale.float().clone())

    def forward(self, x):
        quantized = (x.float() / self.input_scale).clamp(-448., 448.).to(torch.float8_e4m3fn)
        activation = quantized.float() * self.input_scale
        return F.linear(activation, self.weight).to(x.dtype)


def copy_parameter(target, value):
    with torch.no_grad():
        target.copy_(value)


def load_attention(module, weights):
    if isinstance(weights, FullAttnWeights):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(module, name, StaticFP8(getattr(weights, name)))
        copy_parameter(module.q_norm.weight, weights.q_norm)
        copy_parameter(module.k_norm.weight, weights.k_norm)
    else:
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
            setattr(module, name, StaticFP8(getattr(weights, name)))
        for name in ("in_proj_a", "in_proj_b"):
            copy_parameter(getattr(module, name).weight, getattr(weights, name))
        copy_parameter(module.conv1d.weight, weights.conv1d)
        copy_parameter(module.A_log, weights.A_log)
        copy_parameter(module.dt_bias, weights.dt_bias)
        copy_parameter(module.norm.weight, weights.norm)


def load_moe(module, weights):
    copy_parameter(module.gate.weight, weights.router)
    gate_up = torch.cat((decode_nvfp4(weights.experts_gate),
                         decode_nvfp4(weights.experts_up)), dim=1)
    copy_parameter(module.experts.gate_up_proj, gate_up)
    copy_parameter(module.experts.down_proj, decode_nvfp4(weights.experts_down))
    for name, proj in (("gate_proj", weights.shared_gate), ("up_proj", weights.shared_up),
                       ("down_proj", weights.shared_down)):
        copy_parameter(getattr(module.shared_expert, name).weight, decode_nvfp4(proj))
    copy_parameter(module.shared_expert_gate.weight, weights.shared_expert_gate)


class ReferenceCache:
    """Map quickdraw history onto the HF single-token cache interface.
    Kernel-width-minus-one convolution history selects the recurrent fallback."""

    def __init__(self, config, position, full=None, linear=None):
        self.position = position
        self.full = full or {}
        self.linear = linear or {}
        self.layers = [SimpleNamespace(record_past=False) for _ in config.layer_types]
        for idx, (rec, conv) in self.linear.items():
            self.layers[idx].recurrent_states = {0: rec.unsqueeze(0)}
            self.layers[idx].conv_states = {0: conv.unsqueeze(0)}

    def has_previous_state(self, layer_idx, state_idx=0):
        return layer_idx in self.linear

    def update(self, keys, values, layer_idx, *args, **kwargs):
        k, v = self.full[layer_idx]
        k[self.position].copy_(keys[0, :, 0])
        v[self.position].copy_(values[0, :, 0])
        stop = self.position + 1
        return (k[:stop].permute(1, 0, 2).unsqueeze(0),
                v[:stop].permute(1, 0, 2).unsqueeze(0))

    def update_recurrent_state(self, state, layer_idx, *args, **kwargs):
        self.linear[layer_idx][0].copy_(state[0])
        return state


def full_attention_block(w, x, kv_cache, position, *, config):
    _, hf = backend()
    cfg = hf_config(config)
    idx = config.layer_types.index("full_attention")
    module = hf.Qwen3_5MoeAttention(cfg, idx).to(torch.bfloat16).eval()
    load_attention(module, w)
    rotary = hf.Qwen3_5MoeTextRotaryEmbedding(cfg)
    embeddings = rotary(x.reshape(1, 1, -1), torch.full((3, 1, 1), position, dtype=torch.long))
    cache = ReferenceCache(config, position, full={idx: kv_cache})
    return module(x.reshape(1, 1, -1), embeddings, None, past_key_values=cache)[0].reshape(-1)


def gated_delta_net_block(w, x, rec_state, conv_state, position, *, config):
    _, hf = backend()
    idx = config.layer_types.index("linear_attention")
    module = hf.Qwen3_5MoeGatedDeltaNet(hf_config(config), idx).to(torch.bfloat16).eval()
    load_attention(module, w)
    cache = ReferenceCache(config, position, linear={idx: (rec_state, conv_state)})
    return module(x.reshape(1, 1, -1), cache_params=cache).reshape(-1)


def moe_block(w, x, *, config):
    _, hf = backend()
    module = hf.Qwen3_5MoeSparseMoeBlock(hf_config(config)).to(torch.bfloat16).eval()
    load_moe(module, w)
    return module(x.reshape(1, 1, -1)).reshape(-1)


def text_model(ckpt):
    _, hf = backend()
    module = hf.Qwen3_5MoeTextModel(hf_config(ckpt.config)).to(torch.bfloat16).eval()
    # Rebuild RoPE after .to(BF16), which also rounds its frequency buffers.
    module.rotary_emb = hf.Qwen3_5MoeTextRotaryEmbedding(module.config)
    copy_parameter(module.embed_tokens.weight, ckpt.embed_tokens)
    copy_parameter(module.norm.weight, ckpt.final_norm)
    for layer, weights in zip(module.layers, ckpt.layers, strict=True):
        copy_parameter(layer.input_layernorm.weight, weights.input_layernorm)
        copy_parameter(layer.post_attention_layernorm.weight, weights.post_attention_layernorm)
        attn = layer.self_attn if weights.layer_type == "full_attention" else layer.linear_attn
        load_attention(attn, weights.attn)
        load_moe(layer.mlp, weights.moe)
    return module


def state_cache(config, position, state):
    full, linear = {}, {}
    fidx = lidx = 0
    for idx, kind in enumerate(config.layer_types):
        if kind == "full_attention":
            full[idx] = (state.kv_k[fidx], state.kv_v[fidx])
            fidx += 1
        else:
            linear[idx] = (state.rec_state[lidx], state.conv_state[lidx])
            lidx += 1
    return ReferenceCache(config, position, full, linear)


def forward_step(ckpt, token_id, position, state):
    module = text_model(ckpt)
    cache = state_cache(ckpt.config, position, state)
    hidden = module(input_ids=token_id.reshape(1, 1), position_ids=torch.tensor([[position]]),
                    attention_mask={"full_attention": None, "linear_attention": None},
                    past_key_values=cache, use_cache=True).last_hidden_state
    return F.linear(hidden, decode_nvfp4(ckpt.lm_head)).reshape(-1)


FUNCTIONS = {fn.__name__: fn for fn in (
    full_attention_block, gated_delta_net_block, moe_block, forward_step,
)}


def small_config(variant=0):
    return ModelConfig(
        hidden_size=32, num_hidden_layers=4,
        layer_types=(["linear_attention", "full_attention"] * 2 if variant == 0
                     else ["full_attention", "linear_attention"] * 2),
        full_attention_interval=2, num_attention_heads=4 if variant == 0 else 2,
        num_key_value_heads=2 if variant == 0 else 1, head_dim=8,
        linear_num_key_heads=2 if variant == 0 else 1,
        linear_key_head_dim=8 if variant == 0 else 4,
        linear_num_value_heads=4 if variant == 0 else 2, linear_value_head_dim=8,
        linear_conv_kernel_dim=4 if variant == 0 else 2,
        num_experts=4 if variant == 0 else 6, num_experts_per_tok=2 if variant == 0 else 3,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        rms_norm_eps=1e-6 if variant == 0 else .01,
        partial_rotary_factor=.5 if variant == 0 else 1., rope_theta=10000. if variant == 0 else 100.,
        vocab_size=19,
    )


def random_bf16(*shape, scale=.5):
    return (torch.randn(shape) * scale).to(torch.bfloat16)


def random_fp8(out, inp):
    return FP8Linear((torch.randn(out, inp) * .5).to(torch.float8_e4m3fn),
                     torch.tensor(.125), torch.tensor(.25))


def random_fp4(out, inp, experts=None):
    prefix = () if experts is None else (experts,)
    packed = torch.randint(0, 256, (*prefix, out, inp // 2), dtype=torch.uint8)
    scales = (torch.randint(1, 5, (*prefix, out, inp // 16)).float() / 4).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(.25) if experts is None else torch.arange(2, 2 + experts).float() / 16
    calibration = torch.tensor(3.) if experts is None else torch.arange(2, 2 + experts).float()
    return NVFP4Linear(packed, scales, global_scale, calibration)


def random_moe(c):
    h, m, s, e = c.hidden_size, c.moe_intermediate_size, c.shared_expert_intermediate_size, c.num_experts
    return MoEWeights(random_bf16(e, h), random_bf16(1, h),
                      random_fp4(m, h, e), random_fp4(m, h, e), random_fp4(h, m, e),
                      random_fp4(s, h), random_fp4(s, h), random_fp4(h, s))


def random_attention(c, kind):
    h = c.hidden_size
    if kind == "full_attention":
        q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
        return FullAttnWeights(random_fp8(2 * q, h), random_fp8(kv, h), random_fp8(kv, h),
                               random_fp8(h, q), random_bf16(c.head_dim, scale=.2),
                               random_bf16(c.head_dim, scale=.2))
    k = c.linear_num_key_heads * c.linear_key_head_dim
    v = c.linear_num_value_heads * c.linear_value_head_dim
    return GDNWeights(
        random_fp8(2 * k + v, h), random_fp8(v, h),
        random_bf16(c.linear_num_value_heads, h, scale=.1),
        random_bf16(c.linear_num_value_heads, h, scale=.1),
        random_bf16(2 * k + v, 1, c.linear_conv_kernel_dim),
        random_bf16(c.linear_num_value_heads), random_bf16(c.linear_num_value_heads),
        (1 + random_bf16(c.linear_value_head_dim, scale=.2)).to(torch.bfloat16), random_fp8(h, v),
    )


def random_checkpoint(c):
    layers = [LayerWeights(idx, kind, random_bf16(c.hidden_size, scale=.2),
                           random_bf16(c.hidden_size, scale=.2), random_attention(c, kind), random_moe(c))
              for idx, kind in enumerate(c.layer_types)]
    return Checkpoint(c, random_bf16(c.vocab_size, c.hidden_size),
                      random_bf16(c.hidden_size, scale=.2), random_fp4(c.vocab_size, c.hidden_size), layers)


def empty_state(c, capacity=5):
    f = c.layer_types.count("full_attention")
    l = c.layer_types.count("linear_attention")
    channels = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
    return DecodeState(
        # Poison unused cache slots to detect future reads and stray writes.
        torch.full((f, capacity, c.num_key_value_heads, c.head_dim), 9., dtype=torch.bfloat16),
        torch.full((f, capacity, c.num_key_value_heads, c.head_dim), -11., dtype=torch.bfloat16),
        torch.zeros(l, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim),
        torch.zeros(l, channels, c.linear_conv_kernel_dim - 1, dtype=torch.bfloat16), 0,
    )


def cases():
    for variant, position in ((0, 0), (0, 2), (1, 4)):
        torch.manual_seed(110 + variant * 10 + position)
        c = small_config(variant)
        state = empty_state(c)
        state.kv_k[:, :position] = random_bf16(*state.kv_k[:, :position].shape)
        state.kv_v[:, :position] = random_bf16(*state.kv_v[:, :position].shape)
        yield f"attention-v{variant}-p{position}", "full_attention_block", dict(
            w=random_attention(c, "full_attention"), x=random_bf16(c.hidden_size),
            kv_cache=(state.kv_k[0], state.kv_v[0]), position=position, config=c)

    for variant, position in ((0, 0), (0, 3), (1, 4)):
        torch.manual_seed(210 + variant * 10 + position)
        c = small_config(variant)
        state = empty_state(c)
        if position:
            state.rec_state.copy_(torch.randn_like(state.rec_state) * .2)
            state.conv_state.copy_(random_bf16(*state.conv_state.shape))
        yield f"delta-v{variant}-p{position}", "gated_delta_net_block", dict(
            w=random_attention(c, "linear_attention"), x=random_bf16(c.hidden_size),
            rec_state=state.rec_state[0], conv_state=state.conv_state[0], position=position, config=c)

    for variant, top_k in ((0, 1), (0, 2), (1, 3)):
        torch.manual_seed(310 + variant * 10 + top_k)
        c = small_config(variant)
        c.num_experts_per_tok = top_k
        yield f"moe-v{variant}-top{top_k}", "moe_block", dict(
            w=random_moe(c), x=random_bf16(c.hidden_size), config=c)

    for variant in (0, 1):
        torch.manual_seed(410 + variant)
        c = small_config(variant)
        ckpt = random_checkpoint(c)
        state = empty_state(c)
        for position, token in enumerate((3, 7, 5, 11)):
            kwargs = dict(ckpt=ckpt, token_id=torch.tensor(token), position=position, state=state)
            if position in (0, 1, 3):
                yield f"forward-v{variant}-p{position}", "forward_step", kwargs
            else:
                forward_step(**kwargs)
            state.position = position + 1


def validate_native_cache():
    """Check adapter outputs/state against HF's normal DynamicCache."""
    _, hf = backend()
    for variant in (0, 1):
        torch.manual_seed(410 + variant)
        c = small_config(variant)
        ckpt = random_checkpoint(c)
        module = text_model(ckpt)
        cache = hf.DynamicCache(config=module.config)
        state = empty_state(c)
        for pos, token in enumerate((3, 7, 5, 11)):
            token_id = torch.tensor(token)
            actual = forward_step(ckpt, token_id, pos, state)
            hidden = module(
                input_ids=token_id.reshape(1, 1), position_ids=torch.tensor([[pos]]),
                past_key_values=cache, use_cache=True,
                attention_mask={"full_attention": None, "linear_attention": None},
            ).last_hidden_state
            expected = F.linear(hidden, decode_nvfp4(ckpt.lm_head)).reshape(-1)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
            fidx = lidx = 0
            for idx, kind in enumerate(c.layer_types):
                layer = cache.layers[idx]
                if kind == "full_attention":
                    for target, source in ((state.kv_k, layer.keys), (state.kv_v, layer.values)):
                        torch.testing.assert_close(target[fidx, :pos + 1], source[0].transpose(0, 1), rtol=0, atol=0)
                    fidx += 1
                else:
                    torch.testing.assert_close(state.rec_state[lidx], layer.recurrent_states[0][0], rtol=2e-5, atol=1e-6)
                    tail = layer.conv_states[0][0, ..., -(c.linear_conv_kernel_dim - 1):]
                    torch.testing.assert_close(state.conv_state[lidx], tail, rtol=0, atol=0)
                    lidx += 1
            assert state.position == pos, "model math must not advance the engine counter"
            state.position = pos + 1
        print(f"Validated native HF cache against adapter: variant {variant}, four positions")


def capture(directory):
    transformers, hf = backend()
    source_path = Path(inspect.getfile(hf))
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = (f"Transformers {transformers.__version__} Qwen3_5Moe CPU eager attention / grouped_mm experts; "
              f"static FP8 / weight-only NVFP4 adapters; HF source SHA256 {source_hash}")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"transformers": transformers.__version__, "torch": torch.__version__,
                "attention_backend": "eager", "expert_backend": "grouped_mm",
                "rotary_frequency_dtype": "torch.float32",
                "hf_source_sha256": source_hash,
                "hf_source": "https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cache_validation": "native HF DynamicCache, two layer orders, four positions each",
                "cases": []}
    with torch.inference_mode():
        validate_native_cache()
        for name, function, kwargs in cases():
            # Snapshot inputs before HF mutates history.
            before = pack(kwargs)
            expected = FUNCTIONS[function](**kwargs)
            if function == "full_attention_block":
                after = {"kv_cache": kwargs["kv_cache"]}
            elif function == "gated_delta_net_block":
                after = {"rec_state": kwargs["rec_state"], "conv_state": kwargs["conv_state"]}
            elif function == "forward_step":
                after = {"state": kwargs["state"]}
            else:
                after = {}
            assert expected.shape == (kwargs["ckpt"].config.vocab_size if function == "forward_step"
                                       else kwargs["config"].hidden_size,)
            assert expected.dtype == torch.bfloat16 and torch.isfinite(expected).all()
            assert expected.abs().max() > .01, "a near-zero reference cannot reject missing math"
            path = directory / f"{name}.pt"
            save_reference(path, function=function, kwargs=unpack(before, "cpu"),
                           expected=expected, after=after, source=source)
            manifest["cases"].append({"file": path.name, "function": function,
                                      "output_shape": list(expected.shape),
                                      "output_max_abs": expected.abs().max().item()})
            print(f"Saved {path.name}: {function}")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    capture(args.output)


if __name__ == "__main__":
    main()
