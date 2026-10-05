"""Stream Qwen3.6 modelopt mixed-precision language weights, excluding vision.
NVFP4 W4A16, static FP8, and raw BF16 layouts are in docs/architecture.md."""

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

# e2m1 values for nibbles 0b0000..0b1111 (sign, 2-bit exp, 1-bit mantissa).
_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Decode E2M1: low nibble is even input, high nibble is odd input."""
    lut = _FP4_LUT.to(packed.device)
    lo = lut[(packed & 0x0F).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).flatten(-2)


@dataclass
class NVFP4Linear:
    """W4A16 NVFP4 projection (weight-only FP4; activations stay bf16)."""
    weight: torch.Tensor        # uint8 [out, in//2]
    weight_scale: torch.Tensor  # f8e4m3 [out, in//16]
    weight_scale_2: torch.Tensor  # f32 scalar
    input_scale: torch.Tensor | None = None
    group_size: int = 16

    @property
    def out_features(self) -> int:
        return self.weight.shape[-2]

    @property
    def in_features(self) -> int:
        return self.weight.shape[-1] * 2

    def dequantize(self) -> torch.Tensor:
        """Dequantize to BF16 [..., out, in]."""
        w = unpack_fp4(self.weight)
        s = self.weight_scale.float().repeat_interleave(self.group_size, dim=-1)
        global_scale = self.weight_scale_2.float()
        if global_scale.numel() == 1:
            global_scale = global_scale.reshape(())
        else:
            global_scale = global_scale.reshape(*self.weight.shape[:-2], 1, 1)
        return (w * s * global_scale).to(torch.bfloat16)


@dataclass
class FP8Linear:
    """Static FP8 e4m3 projection (weights and activations quantized)."""
    weight: torch.Tensor        # f8e4m3 [out, in]
    weight_scale: torch.Tensor  # f32 scalar
    input_scale: torch.Tensor | None = None  # f32 scalar; engine's static quant scale

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    def dequantize(self) -> torch.Tensor:
        return (self.weight.to(torch.float32) * self.weight_scale.float()).to(torch.bfloat16)


@dataclass
class GDNWeights:
    """Gated-delta-net linear attention block (30 of 40 layers)."""
    in_proj_qkv: FP8Linear      # 2048 -> 8192 (q 2048 | k 2048 | v 4096)
    in_proj_z: FP8Linear        # 2048 -> 4096 (output gate)
    in_proj_a: torch.Tensor     # bf16 [32, 2048]
    in_proj_b: torch.Tensor     # bf16 [32, 2048]
    conv1d: torch.Tensor        # bf16 [8192, 1, 4] depthwise short conv
    A_log: torch.Tensor         # bf16 [32]
    dt_bias: torch.Tensor       # bf16 [32]
    norm: torch.Tensor          # bf16 [128] per-head RMS norm
    out_proj: FP8Linear         # 4096 -> 2048


@dataclass
class FullAttnWeights:
    """Full attention block (every 4th layer; GQA, gated output)."""
    q_proj: FP8Linear           # 2048 -> 8192; each head stores [q 256 | gate 256]
    k_proj: FP8Linear           # 2048 -> 512  (2 KV heads x 256)
    v_proj: FP8Linear           # 2048 -> 512
    o_proj: FP8Linear           # 4096 -> 2048
    q_norm: torch.Tensor        # bf16 [256] per-head RMS norm
    k_norm: torch.Tensor        # bf16 [256]


@dataclass
class MoEWeights:
    """MoE block: router, 256 stacked NVFP4 experts, NVFP4 shared expert."""
    router: torch.Tensor        # bf16 [256, 2048]
    shared_expert_gate: torch.Tensor  # bf16 [1, 2048] sigmoid gate on the shared expert
    experts_gate: NVFP4Linear   # uint8 [256, 512, 1024]
    experts_up: NVFP4Linear
    experts_down: NVFP4Linear
    shared_gate: NVFP4Linear    # 2048 -> 512
    shared_up: NVFP4Linear      # 2048 -> 512
    shared_down: NVFP4Linear    # 512 -> 2048


@dataclass
class LayerWeights:
    idx: int
    layer_type: str             # 'linear_attention' | 'full_attention'
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    attn: GDNWeights | FullAttnWeights
    moe: MoEWeights


@dataclass
class ModelConfig:
    """Subset of text_config that the engine needs."""
    hidden_size: int
    num_hidden_layers: int
    layer_types: list[str]
    full_attention_interval: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    rms_norm_eps: float
    partial_rotary_factor: float
    rope_theta: float
    vocab_size: int

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        raw = json.loads(Path(path).read_text())["text_config"]
        rope = raw.get("rope_parameters", {})
        return cls(
            hidden_size=raw["hidden_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            layer_types=raw["layer_types"],
            full_attention_interval=raw["full_attention_interval"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"],
            linear_num_key_heads=raw["linear_num_key_heads"],
            linear_key_head_dim=raw["linear_key_head_dim"],
            linear_num_value_heads=raw["linear_num_value_heads"],
            linear_value_head_dim=raw["linear_value_head_dim"],
            linear_conv_kernel_dim=raw["linear_conv_kernel_dim"],
            num_experts=raw["num_experts"],
            num_experts_per_tok=raw["num_experts_per_tok"],
            moe_intermediate_size=raw["moe_intermediate_size"],
            shared_expert_intermediate_size=raw["shared_expert_intermediate_size"],
            rms_norm_eps=raw["rms_norm_eps"],
            partial_rotary_factor=raw.get("partial_rotary_factor", 0.25),
            rope_theta=rope.get("rope_theta", 10000000.0),
            vocab_size=raw["vocab_size"],
        )


@dataclass
class Checkpoint:
    config: ModelConfig
    embed_tokens: torch.Tensor          # bf16 [vocab, 2048]
    final_norm: torch.Tensor            # bf16 [2048]
    lm_head: NVFP4Linear                # NVFP4 2048 -> 248320
    layers: list[LayerWeights]
    mtp: dict[str, torch.Tensor] = field(default_factory=dict)  # bf16


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cuda",
    include_mtp: bool = False,
    layers: list[int] | None = None,
) -> Checkpoint:
    """Stream language weights to the target device, excluding vision.
    layers selects indices; MTP weights require include_mtp."""
    path = Path(path)
    config = ModelConfig.from_json(path / "config.json")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    wanted = set(layers) if layers is not None else set(range(config.num_hidden_layers))

    layer_indices = set()
    mtp_names = []
    keep = set()
    for name in weight_map:
        if name.startswith("model.visual."):
            continue
        match = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
        if match:
            index = int(match.group(1))
            if index not in wanted:
                continue
            layer_indices.add(index)
        elif name.startswith("mtp."):
            if not include_mtp:
                continue
            mtp_names.append(name)
        keep.add(name)

    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(safe_open(path / shard, framework="pt"))
            for shard in sorted({weight_map[name] for name in keep})
        }

        def read(name):
            return handles[weight_map[name]].get_tensor(name)

        P = "model.language_model."

        def t(name: str) -> torch.Tensor:
            return read(name).to(device)

        def scalar(name: str) -> torch.Tensor:
            return read(name).float().to(device)

        def nvfp4(prefix: str) -> NVFP4Linear:
            return NVFP4Linear(
                weight=t(prefix + ".weight"),
                weight_scale=t(prefix + ".weight_scale"),
                weight_scale_2=scalar(prefix + ".weight_scale_2"),
                input_scale=(scalar(prefix + ".input_scale")
                             if prefix + ".input_scale" in weight_map else None),
            )

        def fp8(prefix: str) -> FP8Linear:
            return FP8Linear(
                weight=t(prefix + ".weight"),
                weight_scale=scalar(prefix + ".weight_scale"),
                input_scale=(scalar(prefix + ".input_scale")
                             if prefix + ".input_scale" in weight_map else None),
            )

        def stacked_nvfp4(idx_l: int, proj: str) -> NVFP4Linear:
            base = f"{P}layers.{idx_l}.mlp.experts"

            def packed_stack(suffix):
                # Copy directly into final storage to avoid a second expert stack.
                first = read(f"{base}.0.{proj}_proj.{suffix}")
                result = torch.empty((config.num_experts, *first.shape),
                                     dtype=first.dtype, device=device)
                result[0].copy_(first)
                del first
                for expert in range(1, config.num_experts):
                    result[expert].copy_(read(f"{base}.{expert}.{proj}_proj.{suffix}"))
                return result

            input_names = [f"{base}.{e}.{proj}_proj.input_scale" for e in range(config.num_experts)]
            return NVFP4Linear(
                weight=packed_stack("weight"),
                weight_scale=packed_stack("weight_scale"),
                weight_scale_2=torch.stack([
                    scalar(f"{base}.{e}.{proj}_proj.weight_scale_2")
                    for e in range(config.num_experts)]),
                input_scale=(torch.stack([scalar(name) for name in input_names])
                             if all(name in weight_map for name in input_names) else None),
            )

        layer_list: list[LayerWeights] = []
        for idx_l in sorted(layer_indices):
            lp = f"{P}layers.{idx_l}."
            ltype = config.layer_types[idx_l]
            if ltype == "linear_attention":
                attn: GDNWeights | FullAttnWeights = GDNWeights(
                    in_proj_qkv=fp8(lp + "linear_attn.in_proj_qkv"),
                    in_proj_z=fp8(lp + "linear_attn.in_proj_z"),
                    in_proj_a=t(lp + "linear_attn.in_proj_a.weight"),
                    in_proj_b=t(lp + "linear_attn.in_proj_b.weight"),
                    conv1d=t(lp + "linear_attn.conv1d.weight"),
                    A_log=t(lp + "linear_attn.A_log"),
                    dt_bias=t(lp + "linear_attn.dt_bias"),
                    norm=t(lp + "linear_attn.norm.weight"),
                    out_proj=fp8(lp + "linear_attn.out_proj"),
                )
            else:
                assert ltype == "full_attention"
                attn = FullAttnWeights(
                    q_proj=fp8(lp + "self_attn.q_proj"),
                    k_proj=fp8(lp + "self_attn.k_proj"),
                    v_proj=fp8(lp + "self_attn.v_proj"),
                    o_proj=fp8(lp + "self_attn.o_proj"),
                    q_norm=t(lp + "self_attn.q_norm.weight"),
                    k_norm=t(lp + "self_attn.k_norm.weight"),
                )
            moe = MoEWeights(
                router=t(lp + "mlp.gate.weight"),
                shared_expert_gate=t(lp + "mlp.shared_expert_gate.weight"),
                experts_gate=stacked_nvfp4(idx_l, "gate"),
                experts_up=stacked_nvfp4(idx_l, "up"),
                experts_down=stacked_nvfp4(idx_l, "down"),
                shared_gate=nvfp4(lp + "mlp.shared_expert.gate_proj"),
                shared_up=nvfp4(lp + "mlp.shared_expert.up_proj"),
                shared_down=nvfp4(lp + "mlp.shared_expert.down_proj"),
            )
            layer_list.append(LayerWeights(
                idx=idx_l,
                layer_type=ltype,
                input_layernorm=t(lp + "input_layernorm.weight"),
                post_attention_layernorm=t(lp + "post_attention_layernorm.weight"),
                attn=attn,
                moe=moe,
            ))

        return Checkpoint(
            config=config,
            embed_tokens=t(P + "embed_tokens.weight"),
            final_norm=t(P + "norm.weight"),
            lm_head=nvfp4("lm_head"),
            layers=layer_list,
            mtp={name: t(name) for name in mtp_names},
        )
