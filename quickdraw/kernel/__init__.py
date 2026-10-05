"""Bind plain kernel functions once before inference."""

from functools import partial


def bind_kernels(**overrides):
    """Bind projection overrides into reference blocks and the token step.
    Whole-block overrides own their internal operations."""
    from ..models import qwen

    names = ("rms_norm", "fp8_gemv", "nvfp4_gemv", "full_attention_block",
             "gated_delta_net_block", "moe_block")
    assert overrides.keys() <= set(names), "unknown kernel operation"
    ops = {name: getattr(qwen, name) for name in names} | overrides
    if not overrides:
        return ops | {"forward_step": qwen.forward_step}
    for name in ("full_attention_block", "gated_delta_net_block"):
        if name not in overrides:
            ops[name] = partial(ops[name], project_fp8=ops["fp8_gemv"])
    if "moe_block" not in overrides:
        ops["moe_block"] = partial(ops["moe_block"], project_fp4=ops["nvfp4_gemv"])
    ops["forward_step"] = partial(
        qwen.forward_step, normalize=ops["rms_norm"],
        attention=ops["full_attention_block"], delta_net=ops["gated_delta_net_block"],
        experts=ops["moe_block"], project_fp4=ops["nvfp4_gemv"],
    )
    return ops
