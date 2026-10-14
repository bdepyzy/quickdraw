"""Function bindings for the Triton backend."""


def kernels():
    from .attention import full_attention_block
    from .delta import gated_delta_net_block
    from .matvec import fp8_gemv, nvfp4_gemv
    from .moe import moe_block
    from .norm import rms_norm

    return dict(rms_norm=rms_norm, fp8_gemv=fp8_gemv, nvfp4_gemv=nvfp4_gemv,
                full_attention_block=full_attention_block,
                gated_delta_net_block=gated_delta_net_block, moe_block=moe_block)
