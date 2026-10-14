"""RMS normalization over contiguous rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X, W, Out, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    features = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + features, features < N, other=0).to(tl.float32)
    weight = tl.load(W + features, features < N, other=0).to(tl.float32)
    normalized = x * tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    tl.store(Out + row * N + features, normalized * weight, features < N)


def rms_norm(x, weight, eps):
    assert (x.is_cuda or triton.knobs.runtime.interpret) and x.is_contiguous() and weight.is_contiguous()
    output = torch.empty_like(x)
    width = x.shape[-1]
    _rms_kernel[(x.numel() // width,)](x, weight, output, width, eps, triton.next_power_of_2(width), enable_fp_fusion=False)
    return output
