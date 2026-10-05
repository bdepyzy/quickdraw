"""Batch-one FP8 and packed NVFP4 matrix-vector kernels."""

import torch
import triton
import triton.language as tl

from ...models.weights import FP8Linear, NVFP4Linear


@triton.jit
def _fp4(codes):
    codes = codes.to(tl.int32)
    exponent = (codes >> 1) & 3
    mantissa = codes & 1
    # Normal E2M1 magnitudes are (1 + mantissa/2) * 2**(exponent-1).
    power = ((exponent + 126) << 23).to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, mantissa * .5,
                         power * (1.0 + mantissa * .5))
    return tl.where((codes & 8) != 0, -magnitude, magnitude)


@triton.jit
def _nvfp4_kernel(X, W, S, GlobalScale, Y, M: tl.constexpr, K: tl.constexpr,
                  GROUP: tl.constexpr, ROWS: tl.constexpr, PAIRS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    pairs = tl.arange(0, PAIRS)
    packed = tl.load(W + rows[:, None] * (K // 2) + pairs[None, :],
                     (rows[:, None] < M) & (pairs[None, :] < K // 2), other=0)
    scales = tl.load(S + rows[:, None] * (K // GROUP) + (pairs[None, :] * 2 // GROUP),
                     (rows[:, None] < M) & (pairs[None, :] < K // 2), other=0.0).to(tl.float32)
    global_scale = tl.load(GlobalScale).to(tl.float32)
    # Round dequantized weights to BF16 before multiplication.
    even = (_fp4(packed & 15) * scales * global_scale).to(tl.bfloat16).to(tl.float32)
    odd = (_fp4(packed >> 4) * scales * global_scale).to(tl.bfloat16).to(tl.float32)
    xe = tl.load(X + pairs * 2, pairs * 2 < K, other=0).to(tl.float32)
    xo = tl.load(X + pairs * 2 + 1, pairs * 2 + 1 < K, other=0).to(tl.float32)
    result = tl.sum(even * xe[None, :] + odd * xo[None, :], axis=1)
    tl.store(Y + rows, result, rows < M)


@triton.jit
def _fp8_kernel(X, W, WeightScale, InputScale, Y, M: tl.constexpr, K: tl.constexpr,
                ROWS: tl.constexpr, WIDTH: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    columns = tl.arange(0, WIDTH)
    input_scale = tl.load(InputScale).to(tl.float32)
    x = tl.load(X + columns, columns < K, other=0).to(tl.float32)
    # Static FP8 activation scale; saturate before conversion.
    q = tl.minimum(tl.maximum(tl.div_rn(x, input_scale), -448.0), 448.0).to(tl.float8e4nv)
    activation = q.to(tl.float32) * input_scale
    weight = tl.load(W + rows[:, None] * K + columns[None, :],
                     (rows[:, None] < M) & (columns[None, :] < K), other=0.0).to(tl.float32)
    weight = weight * tl.load(WeightScale).to(tl.float32)
    result = tl.sum(weight * activation[None, :], axis=1)
    tl.store(Y + rows, result, rows < M)


def nvfp4_gemv(x: torch.Tensor, proj: NVFP4Linear) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 1 and x.is_contiguous()
    assert proj.weight.ndim == 2 and proj.weight.is_contiguous()
    assert x.numel() == proj.in_features
    assert proj.weight_scale.is_contiguous() and proj.weight_scale_2.numel() == 1
    assert proj.in_features % proj.group_size == 0 and proj.group_size % 2 == 0
    out = torch.empty(proj.out_features, device=x.device, dtype=x.dtype)
    _nvfp4_kernel[(triton.cdiv(proj.out_features, 4),)](
        x, proj.weight, proj.weight_scale, proj.weight_scale_2, out,
        proj.out_features, proj.in_features, proj.group_size, 4,
        triton.next_power_of_2(proj.in_features // 2),
        num_warps=4, enable_fp_fusion=False)
    return out


def fp8_gemv(x: torch.Tensor, proj: FP8Linear) -> torch.Tensor:
    assert x.is_cuda and x.ndim == 1 and x.is_contiguous()
    assert proj.weight.ndim == 2 and proj.weight.is_contiguous()
    assert x.numel() == proj.in_features
    assert proj.input_scale is not None and proj.input_scale.numel() == 1
    assert proj.weight_scale.numel() == 1
    out = torch.empty(proj.out_features, device=x.device, dtype=x.dtype)
    _fp8_kernel[(triton.cdiv(proj.out_features, 4),)](
        x, proj.weight, proj.weight_scale, proj.input_scale, out,
        proj.out_features, proj.in_features, 4, triton.next_power_of_2(proj.in_features),
        num_warps=4, enable_fp_fusion=False)
    return out
