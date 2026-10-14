"""FP8 and packed NVFP4 matrix-vector and matrix-matrix kernels."""

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
def _nvfp4_kernel(X, W, S, GlobalScale, Y, Ids, Routing,
                  M: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr,
                  ROWS: tl.constexpr, PAIRS: tl.constexpr, BANK: tl.constexpr,
                  PER_SLOT_INPUT: tl.constexpr, ROUTED: tl.constexpr, GLOBAL_BANK: tl.constexpr):
    slot = tl.program_id(1)
    expert = tl.load(Ids + slot).to(tl.int32) if BANK else 0
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    pairs = tl.arange(0, PAIRS)
    packed = tl.load(W + expert * M * (K // 2) + rows[:, None] * (K // 2) + pairs[None, :],
                     (rows[:, None] < M) & (pairs[None, :] < K // 2), other=0)
    scales = tl.load(S + expert * M * (K // GROUP) + rows[:, None] * (K // GROUP) + (pairs[None, :] * 2 // GROUP),
                     (rows[:, None] < M) & (pairs[None, :] < K // 2), other=0.0).to(tl.float32)
    global_scale = tl.load(GlobalScale + (expert if GLOBAL_BANK else 0)).to(tl.float32)
    even = _fp4(packed & 15) * scales
    odd = _fp4(packed >> 4) * scales
    input_offset = slot * K if PER_SLOT_INPUT else 0
    xe = tl.load(X + input_offset + pairs * 2, pairs * 2 < K, other=0).to(tl.float32)
    xo = tl.load(X + input_offset + pairs * 2 + 1, pairs * 2 + 1 < K, other=0).to(tl.float32)
    result = tl.sum(even * xe[None, :] + odd * xo[None, :], axis=1)
    if ROUTED:
        global_scale = (global_scale * tl.load(Routing + slot)).to(tl.bfloat16).to(tl.float32)
        result = result.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + slot * M + rows, result * global_scale, rows < M)


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


def nvfp4_gemv(x: torch.Tensor, proj: NVFP4Linear, *, routing_weight=None) -> torch.Tensor:
    if x.ndim == 2:
        assert routing_weight is None
        return matmul(x, proj)
    assert (x.is_cuda or triton.knobs.runtime.interpret) and x.dtype == torch.bfloat16 and x.ndim == 1 and x.is_contiguous()
    assert proj.weight.ndim == 2 and proj.weight.is_contiguous()
    assert x.numel() == proj.in_features
    assert proj.weight_scale.is_contiguous() and proj.weight_scale_2.numel() == 1
    assert proj.in_features % proj.group_size == 0 and proj.group_size % 2 == 0
    out = torch.empty(proj.out_features, device=x.device, dtype=x.dtype)
    _nvfp4_kernel[(triton.cdiv(proj.out_features, 4),)](
        x, proj.weight, proj.weight_scale, proj.weight_scale_2, out,
        None, routing_weight,
        proj.out_features, proj.in_features, proj.group_size, 4,
        triton.next_power_of_2(proj.in_features // 2), False, False, routing_weight is not None, False,
        num_warps=4, enable_fp_fusion=False)
    return out


def expert_gemv(x: torch.Tensor, proj: NVFP4Linear, ids: torch.Tensor, weights=None) -> torch.Tensor:
    assert (x.is_cuda or triton.knobs.runtime.interpret) and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert proj.weight.ndim == 3 and proj.weight.is_contiguous() and proj.weight_scale.is_contiguous()
    assert x.shape[-1] == proj.in_features and x.ndim in (1, 2)
    assert x.ndim == 1 or x.shape[0] == ids.numel()
    assert (ids.is_cuda or triton.knobs.runtime.interpret) and ids.ndim == 1 and ids.is_contiguous()
    assert weights is None or (weights.shape == ids.shape and weights.dtype == torch.float32 and weights.device == x.device)
    assert proj.weight_scale_2.numel() in (1, proj.weight.shape[0])
    assert proj.in_features % proj.group_size == 0 and proj.group_size % 2 == 0
    out = torch.empty((ids.numel(), proj.out_features), device=x.device, dtype=x.dtype)
    _nvfp4_kernel[(triton.cdiv(proj.out_features, 4), ids.numel())](
        x, proj.weight, proj.weight_scale, proj.weight_scale_2, out, ids, weights,
        proj.out_features, proj.in_features, proj.group_size, 4,
        triton.next_power_of_2(proj.in_features // 2), True, x.ndim == 2, weights is not None,
        proj.weight_scale_2.numel() != 1, num_warps=4, enable_fp_fusion=False)
    return out


def fp8_gemv(x: torch.Tensor, proj: FP8Linear) -> torch.Tensor:
    if x.ndim == 2:
        return matmul(x, proj)
    assert (x.is_cuda or triton.knobs.runtime.interpret) and x.ndim == 1 and x.is_contiguous()
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


@triton.jit
def _matmul_kernel(X, W, S, Global, InputScale, Y, M: tl.constexpr, N: tl.constexpr,
                   K: tl.constexpr, GROUP: tl.constexpr, FP8: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + inner
        x = tl.load(X + rows[:, None] * K + kk[None, :],
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        if FP8:
            input_scale = tl.load(InputScale).to(tl.float32)
            x = tl.minimum(tl.maximum(tl.div_rn(x.to(tl.float32), input_scale), -448.), 448.)
            x = x.to(tl.float8e4nv).to(tl.float32) * input_scale
            weight = tl.load(W + columns[None, :] * K + kk[:, None],
                             (columns[None, :] < N) & (kk[:, None] < K), other=0.0).to(tl.float32)
            weight *= tl.load(Global).to(tl.float32)
            acc += tl.dot(x, weight, input_precision="tf32x3")
        else:
            packed = tl.load(W + columns[None, :] * (K // 2) + kk[:, None] // 2,
                             (columns[None, :] < N) & (kk[:, None] < K), other=0)
            code = (packed >> ((kk[:, None] % 2) * 4)) & 15
            scale = tl.load(S + columns[None, :] * (K // GROUP) + kk[:, None] // GROUP,
                            (columns[None, :] < N) & (kk[:, None] < K), other=0.0).to(tl.float32)
            weight = (_fp4(code) * scale).to(tl.bfloat16)
            acc += tl.dot(x, weight)
    if not FP8:
        acc *= tl.load(Global).to(tl.float32)
    tl.store(Y + rows[:, None] * N + columns[None, :], acc,
              (rows[:, None] < M) & (columns[None, :] < N))


def matmul(x, proj):
    assert x.ndim == 2 and x.is_contiguous() and x.dtype == torch.bfloat16
    assert x.shape[1] == proj.in_features and proj.weight.ndim == 2
    rows, width = x.shape
    fp8 = isinstance(proj, FP8Linear)
    out = torch.empty((rows, proj.out_features), device=x.device, dtype=x.dtype)
    _matmul_kernel[(triton.cdiv(rows, 16), triton.cdiv(proj.out_features, 32))](
        x, proj.weight, None if fp8 else proj.weight_scale,
        proj.weight_scale if fp8 else proj.weight_scale_2, proj.input_scale, out,
        rows, proj.out_features, width, 16 if fp8 else proj.group_size, fp8,
        16, 32, 64, num_warps=4, enable_fp_fusion=False)
    return out
