"""Warp-per-row W4A16 decode with directly indexed packed NVFP4 expert banks.

No activation requantization or expanded weight buffers. Gate/up can share one
input read and write SiLU(gate)*up directly. FP32 accumulation and the BF16
rounding boundaries follow models/qwen.py (including routed down's two rounds).
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

_compiled = {}


@cute.jit
def _fp4(code):
    # E2M1 table expressed without divergent branches or a global lookup.
    # Doubled magnitudes [0,1,2,3,4,6,8,12], all in one register.
    magnitude = Float32((Uint32(0xC8643210) >> ((code & 7) * 4)) & 15) * 0.5
    return -magnitude if (code & 8) != 0 else magnitude


@cute.kernel
def _nvfp4(X, W, S, G, Y, Ids, Routing, UpW, UpS, UpG,
           M: cutlass.Constexpr, K: cutlass.Constexpr,
           GROUP: cutlass.Constexpr, WARPS: cutlass.Constexpr,
           BANK: cutlass.Constexpr, SLOT_INPUT: cutlass.Constexpr,
           ROUTED: cutlass.Constexpr, GLOBAL_BANK: cutlass.Constexpr,
           FUSED_UP: cutlass.Constexpr, VEC: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, slot, _ = cute.arch.block_idx()
    lane = tid % 32
    row = block * WARPS + tid // 32
    expert = Int32(0)
    if cutlass.const_expr(BANK):
        expert = Int32(Ids[slot])
    acc = Float32(0)
    up_acc = Float32(0)
    if row < M:
        for tile in cutlass.range_constexpr(cute.ceil_div(K // 2, 32 * VEC)):
            pair = (tile * 32 + lane) * VEC
            if pair < K // 2:
                offset = (expert * M + row) * (K // (2 * VEC)) + pair // VEC
                scale_offset = (expert * M + row) * (K // GROUP) + pair * 2 // GROUP
                word = Uint32(W[offset])
                scale = Float32(S[scale_offset])
                input_offset = Int32(0)
                if cutlass.const_expr(SLOT_INPUT):
                    input_offset = slot * K
                if cutlass.const_expr(FUSED_UP):
                    up_word = Uint32(UpW[offset])
                    up_scale = Float32(UpS[scale_offset])
                for byte in cutlass.range_constexpr(VEC):
                    packed = Int32((word >> (byte * 8)) & 255)
                    xe = Float32(X[input_offset + (pair + byte) * 2])
                    xo = Float32(X[input_offset + (pair + byte) * 2 + 1])
                    acc += (_fp4(packed & 15) * scale) * xe + (_fp4(packed >> 4) * scale) * xo
                    if cutlass.const_expr(FUSED_UP):
                        up_packed = Int32((up_word >> (byte * 8)) & 255)
                        up_acc += (_fp4(up_packed & 15) * up_scale) * xe + (_fp4(up_packed >> 4) * up_scale) * xo
        acc = cute.arch.warp_reduction_sum(acc)
        scale_index = Int32(0)
        if cutlass.const_expr(GLOBAL_BANK):
            scale_index = expert
        global_scale = Float32(G[scale_index])
        if cutlass.const_expr(ROUTED):
            global_scale = Float32(BFloat16(global_scale * Float32(Routing[slot])))
            acc = Float32(BFloat16(acc))
        value = Float32(BFloat16(acc * global_scale))
        if cutlass.const_expr(FUSED_UP):
            up_acc = cute.arch.warp_reduction_sum(up_acc)
            up = Float32(BFloat16(up_acc * Float32(UpG[scale_index])))
            activated = Float32(BFloat16(value / (1.0 + cute.math.exp(-value))))
            value = activated * up
        if lane == 0:
            Y[slot * M + row] = BFloat16(value)


@cute.jit
def _launch_nvfp4(X, W, S, G, Y, Ids, Routing, UpW, UpS, UpG,
                  M: cutlass.Constexpr, K: cutlass.Constexpr,
                  GROUP: cutlass.Constexpr, WARPS: cutlass.Constexpr,
                  SLOTS: cutlass.Constexpr, BANK: cutlass.Constexpr,
                  SLOT_INPUT: cutlass.Constexpr, ROUTED: cutlass.Constexpr,
                  GLOBAL_BANK: cutlass.Constexpr, FUSED_UP: cutlass.Constexpr, VEC: cutlass.Constexpr,
                  stream: cuda.CUstream):
    _nvfp4(X, W, S, G, Y, Ids, Routing, UpW, UpS, UpG, M, K, GROUP,
           WARPS, BANK, SLOT_INPUT, ROUTED, GLOBAL_BANK, FUSED_UP, VEC).launch(
               grid=(cute.ceil_div(M, WARPS), SLOTS, 1),
               block=(WARPS * 32, 1, 1), stream=stream)


def _tensor(tensor):
    return None if tensor is None else from_dlpack(tensor.reshape(-1))


def _check(x, proj):
    assert x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.ndim in (1, 2) and x.shape[-1] == proj.in_features
    assert proj.weight.is_contiguous() and proj.weight.dtype == torch.uint8
    assert proj.weight_scale.is_contiguous() and proj.weight_scale.dtype == torch.float8_e4m3fn
    assert proj.weight_scale_2.dtype == torch.float32
    assert proj.in_features % proj.group_size == 0 and proj.group_size % 2 == 0
    assert proj.weight_scale.shape == (*proj.weight.shape[:-1], proj.in_features // proj.group_size)
    assert x.device == proj.weight.device == proj.weight_scale.device == proj.weight_scale_2.device


def _run(x, proj, *, ids=None, routing=None, up=None, warps=4):
    _check(x, proj)
    bank = ids is not None
    assert proj.weight.ndim == (3 if bank else 2)
    slots = ids.numel() if bank else 1
    assert x.ndim == 1 or (bank and x.shape[0] == slots)
    assert proj.weight_scale_2.numel() in (1, proj.weight.shape[0] if bank else 1)
    if bank:
        assert ids.device == x.device and ids.dtype == torch.int32 and ids.ndim == 1 and ids.is_contiguous()
    if routing is not None:
        assert routing.device == x.device and routing.dtype == torch.float32 and routing.numel() == slots and routing.is_contiguous()
    if up is not None:
        _check(x, up)
        assert up.weight.shape == proj.weight.shape
        assert up.group_size == proj.group_size
        assert up.weight_scale_2.shape == proj.weight_scale_2.shape
        assert routing is None
    out = torch.empty((slots, proj.out_features) if bank else (proj.out_features,), device=x.device, dtype=x.dtype)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    vector = 4 if proj.group_size % 8 == 0 else 1
    packed_weight = proj.weight.view(torch.int32) if vector == 4 else proj.weight
    packed_up = (up.weight.view(torch.int32) if vector == 4 else up.weight) if up else None
    args = (x, packed_weight, proj.weight_scale, proj.weight_scale_2, out, ids, routing,
            packed_up, up.weight_scale if up else None, up.weight_scale_2 if up else None)
    constants = (proj.out_features, proj.in_features, proj.group_size, warps,
                 slots, bank, x.ndim == 2, routing is not None,
                 proj.weight_scale_2.numel() != 1, up is not None, vector)
    key = (x.device, constants, tuple(None if t is None else (t.numel(), t.dtype) for t in args))
    tensors = tuple(map(_tensor, args))
    if key not in _compiled:
        _compiled[key] = cute.compile(_launch_nvfp4, *tensors, *constants, stream)
    _compiled[key](*tensors, stream)
    return out


def nvfp4_gemv(x, proj, *, routing_weight=None):
    """Single BF16 vector times a packed NVFP4 matrix; no prefill fallback."""
    assert x.ndim == 1
    return _run(x, proj, routing=routing_weight)


def expert_gemv(x, proj, ids, weights=None):
    """Read selected experts directly, with optional per-slot inputs and routing."""
    return _run(x, proj, ids=ids, routing=weights)


def gate_up(x, gate, up, ids=None):
    """Fuse gate and up projections, BF16 SiLU, and BF16 multiplication."""
    return _run(x, gate, ids=ids, up=up)
