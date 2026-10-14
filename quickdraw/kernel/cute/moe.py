"""Single-token CuTe routing, fused packed expert gate/up, and output reduction."""

import torch
import torch.nn.functional as F
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32
from cuda.bindings import driver as cuda

from .matvec import _tensor, expert_gemv, gate_up, nvfp4_gemv

_compiled = {}


@cute.kernel
def _route(Logits, Ids, Weights, E: cutlass.Constexpr, TOPK: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    values = cute.make_rmem_tensor((cute.ceil_div(E, 32),), Float32)
    maximum = Float32(-float("inf"))
    for i in cutlass.range_constexpr(cute.ceil_div(E, 32)):
        index = i * 32 + lane
        value = Float32(-float("inf"))
        if index < E:
            value = Float32(Logits[index])
        values[i] = value
        maximum = value if value > maximum else maximum
    maximum = cute.arch.warp_reduction_max(maximum)
    denominator = Float32(0)
    for i in cutlass.range_constexpr(cute.ceil_div(E, 32)):
        value = cute.math.exp(values[i] - maximum)
        values[i] = value
        denominator += value
    denominator = cute.arch.warp_reduction_sum(denominator)
    for i in cutlass.range_constexpr(cute.ceil_div(E, 32)):
        values[i] = values[i] / denominator
    chosen = cute.make_rmem_tensor((TOPK,), Float32)
    total = Float32(0)
    for slot in cutlass.range_constexpr(TOPK):
        best = Float32(-float("inf"))
        for i in cutlass.range_constexpr(cute.ceil_div(E, 32)):
            best = values[i] if values[i] > best else best
        best = cute.arch.warp_reduction_max(best)
        candidate = Float32(2147483520)
        for i in cutlass.range_constexpr(cute.ceil_div(E, 32)):
            index = i * 32 + lane
            if index < E and values[i] == best:
                candidate = Float32(index) if Float32(index) < candidate else candidate
        expert = Int32(-cute.arch.warp_reduction_max(-candidate))
        for i in cutlass.range_constexpr(cute.ceil_div(E, 32)):
            if i * 32 + lane == expert:
                values[i] = Float32(-float("inf"))
        chosen[slot] = best
        total += best
        if lane == 0:
            Ids[slot] = expert
    if lane == 0:
        for slot in cutlass.range_constexpr(TOPK):
            Weights[slot] = chosen[slot] / total


@cute.kernel
def _reduce(Experts, Shared, Strength, Out, H: cutlass.Constexpr, TOPK: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    feature = block * 256 + tid
    if feature < H:
        routed = Float32(0)
        for slot in cutlass.range_constexpr(TOPK):
            routed += Float32(Experts[slot * H + feature])
        routed = Float32(BFloat16(routed))
        strength = Float32(BFloat16(1.0 / (1.0 + cute.math.exp(-Float32(Strength[0])))))
        shared = Float32(BFloat16(Float32(Shared[feature]) * strength))
        Out[feature] = BFloat16(routed + shared)


@cute.jit
def _launch_route(Logits, Ids, Weights, E: cutlass.Constexpr, TOPK: cutlass.Constexpr, stream: cuda.CUstream):
    _route(Logits, Ids, Weights, E, TOPK).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def _launch_reduce(Experts, Shared, Strength, Out, H: cutlass.Constexpr, TOPK: cutlass.Constexpr, stream: cuda.CUstream):
    _reduce(Experts, Shared, Strength, Out, H, TOPK).launch(
        grid=(cute.ceil_div(H, 256), 1, 1), block=(256, 1, 1), stream=stream)


def route(logits, topk):
    assert logits.is_cuda and logits.is_contiguous() and logits.ndim == 1
    assert 1 <= topk <= logits.numel()
    ids = torch.empty(topk, device=logits.device, dtype=torch.int32)
    weights = torch.empty(topk, device=logits.device, dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream)
    tensors = tuple(map(_tensor, (logits, ids, weights)))
    key = ("route", logits.device, logits.dtype, logits.numel(), topk)
    if key not in _compiled:
        _compiled[key] = cute.compile(_launch_route, *tensors, logits.numel(), topk, stream)
    _compiled[key](*tensors, stream)
    return ids, weights


def moe_block(w, x, *, config):
    assert x.is_cuda and x.ndim == 1 and x.dtype == torch.bfloat16 and x.is_contiguous()
    ids, weights = route(F.linear(x, w.router), config.num_experts_per_tok)
    hidden = gate_up(x, w.experts_gate, w.experts_up, ids)
    contributions = expert_gemv(hidden, w.experts_down, ids, weights)
    shared_hidden = gate_up(x, w.shared_gate, w.shared_up)
    shared = nvfp4_gemv(shared_hidden, w.shared_down)
    strength = F.linear(x, w.shared_expert_gate)
    out = torch.empty_like(x)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    tensors = tuple(map(_tensor, (contributions, shared, strength, out)))
    key = ("reduce", x.device, x.numel(), config.num_experts_per_tok)
    if key not in _compiled:
        _compiled[key] = cute.compile(_launch_reduce, *tensors, x.numel(), config.num_experts_per_tok, stream)
    _compiled[key](*tensors, stream)
    return out
