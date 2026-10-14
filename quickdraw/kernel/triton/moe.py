"""GPU routing and direct selected-expert reads from packed NVFP4 banks."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .matvec import _fp4, expert_gemv, nvfp4_gemv


@triton.jit
def _route_kernel(Logits, Ids, Weights, E: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    Logits += row * E
    Ids += row * TOPK
    Weights += row * TOPK
    experts = tl.arange(0, BLOCK)
    logits = tl.load(Logits + experts, experts < E, other=-float("inf")).to(tl.float32)
    probabilities = tl.exp(logits - tl.max(logits, 0))
    probabilities /= tl.sum(probabilities, 0)
    total = tl.full((), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        best = tl.max(probabilities, 0)
        expert = tl.min(tl.where(probabilities == best, experts, 2147483647), 0)
        tl.store(Ids + slot, expert)
        tl.store(Weights + slot, best)
        total += best
        probabilities = tl.where(experts == expert, -float("inf"), probabilities)
    for slot in tl.static_range(TOPK):
        tl.store(Weights + slot, tl.load(Weights + slot) / total)


@triton.jit
def _silu_kernel(Gate, Up, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(Gate + offsets, offsets < N, other=0).to(tl.float32)
    up = tl.load(Up + offsets, offsets < N, other=0).to(tl.float32)
    activated = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + offsets, activated * up, offsets < N)


@triton.jit
def _reduce_kernel(Experts, Shared, Strength, Out, H: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    Experts += row * TOPK * H
    Shared += row * H
    Strength += row
    Out += row * H
    features = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, SLOTS)
    contributions = tl.load(Experts + slots[:, None] * H + features[None, :],
                            (slots[:, None] < TOPK) & (features[None, :] < H), other=0).to(tl.float32)
    routed = tl.sum(contributions, 0).to(tl.bfloat16).to(tl.float32)
    strength = tl.sigmoid(tl.load(Strength).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    shared = tl.load(Shared + features, features < H, other=0).to(tl.float32)
    gated_shared = (shared * strength).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + features, routed + gated_shared, features < H)


def moe_block(w, x, *, config):
    assert 1 <= config.num_experts_per_tok <= config.num_experts
    logits = F.linear(x, w.router)
    tokens = x.shape[0] if x.ndim == 2 else 1
    ids = torch.empty((tokens, config.num_experts_per_tok), device=x.device, dtype=torch.int32)
    weights = torch.empty_like(ids, dtype=torch.float32)

    _route_kernel[(tokens,)](logits, ids, weights, config.num_experts, config.num_experts_per_tok,
                        triton.next_power_of_2(config.num_experts), enable_fp_fusion=False)
    if x.ndim == 2:
        sorted_ids, order = ids.flatten().sort(stable=True)
        boundaries = torch.searchsorted(sorted_ids, torch.arange(config.num_experts + 1, device=x.device))
        tiles = (boundaries.diff() + 15) // 16
        tile_starts = torch.cat((tiles.new_zeros(1), tiles.cumsum(0)))
        grouping = (order, boundaries, tile_starts, config.num_experts_per_tok)
        gate = grouped_matmul(x, w.experts_gate, grouping)
        up = grouped_matmul(x, w.experts_up, grouping)
    else:
        ids, weights = ids.view(-1), weights.view(-1)
        gate = expert_gemv(x, w.experts_gate, ids)
        up = expert_gemv(x, w.experts_up, ids)
    hidden = torch.empty_like(gate)
    _silu_kernel[(triton.cdiv(hidden.numel(), 256),)](gate, up, hidden, hidden.numel(), 256, enable_fp_fusion=False)

    contributions = (grouped_matmul(hidden, w.experts_down, grouping, weights) if x.ndim == 2 else expert_gemv(hidden, w.experts_down, ids, weights))
    shared_gate = nvfp4_gemv(x, w.shared_gate)
    shared_up = nvfp4_gemv(x, w.shared_up)
    shared_hidden = torch.empty_like(shared_gate)
    _silu_kernel[(triton.cdiv(shared_hidden.numel(), 256),)](shared_gate, shared_up, shared_hidden, shared_hidden.numel(), 256, enable_fp_fusion=False)

    shared = nvfp4_gemv(shared_hidden, w.shared_down)
    strength = F.linear(x, w.shared_expert_gate)
    output = torch.empty_like(x)
    _reduce_kernel[(triton.cdiv(x.shape[-1], 256), tokens)](
        contributions, shared, strength, output, x.shape[-1], config.num_experts_per_tok,
        triton.next_power_of_2(config.num_experts_per_tok), 256, enable_fp_fusion=False)
    return output


@triton.jit
def _grouped_kernel(X, W, S, Global, Order, Boundaries, TileStarts, Routing, Y,
                    R: tl.constexpr, E: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    GROUP: tl.constexpr, TOPK: tl.constexpr, ROUTED: tl.constexpr,
                    GLOBAL_BANK: tl.constexpr, BE: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    if tile < tl.load(TileStarts + E):
        experts = tl.arange(0, BE)
        ends = tl.load(TileStarts + experts + 1, experts < E, other=2147483647)
        expert = tl.sum((tile >= ends).to(tl.int32), 0)
        start = tl.load(Boundaries + expert)
        stop = tl.load(Boundaries + expert + 1)
        row = start + (tile - tl.load(TileStarts + expert)) * BM + tl.arange(0, BM)
        route = tl.load(Order + row, row < stop, other=0)
        inputs = row if ROUTED else route // TOPK
        columns = tl.program_id(1) * BN + tl.arange(0, BN)
        inner = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + inner
            activation = tl.load(X + inputs[:, None] * K + kk[None, :],
                                 (row[:, None] < stop) & (kk[None, :] < K), other=0)
            packed = tl.load(W + expert * N * (K // 2) + columns[None, :] * (K // 2) + kk[:, None] // 2,
                             (columns[None, :] < N) & (kk[:, None] < K), other=0)
            scale = tl.load(S + expert * N * (K // GROUP) + columns[None, :] * (K // GROUP) + kk[:, None] // GROUP,
                            (columns[None, :] < N) & (kk[:, None] < K), other=0.0).to(tl.float32)
            codes = (packed >> ((kk[:, None] % 2) * 4)) & 15
            weight = (_fp4(codes) * scale).to(tl.bfloat16)
            acc += tl.dot(activation, weight)
        global_scale = tl.load(Global + (expert if GLOBAL_BANK else 0)).to(tl.float32)
        if ROUTED:
            routing = tl.load(Routing + route, row < stop, other=0)
            global_scale = (global_scale * routing).to(tl.bfloat16).to(tl.float32)
            acc = acc.to(tl.bfloat16).to(tl.float32)
            acc *= global_scale[:, None]
        else:
            acc *= global_scale
        destination = route if ROUTED else row
        tl.store(Y + destination[:, None] * N + columns[None, :], acc,
                  (row[:, None] < stop) & (columns[None, :] < N))


def grouped_matmul(x, proj, grouping, routing=None):
    order, boundaries, tile_starts, topk = grouping
    routes, experts = order.numel(), proj.weight.shape[0]
    out = torch.empty((routes, proj.out_features), device=x.device, dtype=x.dtype)
    _grouped_kernel[(triton.cdiv(routes, 16) + experts, triton.cdiv(proj.out_features, 32))](
        x, proj.weight, proj.weight_scale, proj.weight_scale_2, order, boundaries, tile_starts,
        routing, out, routes, experts, proj.out_features, proj.in_features, proj.group_size,
        topk, routing is not None, proj.weight_scale_2.numel() != 1,
        triton.next_power_of_2(experts), 16, 32, 64, num_warps=4, enable_fp_fusion=False)
    return out
