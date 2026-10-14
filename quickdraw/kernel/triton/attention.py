"""Causal GQA over contiguous KV with a host or device position."""

import torch
import triton
import triton.language as tl

from .matvec import fp8_gemv


@triton.jit
def _prepare(QG, K, V, QNorm, KNorm, Q, KC, VC, Position,
             HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr,
             THETA: tl.constexpr, EPS: tl.constexpr, DEVICE_POS: tl.constexpr,
             BLOCK: tl.constexpr):
    head, token = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BLOCK)
    position = tl.load(Position) if DEVICE_POS else Position
    position += token
    if head < HQ:
        offset = token * HQ * 2 * D + head * 2 * D
        x = tl.load(QG + offset + d, d < D, other=0).to(tl.float32)
        gamma = 1 + tl.load(QNorm + d, d < D, other=0).to(tl.float32)
    else:
        offset = token * HK * D + (head - HQ) * D
        x = tl.load(K + offset + d, d < D, other=0).to(tl.float32)
        gamma = 1 + tl.load(KNorm + d, d < D, other=0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    normalized = (x * inv_rms * gamma).to(tl.bfloat16).to(tl.float32)
    if ROT > 0:
        partner = tl.where(d < ROT // 2, d + ROT // 2, d - ROT // 2)
        if head < HQ:
            other = tl.load(QG + offset + partner, d < ROT, other=0).to(tl.float32)
            other_gamma = 1 + tl.load(QNorm + partner, d < ROT, other=0).to(tl.float32)
        else:
            other = tl.load(K + offset + partner, d < ROT, other=0).to(tl.float32)
            other_gamma = 1 + tl.load(KNorm + partner, d < ROT, other=0).to(tl.float32)
        other = (other * inv_rms * other_gamma).to(tl.bfloat16).to(tl.float32)
        angle = position * tl.exp(-tl.log(THETA) * (d % (ROT // 2)) / ROT * 2)
        cos = tl.cos(angle).to(tl.bfloat16).to(tl.float32)
        sin = tl.sin(angle).to(tl.bfloat16).to(tl.float32)
        rotated = normalized * cos + tl.where(d < ROT // 2, -other, other) * sin
        normalized = tl.where(d < ROT, rotated, normalized)
    if head < HQ:
        tl.store(Q + token * HQ * D + head * D + d, normalized, d < D)
    else:
        destination = position * HK * D + (head - HQ) * D + d
        tl.store(KC + destination, normalized, d < D)
        value = tl.load(V + offset + d, d < D, other=0)
        tl.store(VC + destination, value, d < D)


@triton.jit
def _attend(Q, QG, KC, VC, Out, Position, HQ: tl.constexpr, HK: tl.constexpr,
            D: tl.constexpr, DEVICE_POS: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    head, token = tl.program_id(0), tl.program_id(1)
    position = tl.load(Position) if DEVICE_POS else Position
    position += token
    d = tl.arange(0, BD)
    q = tl.load(Q + token * HQ * D + head * D + d, d < D, other=0).to(tl.float32)
    time = tl.arange(0, BT)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    accumulator = tl.full((BD,), 0, tl.float32)
    for start in range(tl.cdiv(position + 1, BT)):
        slots = start * BT + time
        offsets = slots[:, None] * HK * D + (head // (HQ // HK)) * D + d[None, :]
        mask = (slots[:, None] <= position) & (d[None, :] < D)
        keys = tl.load(KC + offsets, mask, other=0).to(tl.float32)
        scores = tl.sum(keys * q[None, :], 1) * D ** -.5
        scores = tl.where(slots <= position, scores, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(scores, 0))
        rescale = tl.exp(maximum - new_max)
        probabilities = tl.exp(scores - new_max)
        values = tl.load(VC + offsets, mask, other=0).to(tl.float32)
        accumulator = accumulator * rescale + tl.sum(probabilities[:, None] * values, 0)
        denominator = denominator * rescale + tl.sum(probabilities, 0)
        maximum = new_max
    attended = (accumulator / denominator).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(QG + token * HQ * 2 * D + head * 2 * D + D + d,
                   d < D, other=0).to(tl.float32)
    gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + token * HQ * D + head * D + d, attended * gate, d < D)


def full_attention_block(w, x, kv_cache, position, *, config):
    tokens = x.shape[0] if x.ndim == 2 else 1
    qg, k, v = (fp8_gemv(x, proj) for proj in (w.q_proj, w.k_proj, w.v_proj))
    q = torch.empty((tokens, config.num_attention_heads * config.head_dim), dtype=x.dtype, device=x.device)
    gated = torch.empty_like(q)
    grid = (config.num_attention_heads + config.num_key_value_heads, tokens)
    _prepare[grid](qg, k, v, w.q_norm, w.k_norm, q, *kv_cache, position,
                   config.num_attention_heads, config.num_key_value_heads, config.head_dim,
                   int(config.head_dim * config.partial_rotary_factor), float(config.rope_theta),
                   config.rms_norm_eps, isinstance(position, torch.Tensor),
                   triton.next_power_of_2(config.head_dim), enable_fp_fusion=False)
    _attend[(config.num_attention_heads, tokens)](
        q, qg, *kv_cache, gated, position, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, isinstance(position, torch.Tensor),
        triton.next_power_of_2(config.head_dim), 32, enable_fp_fusion=False)
    return fp8_gemv(gated if x.ndim == 2 else gated.view(-1), w.o_proj)
