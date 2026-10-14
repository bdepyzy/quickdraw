"""Convolution and FP32 gated-delta recurrence for tokens or chunks."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .matvec import fp8_gemv


@triton.jit
def _conv_sequence(X, W, State, Out, C: tl.constexpr, WIDTH: tl.constexpr,
                    T: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    token = tl.arange(0, BT)
    acc = tl.zeros((BT, BC), tl.float32)
    for tap in tl.static_range(WIDTH):
        source = token + tap - WIDTH + 1
        recent = tl.load(X + source[:, None] * C + channel[None, :],
                         (source[:, None] >= 0) & (token[:, None] < T) & (channel[None, :] < C), other=0)
        previous = tl.load(State + channel[None, :] * (WIDTH - 1) + source[:, None] + WIDTH - 1,
                           (source[:, None] < 0) & (token[:, None] < T) & (channel[None, :] < C), other=0)
        value = tl.where(source[:, None] >= 0, recent, previous).to(tl.float32)
        weight = tl.load(W + channel * WIDTH + tap, channel < C, other=0).to(tl.float32)
        acc += (value * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + token[:, None] * C + channel[None, :], acc * tl.sigmoid(acc),
              (token[:, None] < T) & (channel[None, :] < C))
    for tap in tl.static_range(WIDTH - 1):
        source = T - WIDTH + 1 + tap
        if source >= 0:
            history = tl.load(X + source * C + channel, channel < C, other=0)
        else:
            history = tl.load(State + channel * (WIDTH - 1) + source + WIDTH - 1, channel < C, other=0)
        tl.store(State + channel * (WIDTH - 1) + tap, history, channel < C)


@triton.jit
def _conv_kernel(X, W, State, Out, C: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    channel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + channel, channel < C, other=0)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for t in tl.static_range(WIDTH - 1):
        previous = tl.load(State + channel * (WIDTH - 1) + t, channel < C, other=0)
        weight = tl.load(W + channel * WIDTH + t, channel < C, other=0)
        product = (previous.to(tl.float32) * weight.to(tl.float32)).to(tl.bfloat16)
        acc += product.to(tl.float32)
        if t > 0:
            tl.store(State + channel * (WIDTH - 1) + t - 1, previous, channel < C)
    weight = tl.load(W + channel * WIDTH + WIDTH - 1, channel < C, other=0)
    acc += (value.to(tl.float32) * weight.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    if WIDTH > 1:
        tl.store(State + channel * (WIDTH - 1) + WIDTH - 2, value, channel < C)
    tl.store(Out + channel, acc * tl.sigmoid(acc), channel < C)


@triton.jit
def _delta_kernel(QKV, A, B, ALog, Bias, State, Out, KH: tl.constexpr, HV: tl.constexpr,
                   K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, T: tl.constexpr = 1):
    head = tl.program_id(0)
    key_head = head // (HV // KH)
    keys = tl.arange(0, BK)
    values = tl.program_id(1) * BV + tl.arange(0, BV)
    offsets = head * K * V + keys[:, None] * V + values[None, :]
    mask = (keys[:, None] < K) & (values[None, :] < V)
    state = tl.load(State + offsets, mask, other=0).to(tl.float32)
    for token in range(T):
        q = tl.load(QKV + token * (2 * KH * K + HV * V) + key_head * K + keys, keys < K, other=0).to(tl.float32)
        k = tl.load(QKV + token * (2 * KH * K + HV * V) + KH * K + key_head * K + keys, keys < K, other=0).to(tl.float32)
        v = tl.load(QKV + token * (2 * KH * K + HV * V) + 2 * KH * K + head * V + values, values < V, other=0).to(tl.float32)
        q *= tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * K ** -.5
        k *= tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
        alpha = tl.load(A + token * HV + head).to(tl.float32) + tl.load(Bias + head).to(tl.float32)
        softplus = tl.where(alpha > 20, alpha, tl.log(1 + tl.exp(alpha)))
        decay = tl.exp(-tl.exp(tl.load(ALog + head).to(tl.float32)) * softplus)
        beta = tl.sigmoid(tl.load(B + token * HV + head).to(tl.float32))
        state *= decay
        correction = (v - tl.sum(state * k[:, None], 0)) * beta
        state += k[:, None] * correction[None, :]
        tl.store(Out + token * HV * V + head * V + values, tl.sum(state * q[:, None], 0), values < V)
    tl.store(State + offsets, state, mask)


@triton.jit
def _gate_kernel(X, Z, W, Out, V: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    head = tl.program_id(0)
    x = tl.load(X + head * V + offsets, offsets < V, other=0).to(tl.float32)
    z = tl.load(Z + head * V + offsets, offsets < V, other=0).to(tl.float32)
    weight = tl.load(W + offsets, offsets < V, other=0).to(tl.float32)
    normalized = x * tl.rsqrt(tl.sum(x * x, 0) / V + EPS)
    tl.store(Out + head * V + offsets, normalized * weight * z * tl.sigmoid(z), offsets < V)


def gated_delta_net_block(w, x, rec_state, conv_state, position, *, config):
    assert config.linear_num_value_heads % config.linear_num_key_heads == 0
    assert rec_state.is_contiguous() and conv_state.is_contiguous()
    mixed = fp8_gemv(x, w.in_proj_qkv)
    z = fp8_gemv(x, w.in_proj_z)
    a, b = F.linear(x, w.in_proj_a), F.linear(x, w.in_proj_b)
    convolved = torch.empty_like(mixed)
    tokens = x.shape[0] if x.ndim == 2 else 1
    channels = mixed.shape[-1]
    if x.ndim == 2:
        _conv_sequence[(triton.cdiv(channels, 64),)](
            mixed, w.conv1d, conv_state, convolved, channels, config.linear_conv_kernel_dim,
            tokens, triton.next_power_of_2(tokens), 64, enable_fp_fusion=False)
    else:
        _conv_kernel[(triton.cdiv(channels, 128),)](
            mixed, w.conv1d, conv_state, convolved, channels, config.linear_conv_kernel_dim,
            128, enable_fp_fusion=False)
    heads, key_dim, value_dim = (config.linear_num_value_heads, config.linear_key_head_dim,
                                  config.linear_value_head_dim)
    core = torch.empty((tokens, heads, value_dim), dtype=x.dtype, device=x.device)
    _delta_kernel[(heads, triton.cdiv(value_dim, 16))](
        convolved, a, b, w.A_log, w.dt_bias, rec_state, core,
        config.linear_num_key_heads, heads, key_dim, value_dim,
        triton.next_power_of_2(key_dim), 16, tokens, num_warps=4, enable_fp_fusion=False)
    gated = torch.empty_like(z)
    _gate_kernel[(tokens * heads,)](core, z, w.norm, gated, value_dim, config.rms_norm_eps,
                           triton.next_power_of_2(value_dim), enable_fp_fusion=False)
    return fp8_gemv(gated, w.out_proj)
