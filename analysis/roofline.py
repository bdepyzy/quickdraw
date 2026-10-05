#!/usr/bin/env python3
"""Model ordinary BS=1 decode traffic from Qwen checkpoint formats and state sizes.
Derived bytes exclude cache effects and spills; not a measured or speculative rate."""

import json
import sys
from pathlib import Path

MODEL = Path('/home/bdepyzy/data/models/vllm/Qwen3.6-35B-A3B-NVFP4')
RTX5090_PEAK_GBS = 1792.0  # theoretical GDDR7 bandwidth, GB/s


def main():
    cfg = json.loads((MODEL / 'config.json').read_text())['text_config']
    H = cfg['hidden_size']
    L = cfg['num_hidden_layers']
    E = cfg['num_experts']
    TOPK = cfg['num_experts_per_tok']
    MI = cfg['moe_intermediate_size']
    SI = cfg['shared_expert_intermediate_size']
    LKH, LKD = cfg['linear_num_key_heads'], cfg['linear_key_head_dim']
    LVH, LVD = cfg['linear_num_value_heads'], cfg['linear_value_head_dim']
    AH, AD = cfg['num_attention_heads'], cfg['head_dim']
    KV = cfg['num_key_value_heads']
    VOCAB = cfg['vocab_size']
    full_every = cfg['full_attention_interval']
    n_linear = sum(1 for t in cfg['layer_types'] if t == 'linear_attention')
    n_full = L - n_linear

    FP8 = 1.0
    NVFP4 = 0.5 + 1.0 / 16  # 4-bit weight + 1 FP8 scale per group of 16
    BF16 = 2.0

    expert = 3 * H * MI
    routed = L * TOPK * expert * NVFP4
    shared = L * (3 * H * SI) * NVFP4
    router = L * H * E * BF16
    lin_qkv = H * (2 * LKH * LKD + LVH * LVD)
    lin_z = H * (LVH * LVD)
    lin_o = (LVH * LVD) * H
    linear_attn = n_linear * (lin_qkv + lin_z + lin_o) * FP8
    fa_q = H * (AH * AD)
    fa_k = H * (KV * AD)
    fa_v = H * (KV * AD)
    fa_o = (AH * AD) * H
    full_attn = n_full * (fa_q + fa_k + fa_v + fa_o) * FP8
    lm_head = H * VOCAB * NVFP4

    weights = routed + shared + router + linear_attn + full_attn + lm_head

    rec_state = n_linear * (LVH * LKD * LVD) * 4 * 2   # fp32 recurrent state r+w
    conv_state = n_linear * (cfg['linear_conv_kernel_dim'] - 1) * (2 * LKH * LKD + LVH * LVD) * 2 * 2
    print(f'layers: {L} ({n_linear} linear-attn, {n_full} full-attn), experts {E} top-{TOPK}')
    print(f'\nbytes per decoded token (weights, one user):')
    rows = [('routed experts (NVFP4)', routed), ('shared expert (NVFP4)', shared),
            ('router (bf16)', router), ('linear-attn projections (FP8)', linear_attn),
            ('full-attn projections (FP8)', full_attn), ('lm_head (NVFP4)', lm_head)]
    for name, b in rows:
        print(f'  {name:34s} {b/2**20:8.1f} MiB')
    print(f'  {"TOTAL weights":34s} {weights/2**20:8.1f} MiB ({weights/1e9:.3f} GB)')
    print(f'\nstate traffic per token:')
    print(f'  recurrent state r+w (fp32)        {rec_state/2**20:8.1f} MiB')
    print(f'  conv state r+w (bf16)             {conv_state/2**20:8.1f} MiB')
    for ctx in (128, 2048, 8192):
        kv = n_full * 2 * KV * AD * ctx * 2  # K+V, bf16
        print(f'  KV cache read @ ctx {ctx:5d}        {kv/2**20:8.1f} MiB')
    print(f'\ntheoretical floor at {RTX5090_PEAK_GBS:.0f} GB/s peak '
          f'(weights+recurrent only, 100% efficiency):')
    base = weights + rec_state + conv_state
    for ctx in (128, 2048, 8192):
        kv = n_full * 2 * KV * AD * ctx * 2
        total = base + kv
        ms = total / (RTX5090_PEAK_GBS * 1e9) * 1e3
        print(f'  ctx {ctx:5d}: {total/1e9:.3f} GB -> {ms:.3f} ms/token -> {1000/ms:.0f} tok/s ceiling')
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            ms = float(arg)
            bw = base / (ms * 1e-3) / 1e9
            print(f'\nmeasured {ms:.3f} ms/token -> estimated {bw:.0f} GB/s weight+state traffic '
                  f'= {100*bw/RTX5090_PEAK_GBS:.0f}% of peak')


if __name__ == '__main__':
    main()
