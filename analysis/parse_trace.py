#!/usr/bin/env python3
"""Group trace kernels by CUPTI correlation ID and CUDA graph launch.
Decode steps follow CPU launch order; warmup/drain steps are excluded."""

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load_events(path):
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    data = json.loads(raw)
    events = data['traceEvents'] if isinstance(data, dict) else data
    return [e for e in events if isinstance(e, dict) and e.get('ph') == 'X']


# Kernel-name categories, first match wins.
RULES = [
    ('routing', r'(?i)topk|top_k|moe.*rout|router|argsort|sort.*expert|group.*topk|'
                r'moe_align|align_single_token|align.*token|expert.*(select|dispatch|map)|cumsum'),
    ('reduction_moe', r'(?i)moe.*(reduce|sum|aggregate)|weighted.*reduce'),
    ('marlin_moe', r'(?i)marlin_moe|marlin.*moe|moe.*marlin|fused_marlin|moe_wna16'),
    ('marlin_shared_lmhead', r'(?i)marlin|gptq|awq|wna16'),
    ('quant', r'(?i)_static_quant_fp8|quant_fp8|ConvertToFloat8|float8_e4m3|per_token_quant|'
              r'scaled_fp8_quant|dynamic.*quant'),
    ('attention', r'(?i)flash.*(attn|attention)|flashinfer.*(attention|decode|prefill)|fmha|'
                  r'attention|splitkv|split_kv|merge_?attn|merge_?states|pagedkv|kvcache|kv_cache|'
                  r'kv_indices|mla|BatchPrefill|BatchDecode|reshape_and_cache'),
    ('recurrent_conv', r'(?i)delta|gdn|gated|mamba|ssm|causal_conv|conv1d|chunk_scan|'
                       r'chunk_state|fused_recurrent|linear_attn|replay_state|wy_'),
    ('sampling', r'(?i)argmax|sample|greedy|multinomial|temperature|logits'),
    ('moe_bf16', r'(?i)fused_moe|moefcgemm|moe.*gemm|moe_kernel'),
    ('fp8_gemm', r'(?i)fp8|nvfp4|fp4|scaled_mm|sm120|sm100|sm90'),
    ('gemv_bf16', r'(?i)gemvx|matmul|sgemm|hgemm|cutlass.*gemm|cudnn.*gemm|splitk|cublas'),
    ('memcpy_memset', r'(?i)memcpy|memset|copy|fill|zeros'),
    ('norm_act_elem', r'(?i)rms_?norm|layernorm|layer_norm|silu|gelu|act_and_mul|elementwise|vectorized|'
                      r'reduce_kernel|softmax|rope|rotary|embedding|cast|add|mul|'
                      r'residual|sigmoid|index|gather|scatter|cat|split|permute|transpose|scan'),
]


def categorize(name):
    for cat, pat in RULES:
        if re.search(pat, name):
            return cat
    return 'other'


def intervals_union(intervals):
    """Union coverage of (start, end) intervals, in microseconds."""
    if not intervals:
        return 0.0
    total, cur_s, cur_e = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + cur_e - cur_s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--dump-kernels', type=int, default=0, metavar='N',
                   help='print top-N kernel names by total duration and exit')
    p.add_argument('--drop-steps', type=int, default=3,
                   help='drop this many first/last steps as warmup/drain')
    p.add_argument('--engine-steps', action='store_true',
                   help='cluster launches into engine steps at big-anchor launches (spec decode)')
    p.add_argument('--engine-anchor', type=int, default=500,
                   help='kernel count at/above which a launch starts a new engine step')
    p.add_argument('--steps-out', type=Path)
    a = p.parse_args()

    events = load_events(a.trace)
    kernels = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    print(f'events: {len(events)}, gpu work: {len(kernels)}, runtime api: {len(runtime)}')

    if a.dump_kernels:
        totals = defaultdict(float)
        counts = defaultdict(int)
        for k in kernels:
            totals[k['name']] += k.get('dur', 0)
            counts[k['name']] += 1
        for name, total in sorted(totals.items(), key=lambda kv: -kv[1])[:a.dump_kernels]:
            print(f'{total/1000:10.3f} ms  x{counts[name]:6d}  {name[:150]}')
        return

    by_corr = {}
    for e in runtime:
        corr = (e.get('args') or {}).get('correlation')
        if corr is not None:
            by_corr[corr] = e
    launches = {}
    eager = []
    for k in kernels:
        corr = (k.get('args') or {}).get('correlation')
        src = by_corr.get(corr)
        if src is not None and 'GraphLaunch' in src['name']:
            launches.setdefault(id(src), {'api': src, 'kernels': []})['kernels'].append(k)
        else:
            eager.append(k)
    groups = sorted(launches.values(), key=lambda g: g['api']['ts'])
    print(f'graph launches: {len(groups)}, eager gpu work: {len(eager)}')
    if len(groups) < 2 * a.drop_steps + 3:
        print('too few graph launches; is this a decode trace with CUDA graphs?')
    if a.engine_steps:
        # Group draft/tail launches with their preceding verification pass.
        clusters = []
        for g in groups:
            if len(g['kernels']) >= a.engine_anchor or not clusters:
                clusters.append([])
            clusters[-1].append(g)
        print(f'engine steps: {len(clusters)} (anchor >= {a.engine_anchor} kernels)')
        groups = [{'api': c[0]['api'], 'kernels': [k for g in c for k in g['kernels']]}
                  for c in clusters]
    steps = groups[a.drop_steps:len(groups) - a.drop_steps] if groups else []

    rows = []
    for g in steps:
        ks = g['kernels']
        span = max(k['ts'] + k['dur'] for k in ks) - min(k['ts'] for k in ks)
        union = intervals_union(sorted((k['ts'], k['ts'] + k['dur']) for k in ks))
        per_stream = defaultdict(list)
        for k in ks:
            per_stream[k['tid']].append((k['ts'], k['ts'] + k['dur']))
        per_cat = defaultdict(float)
        for k in ks:
            per_cat[categorize(k['name'])] += k['dur']
        rows.append({'span_us': span, 'union_us': union, 'gap_us': span - union,
                     'n_kernels': len(ks), 'api_dur_us': g['api']['dur'],
                     'api_ts': g['api']['ts'],
                     'streams': {str(t): round(intervals_union(iv), 1)
                                 for t, iv in sorted(per_stream.items())},
                     'categories': {c: round(v, 1) for c, v in
                                    sorted(per_cat.items(), key=lambda kv: -kv[1])}})

    for prev, cur in zip(rows, rows[1:]):
        cur['launch_interval_us'] = cur['api_ts'] - prev['api_ts']

    def med(key):
        vals = [r[key] for r in rows if key in r]
        return statistics.median(vals) if vals else 0.0

    print(f'steps kept: {len(rows)}')
    print(f'median span: {med("span_us"):.1f} us, union busy: {med("union_us"):.1f} us, '
          f'uncovered: {med("gap_us"):.1f} us, kernels/step: {med("n_kernels"):.0f}, '
          f'launch interval: {med("launch_interval_us"):.1f} us, api dur: {med("api_dur_us"):.1f} us')
    cats = defaultdict(list)
    for r in rows:
        for c, v in r['categories'].items():
            cats[c].append(v)
    print('\nper-step category sums (median us):')
    for c, vals in sorted(cats.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f'  {c:16s} {statistics.median(vals):8.1f}')
    streams = defaultdict(list)
    for r in rows:
        for t, v in r['streams'].items():
            streams[t].append(v)
    print('\nper-step stream coverage (median us):')
    for t, vals in sorted(streams.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f'  stream {t:>4s} {statistics.median(vals):8.1f}')

    if a.steps_out:
        a.steps_out.write_text(json.dumps({'trace': str(a.trace), 'steps': rows}, indent=2) + '\n')
        print(f'\nwrote {a.steps_out}')


if __name__ == '__main__':
    main()
