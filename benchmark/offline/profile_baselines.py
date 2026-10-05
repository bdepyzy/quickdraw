#!/usr/bin/env python3
"""Capture decode traces under analysis/traces/<label>-<scenario>/.
SGLang uses its server profiling API; vLLM uses the offline LLM profiler."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = '/home/bdepyzy/data/models/vllm/Qwen3.6-35B-A3B-NVFP4'
THREAD_ENV = {key: '1' for key in ['MAX_JOBS', 'FLASHINFER_NVCC_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS',
                                   'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}
THREAD_ENV.update(TOKENIZERS_PARALLELISM='false', VLLM_CACHE_ROOT='/home/bdepyzy/data/models/vllm')


def engine_command(cfg, port=8001):
    if cfg['engine'] == 'vllm':
        return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODEL,
                '--host', '127.0.0.1', '--port', str(port), '--quantization', 'modelopt',
                '--safetensors-load-strategy', 'lazy', '--language-model-only', '--mm-processor-cache-gb', '0',
                '--max-model-len', '32768', '--max-num-seqs', '1', '--gpu-memory-utilization', '0.90',
                '--reasoning-parser', 'qwen3'] + cfg.get('args', [])
    if cfg['engine'] == 'sglang':
        return [sys.executable, '-m', 'sglang.launch_server', '--model-path', MODEL,
                '--host', '127.0.0.1', '--port', str(port)] + cfg.get('args', [])
    raise ValueError(f"Unknown engine: {cfg['engine']}")


BASE = 'http://127.0.0.1:8001'
TRACES = ROOT / 'analysis' / 'traces'
DATASET = ROOT / 'benchmark' / 'dataset.json'


def api(route, body=None, timeout=600, stream=False):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(
        BASE + route, data=data, headers={'Content-Type': 'application/json'}), timeout=timeout)


def wait_healthy(server, timeout=1800):
    start = time.monotonic()
    while True:
        if server.poll() is not None:
            raise RuntimeError(f'Server exited {server.returncode}; see log')
        try:
            with api('/health', timeout=2) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except Exception:
            pass
        if time.monotonic() - start > timeout:
            raise TimeoutError('Server not ready')
        time.sleep(2)


def prompt_ids(scenario, run=7):
    d = json.loads(DATASET.read_text())
    return next(e['prompt'] for e in d['examples']
                if e['scenario'] == scenario and e['run'] == run)


def stream_request(model, ids, tokens, first_token, done):
    """Streaming completion; sets first_token when the first chunk arrives."""
    body = {'model': model, 'prompt': ids, 'max_tokens': tokens, 'temperature': 0,
            'ignore_eos': True, 'stream': True}
    try:
        with api('/v1/completions', body) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                raw = line[5:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                if event.get('error'):
                    raise RuntimeError(event['error'])
                if any(c.get('text') for c in event.get('choices', [])):
                    first_token.set()
    finally:
        done.set()


def sglang_capture(cfg, label, scenarios, steps, tokens):
    outroot = TRACES
    outroot.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **THREAD_ENV, **cfg.get('env', {}),
               SGLANG_TORCH_PROFILER_DIR=str(outroot))
    log = (outroot / f'{label}-profile-server.log').open('w')
    server = subprocess.Popen(engine_command(cfg), cwd=ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        startup = wait_healthy(server)
        print(f'{label}: healthy after {startup:.0f}s', flush=True)
        with api('/v1/models') as r:
            model = json.load(r)['data'][0]['id']
        for warm in ['128', '2048']:
            with api('/v1/completions', {'model': model, 'prompt': prompt_ids(warm, 0),
                                         'max_tokens': 32, 'temperature': 0, 'ignore_eos': True}):
                pass
        for scenario in scenarios:
            outdir = outroot / f'{label}-{scenario}'
            outdir.mkdir(exist_ok=True)
            first_token, done = threading.Event(), threading.Event()
            t = threading.Thread(target=stream_request,
                                 args=(model, prompt_ids(scenario), tokens, first_token, done),
                                 daemon=True)
            t.start()
            if not first_token.wait(timeout=300):
                raise RuntimeError(f'{scenario}: no first token within 300s')
            body = {'output_dir': str(outdir), 'num_steps': str(steps),
                    'activities': ['CPU', 'GPU'], 'profile_by_stage': False,
                    'merge_profiles': False, 'profile_prefix': f'{label}-{scenario}'}
            t0 = time.monotonic()
            with api('/start_profile', body, timeout=600) as r:
                resp = r.read().decode(errors='replace')[:200]
            print(f'{label} {scenario}: profiled {steps} steps in '
                  f'{time.monotonic()-t0:.1f}s -> {resp!r}', flush=True)
            if not done.wait(timeout=300):
                raise RuntimeError(f'{scenario}: request did not finish')
            t.join(timeout=60)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline and not any(outdir.iterdir()):
                time.sleep(2)
            files = list(outdir.rglob('*'))
            print(f'{label} {scenario}: {len(files)} trace files in {outdir}', flush=True)
            if not files:
                raise RuntimeError(f'{scenario}: no trace files written')
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        log.close()


def cli_to_kwargs(cfg):
    """Map the saved config's CLI args to EngineArgs kwargs for the offline LLM API."""
    flags = {'--language-model-only': 'language_model_only'}
    known = {'--model': 'model', '--quantization': 'quantization',
             '--safetensors-load-strategy': 'safetensors_load_strategy',
             '--mm-processor-cache-gb': 'mm_processor_cache_gb',
             '--max-model-len': 'max_model_len', '--max-num-seqs': 'max_num_seqs',
             '--gpu-memory-utilization': 'gpu_memory_utilization',
             '--reasoning-parser': 'reasoning_parser',
             '--max-num-batched-tokens': 'max_num_batched_tokens',
             '--attention-backend': 'attention_backend',
             '--kernel-config': 'kernel_config',
             '--speculative-config': 'speculative_config'}
    base = ['--model', MODEL, '--quantization', 'modelopt', '--safetensors-load-strategy', 'lazy',
            '--language-model-only', '--mm-processor-cache-gb', '0', '--max-model-len', '32768',
            '--max-num-seqs', '1', '--gpu-memory-utilization', '0.90', '--reasoning-parser', 'qwen3']
    full = base + list(cfg.get('args', []))
    i, kwargs = 0, {}
    while i < len(full):
        arg = full[i]
        if arg in flags:
            kwargs[flags[arg]] = True
            i += 1
        elif arg in known:
            key = known[arg]
            raw = full[i + 1]
            for conv in (json.loads, int, float):
                try:
                    raw = conv(raw)
                    break
                except (ValueError, TypeError):
                    continue
            kwargs[key] = raw
            i += 2
        else:
            raise ValueError(f'Unmapped CLI arg: {arg}; extend cli_to_kwargs')
    return kwargs


def vllm_capture(cfg, label, scenarios, steps, tokens):
    for key, val in THREAD_ENV.items():
        os.environ.setdefault(key, val)
    for key, val in cfg.get('env', {}).items():
        os.environ[key] = val
    outroot = TRACES
    outroot.mkdir(parents=True, exist_ok=True)
    kwargs = cli_to_kwargs(cfg)
    kwargs['profiler_config'] = {'profiler': 'torch', 'torch_profiler_dir': str(outroot)}
    print(f'{label}: LLM({sorted(kwargs)})', flush=True)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(**kwargs)
    sp_warm = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    llm.generate([TokensPrompt(prompt_token_ids=prompt_ids('128', 0))], sp_warm)
    sp = SamplingParams(temperature=0, max_tokens=steps + 8, ignore_eos=True)
    for scenario in scenarios:
        llm.start_profile(profile_prefix=f'{label}-{scenario}')
        t0 = time.monotonic()
        llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(scenario))], sp)
        llm.stop_profile()
        print(f'{label} {scenario}: {steps + 8} tokens profiled in '
              f'{time.monotonic()-t0:.1f}s', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--scenarios', nargs='+', default=['128', '2048', '8192'])
    p.add_argument('--steps', type=int, default=40, help='decode steps to profile per scenario')
    p.add_argument('--tokens', type=int, default=96, help='request length; must exceed steps + prefill')
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    label = cfg['label']
    if a.tokens < a.steps + 20:
        p.error('--tokens must be at least --steps + 20')
    if cfg['engine'] == 'sglang':
        sglang_capture(cfg, label, a.scenarios, a.steps, a.tokens)
    elif cfg['engine'] == 'vllm':
        vllm_capture(cfg, label, a.scenarios, a.steps, a.tokens)
    else:
        p.error(f"unknown engine {cfg['engine']}")
    print('done', flush=True)


if __name__ == '__main__':
    main()
