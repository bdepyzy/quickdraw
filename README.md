# quickdraw

Batch-one text inference for Qwen3.6-35B-A3B-NVFP4 on one RTX 5090.
The runtime provides a Torch reference path and optional Triton FP8/NVFP4
projections. Prefill currently runs token by token; MoE routing transfers expert
IDs to CPU. Batched prefill, fused blocks, and CUDA graph replay are not implemented.

## Layout

| Path | Responsibility |
|---|---|
| `quickdraw/models/weights.py` | Configuration, packed weights, streamed loading |
| `quickdraw/models/qwen.py` | Reference model operations |
| `quickdraw/engine/engine.py` | Request state, generation, EOS, reset |
| `quickdraw/kvcache/state.py` | KV, recurrent, and convolution buffers |
| `quickdraw/kernel/` | Function binding and GPU kernels |
| `benchmark/offline/` | Timing, saved-baseline comparison, profiling, parity |
| `tests/` | Formula, state, request-control, and integration checks |

[Runtime interfaces](docs/architecture.md) specify kernel binding, execution hooks,
weight formats, and cache ownership. [todo.md](todo.md) tracks implementation work
and measured performance.

## Inference

The default checkpoint path is `~/data/models/vllm/Qwen3.6-35B-A3B-NVFP4`.
The retained `.venv-vllm` environment supplies the tokenizer:

```sh
.venv/bin/python scripts/direct_run.py --cache-tag quickdraw -- \
  .venv-vllm/bin/python -m examples.offline --backend triton \
  --prompt "What is 2 + 2? Answer briefly." --max-tokens 16
```

The runner requires CUDA, applies the checkpoint chat template with thinking
disabled, and reads its EOS IDs. `--raw` selects plain text completion;
`--report PATH` saves timings and VRAM. `--backend reference` selects Torch.

GPU jobs run sequentially through `scripts/direct_run.py`: 12 GiB host RAM cap,
no child swap, two CPU cores, and one compiler worker. The retained environments
are `.venv`, `.venv-vllm`, and `.venv-sglang`.

## Benchmarks

```sh
uv run benchmark/offline/bench.py --backend triton \
  --input-lens 128 512 --output-len 32 --runs 5
uv run benchmark/offline/compare.py
uv run benchmark/offline/compare.py --baselines benchmark/runs/compare-SESSION
uv run benchmark/offline/compare.py --report benchmark/runs/compare-SESSION
```

Comparison defaults to BS=1, 128 input IDs, 16 outputs, one warmup, and five
measured Quickdraw requests. It reads the newest complete saved vLLM/SGLang
baselines matching the checkpoint, GPU, and workload. Missing baselines fail the
run. `--refresh-baselines` explicitly enables sequential offline baseline runs:

```sh
uv run benchmark/offline/compare.py --refresh-baselines \
  --input-lens 128 2048 --output-len 128
```

All engines receive identical synthetic IDs with greedy generation, EOS ignored,
BF16 KV, FP32 recurrence, and no prefix reuse or speculation. Baseline CUDA graphs
are enabled; vLLM uses async scheduling. Comparison workers reclaim checkpoint
file cache before timing and reject measured runs with host RAM throttling.

Reports contain median TTFT, decode milliseconds per token, decode tokens per
second, total time, and percentage gaps. TTFT includes CPU delivery of the first
ID; decode excludes that ID. Loading and warmup are excluded. Sessions under
`benchmark/runs/` retain samples, output IDs, medians, settings, versions, and logs.
Latency gaps use baseline time as the denominator; TPS deficit uses baseline TPS.

`BW SoL est` divides modeled active-weight and state traffic by decode time and
1792 GB/s peak bandwidth. It excludes L2 reuse, intermediates, and repacking;
it is not measured DRAM utilization. `--peak-bandwidth-gbs` overrides the peak.

## Profiling

```sh
uv run benchmark/offline/profile_torch.py
uv run benchmark/offline/profile_nsys.py
uv run benchmark/offline/profile_ncu.py --profile-phase decode
```

Defaults are Triton, 128 input IDs, 16 outputs, three timed runs, and one warmup.
Each tool prints inference times and a session path under `benchmark/profiles/`.

| Tool | Saved output |
|---|---|
| Torch | Chrome trace `.trace.json` and operator table `.operators.txt` |
| Nsight Systems | `.nsys-rep` |
| Nsight Compute | `.ncu-rep` |

The default capture phase is `generation`, covering prefill and all requested
outputs with separate `quickdraw.prefill` and `quickdraw.decode` ranges.
`--profile-phase prefill` captures prefill; `--profile-phase decode` captures one
additional decode step. Capture excludes loading and warmup and runs separately
from timing samples. Shape recording is disabled unless `--profile-shapes` is set.

The Compute wrapper collects `SpeedOfLight` for one matching NVFP4 launch.
`--ncu-kernel`, `--ncu-launch-count`, and `--ncu-section` control collection.
The underlying `bench.py --profile ncu` defaults to `LaunchStats` for ten launches.
Nsight tools must be on `PATH` or under `.runtime/profilers/opt/nvidia/`;
Compute requires performance-counter access. Failed captures retain completed
reports and diagnostic logs and return nonzero.

## Validation

```sh
uv run -m tests.models.mathcheck --strict
uv run -m tests.engine.check --strict
uv run -m unittest discover -s tests/benchmark
.venv/bin/python scripts/direct_run.py --cache-tag quickdraw -- \
  .venv/bin/python -m tests.kernel.check
HF_HUB_OFFLINE=1 .venv/bin/python scripts/direct_run.py --cache-tag correctness -- \
  .venv-vllm/bin/python -m tests.models.integrationcheck
```

Formula checks include 15 independent block/state fixtures; engine checks use
synthetic logits. [Reference provenance](references/model/README.md) describes
formats and tolerances. GPU kernel checks support `--benchmark` for projection
latency. These checks do not establish full-model engine parity.

Full-model comparison uses a saved vLLM capture:

```sh
.venv/bin/python scripts/direct_run.py --cache-tag quickdraw -- \
  .venv/bin/python -m benchmark.offline.parity compare --backend triton
```

A new capture requires an explicit separate run:

```sh
VLLM_HAS_FLASHINFER_CUBIN=1 HF_HUB_OFFLINE=1 .venv/bin/python scripts/direct_run.py --cache-tag vllm -- \
  .venv-vllm/bin/python -m benchmark.offline.parity capture
```

Both engines process identical forced token histories. Default limits require
matching greedy IDs, logit RMSE at most 0.25, and probability total variation at
most 0.05. The 2026-10-04 run matched 48/48 greedy choices but failed numerical
limits, with maximum RMSE 0.704. [The report](benchmark/parity/vllm.json) records
reset reproducibility and repeated-request variation. Internal vLLM state is not
compared. Fixed-length outputs include positions after EOS as stress cases.

Historical server benchmarks and trace analysis remain under `benchmark/online/`
and `analysis/`. Their speculative settings differ from the offline comparison.
