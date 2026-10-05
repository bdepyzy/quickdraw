# Quickdraw implementation roadmap

Updated: 2026-10-04.

Build a fast, correct inference library for one RTX 5090 and one active user.
Start with Qwen3.6-35B-A3B-NVFP4. Keep the request engine and kernel interfaces
small enough to support other models later. Specialize computation where the
measured workload benefits. Do not promise that one implementation will be the
fastest for every model, prompt length, and GPU.

This file is a plan. Unchecked items are not implemented or verified.

## 1. Current state and measured performance

Completed foundations:

- [x] Load the local quantized checkpoint into GPU memory.
- [x] Implement reference attention, Gated DeltaNet, MoE, and token-to-logits math.
- [x] Implement request reset, prefill, greedy decode, EOS, and capacity handling.
- [x] Add Triton FP8 and NVFP4 matrix-vector projections.
- [x] Provide plain function hooks for replacement kernels and token execution.
- [x] Provide formula checks, saved independent block references, and engine checks.
- [x] Save Torch, Nsight Systems, and Nsight Compute profiling sessions.
- [x] Measure five requests per engine and save all samples and medians.
- [x] Make comparison use saved baselines by default. Start only Quickdraw.
- [x] Require `--refresh-baselines` before launching vLLM or SGLang.
- [x] Reclaim checkpoint file cache before comparison timing. Reject runs that
  hit the host RAM soft limit during measurement.

The optimized backend currently replaces quantized projections. Much of the
model still executes as individual Torch operations. A complete fused MoE block,
a batched prefill path, and reusable decode CUDA Graphs remain to be built.

Measured medians for BS=1, 128 input IDs, and 16 output IDs:

| Engine | TTFT ms | Decode ms/token | Decode tokens/s | Total ms | BW SoL estimate |
|---|---:|---:|---:|---:|---:|
| vLLM | 188.556 | 3.659 | 273.32 | 243.500 | 36.40% |
| SGLang | 115.575 | 4.806 | 208.07 | 195.606 | 27.71% |
| Quickdraw | 9584.016 | 80.413 | 12.44 | 10929.769 | 1.66% |

Conditions: one warmup, five measured requests, greedy generation, EOS ignored,
BF16 KV, FP32 recurrent state, no prefix reuse, and no speculative decoding.
Baselines use CUDA Graphs; vLLM uses async scheduling. Each worker has two CPU
cores, one compiler worker, a 12 GiB host RAM cap, and no child swap.
Loading and warmup are excluded. TTFT includes first-token delivery to CPU.
Columns are independent medians, so they need not add up exactly.

Saved evidence:

- [Timing report](benchmark/runs/compare-20261004T224216.021733Z/timings.txt).
- [Samples, versions, settings, and comparison percentages](benchmark/runs/compare-20261004T224216.021733Z/results.json).
- [Full-model numerical comparison](benchmark/parity/vllm.json).
- [Kernel interfaces and cache contracts](docs/architecture.md).

The earlier numerical comparison matched 48/48 greedy choices, but failed its
strict numerical error limits: logit RMSE at most 0.25 and probability total
variation at most 0.05. Token agreement is not complete numerical parity.
Timing checks also do not prove model correctness.

## 2. Rules for the implementation

- [ ] Keep one request engine. Avoid executor subclasses and wrappers that only
  pass arguments through to another function.
- [ ] Keep model wiring in `quickdraw/models/`, request control in
  `quickdraw/engine/`, history buffers in `quickdraw/kvcache/`, and GPU computation
  in `quickdraw/kernel/`.
- [ ] Bind kernel choices before generation. Do not select a backend repeatedly
  inside the token loop.
- [ ] Preserve a readable reference path while replacing operations.
- [ ] Treat prefill and decode as separate computation paths. They share weights
  and history, but have different shapes and useful parallelism.
- [ ] Allocate persistent history and workspace before graph capture.
- [ ] Run GPU engines sequentially. Use the retained environments and
  `scripts/direct_run.py`; do not create services or extra virtual environments.
- [ ] Keep compiler workers at one and compiler CPU affinity at two cores.
- [ ] Record a change's output error, state error, timing, and memory cost before
  deciding to keep it.

The existing `step(checkpoint, token, position, state)` hook returns logits and
updates history. The engine advances position and selects the next token.
Keep that ownership explicit when extending the interfaces.

## 3. Correctness requirements for every optimization

- [ ] Check block output against saved independent references before measuring it.
- [ ] Check changed history as well as the output: KV slots, recurrent matrices,
  and convolution history.
- [ ] Check multiple processed positions, including position zero, later history,
  a context boundary, and a reset between requests.
- [ ] For sequence operations, compare final state and outputs against processing
  the same token IDs one at a time.
- [ ] Use the same forced token history for numerical comparisons. If engines
  generate different tokens, their later logits no longer describe the same input.
- [ ] Investigate the existing logit error before declaring vLLM/SGLang parity.
  Inspect activation scales, weight scales, norm conventions, accumulation, and
  rounding. Do not increase error limits merely to make a check pass.
- [ ] Set and document an error budget for each new precision change.
- [ ] Keep engine control checks separate from numerical checks and speed checks.

Preserve these request rules:

- Position counts tokens that have entered model state.
- A selected pending token has not yet entered model state.
- For a prompt of length P and N outputs, ordinary generation processes
  P + N - 1 tokens, unless EOS or capacity stops it earlier.
- The final returned output stays pending. Do not process it an extra time.
- Reset clears recurrent and convolution history and the valid KV length.
- A zero output limit does not modify existing state.
- Return EOS when selected, then stop before processing it again.

Useful checks:

```sh
uv run -m tests.models.mathcheck --strict
uv run -m tests.engine.check --strict
uv run -m unittest tests.benchmark.test_compare tests.benchmark.test_bench
.venv/bin/python scripts/direct_run.py --cache-tag quickdraw -- \
  .venv/bin/python -m tests.kernel.check
```

Compare against the existing full-model capture without starting a baseline engine:

```sh
.venv/bin/python scripts/direct_run.py --cache-tag quickdraw -- \
  .venv/bin/python -m benchmark.offline.parity compare --backend triton
```

That command may still fail the current strict limits. Keep the report and use it
to track error. Capturing a new baseline is a separate, explicit action.

## 4. First milestone: a proper prefill path

Current problem: `Engine.prefill` calls the one-token model step for every prompt
token. Each step reads weights and launches many small operations. It also runs
the vocabulary projection for every token, then discards all but the last logits.

### 4.1 Remove unused vocabulary projections

- [ ] Separate hidden-state execution from final normalization and the LM head,
  or add a small explicit execution option for whether logits are required.
- [ ] Run all layers for every prompt token so that history remains correct.
- [ ] Run final normalization and vocabulary projection only for the final prompt
  token when only the next-token logits are requested.
- [ ] Preserve the existing full `forward_step` behavior for decode and reference
  checks. Update callers and checks together if its contract changes.
- [ ] Compare returned logits and all final history against the original prefill.
- [ ] Measure the saved time. Treat this as a small first change, not a substitute
  for batched prefill.

### 4.2 Process multiple prompt tokens together

- [ ] Add an explicit prefill operation that accepts a token sequence and returns
  final-token logits plus correctly updated history.
- [ ] Embed the sequence into a hidden matrix `[prompt_tokens, hidden_size]`.
- [ ] Apply each layer to the token matrix or a bounded chunk. Do not turn the
  prompt into multiple independent requests: later tokens depend on earlier ones.
- [ ] Use matrix-matrix projections for prefill. The current GEMV kernels accept
  one hidden vector and are not a batched projection implementation.
- [ ] Support FP8 projections with the specified activation and weight scales.
- [ ] Support NVFP4 weight-only projections with BF16 activations.
- [ ] For full attention, apply head normalization and RoPE at each token's position,
  write its K/V, and enforce the causal mask.
- [ ] For Gated DeltaNet, implement a chunked recurrence with the correct initial
  state, convolution boundary history, and final state.
- [ ] For MoE, route each prompt token, group work by expert, compute the selected
  experts, and restore the original token order before residual addition.
- [ ] Calculate vocabulary logits only for the final prompt token.
- [ ] Bound temporary memory. Tune chunk size within the available GPU memory.

Use a trusted chunked GDN implementation as a reference while building the kernel:
[FLA gated delta rule](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py).
Borrowing a formulation or temporarily using an existing kernel does not prove
that Quickdraw's full prefill path is correct.

Acceptance requirements:

- [ ] Single-token and multi-token prompts give equivalent final logits and state
  within the declared error budget.
- [ ] Prompts that cross chunk boundaries retain correct convolution and recurrence.
- [ ] Decode after prefill agrees with decode after the sequential reference.
- [ ] Repeated requests and reset do not leak state.
- [ ] TTFT improves on the saved workload without increasing numerical error.
- [ ] Long prompts fit without allocating unbounded intermediates.

## 5. Second milestone: GPU expert routing and grouped MoE

Current problem: `moe_block` calls `.tolist()` on GPU expert IDs and executes the
selected experts in a Python loop. These transfers make the CPU wait for routing
before it can submit expert work. The model has 256 experts per layer and selects
8 per token, plus one shared expert.

- [ ] Keep router scores, selected IDs, and selected weights in GPU tensors.
- [ ] Preserve softmax in FP32, top-k selection, and normalization of selected
  probabilities. Preserve the shared expert and its sigmoid gate.
- [ ] Give expert kernels the packed expert bank and a GPU tensor of selected IDs.
  Read the correct expert offsets inside the kernel.
- [ ] Avoid copying complete expert weights into newly gathered temporary tensors.
- [ ] Compute gate and up projections together where useful, then apply
  `silu(gate) * up`.
- [ ] Compute down projections and weighted accumulation across selected experts.
- [ ] Preserve the reference's rounding points and accumulation contract. Parallel
  reduction order can change results; check the resulting error explicitly.
- [ ] Use fixed-size decode buffers for the selected experts and their activations.
- [ ] Add a grouped matrix-matrix path for prefill, where an expert can receive
  multiple prompt tokens.
- [ ] Check repeated expert IDs across tokens, expert-bank boundaries, and router
  scores that are close to a selection boundary.

Acceptance requirements:

- [ ] No expert-ID `.tolist()` or `.item()` remains in the GPU execution path.
- [ ] Output agrees with the MoE reference for several routing patterns.
- [ ] Nsight Systems shows fewer CPU/GPU waits and fewer expert launches.
- [ ] Complete decode gets faster; an isolated projection speedup is insufficient.

## 6. Third milestone: fixed workspace and CUDA Graph decode

CUDA Graphs submit previously defined GPU work with less repeated CPU setup.
They do not make incorrect state updates correct, remove all memory traffic,
or turn a Python expert loop into dynamic GPU routing.
[CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

### 6.1 Prepare the execution path

- [ ] Allocate stable token, position, intermediate, routing, and output buffers.
- [ ] Add output-buffer arguments where needed so kernels can reuse workspace.
- [ ] Prepare constant norm gamma and other constant derived values once, preserving
  the intended precision and avoiding a second `1 + weight` transformation.
- [ ] Finish weight repacking before capture. Stream repacking by layer if keeping
  both full old and new weight banks would exceed VRAM.
- [ ] Remove CPU decisions that depend on GPU values from the captured path.
- [ ] Make the current position a GPU input. A Python integer captured once does
  not automatically advance on replay.
- [ ] Make attention write the current slot and read only the valid prefix.

### 6.2 Capture and replay

- [ ] Warm up all required kernels before capture.
- [ ] Capture using scratch history. Do not let warmup or capture consume the live
  request's prompt or pending token.
- [ ] Bind the graph to the intended weights, workspace, and history addresses.
- [ ] Handle changing attention length with a fixed masked range or context buckets.
  Padding to maximum capacity can waste work; measure the bucket tradeoff.
- [ ] Cross a bucket boundary without losing history or processing a token twice.
- [ ] Reset in place without invalidating captured addresses.
- [ ] Initially preserve the existing step-to-logits interface. If greedy selection
  moves inside the graph later, extend the caller contract and its checks explicitly.
- [ ] Retain CPU token delivery where the public streaming API requires it. Avoid
  extra synchronization inside layers.

Acceptance requirements:

- [ ] Eager and graph execution agree at several positions and bucket boundaries.
- [ ] One replay processes exactly one pending token.
- [ ] Repeated reset and prefill do not require new addresses for request history.
- [ ] Graph outputs and state remain correct after many decode steps.
- [ ] Nsight Systems shows reduced submission gaps.
- [ ] Warm decode latency improves after including token delivery to CPU.

## 7. Fourth milestone: fuse and tune the measured expensive operations

Optimize the current bottleneck after each milestone. Its location can change.

- [ ] Fuse residual addition and RMS normalization where their inputs and consumers
  permit it. Keep variance accumulation in FP32.
- [ ] Fuse normalization and FP8 activation preparation where useful. If BF16
  consumers also need the normalized vector, preserve that representation.
- [ ] Fuse Q/K head normalization, RoPE, and KV writes where this reduces launches
  and temporary traffic.
- [ ] Implement GQA attention without materializing copies of KV for query heads.
- [ ] Fuse convolution, control-feature processing, and the GDN recurrent update
  where feasible. Do not overwrite state before its old value has been consumed.
- [ ] Fuse GDN output normalization and gating where the precision contract allows it.
- [ ] Tune NVFP4 and FP8 decode projections for their actual shapes, including the
  large vocabulary head. Sweep rows per program, warps, and reduction tiling.
- [ ] Inspect register use, spills, occupancy, and achieved bandwidth with NCU.
- [ ] Evaluate supported tensor-core matrix kernels for prefill. Confirm SM120
  support and the actual execution format instead of inferring it from a filename.
- [ ] Keep W4A16 and W4A4 experiments distinct. Quantizing activations to FP4 changes
  computation and requires an independent numerical check.
- [ ] Repack weights once if it improves access. Include startup time and peak VRAM
  in the evaluation, even though warm timings exclude loading.

Keep a fusion when it improves the complete workload and passes correctness
checks. Fewer launches can still lose performance through spills, extra work,
or poor access patterns.

## 8. Cache plan for one user

Start with contiguous history. The current model has 10 full-attention layers and
30 Gated DeltaNet layers. Queries are temporary; keys and values persist.

| State | Current storage cost |
|---|---:|
| BF16 KV per processed token | 20,480 bytes |
| BF16 KV at capacity 8192 | 160 MiB |
| BF16 KV at capacity 32768 | 640 MiB |
| FP32 GDN recurrent state | 60 MiB |
| BF16 convolution history | About 1.41 MiB |

- [ ] Keep direct slot indexing and a valid-prefix length for one active request.
- [ ] Choose capacity before allocation and graph capture. Do not grow buffers
  during replay.
- [ ] Keep cache addresses stable across reset.
- [ ] Avoid expanded GQA caches and unnecessary full-state copies.
- [ ] Measure attention traffic as context grows before changing the KV format.

### 8.1 Reuse a conversation prefix

- [ ] Detect an exact matching token prefix, including the chat template and all
  tokens already processed. Similar text is not sufficient.
- [ ] Retain its KV prefix, recurrent state, convolution history, and position.
- [ ] Process only the new suffix when the history is valid.
- [ ] Track the pending token explicitly. Do not include a token in reused history
  unless the model has processed it.
- [ ] Invalidate reuse when the checkpoint, relevant execution settings, or prefix
  changes.
- [ ] Start with one retained conversation boundary. Each additional GDN/conv
  snapshot costs about 61.4 MiB before its KV prefix is counted.
- [ ] Compare reused-prefix execution against a fresh full prefill.
- [ ] Benchmark reuse separately. The saved baseline workload has prefix reuse off.

### 8.2 Later cache experiments

- [ ] Evaluate bounded checkpoints at useful conversation or chunk boundaries.
  Trade snapshot memory and restore time against avoided prefill work.
- [ ] Consider paging only when multiple requests, shared prefixes, or allocation
  pressure justify the metadata and kernel changes.
- [ ] Evaluate FP8 KV at long context with explicit scales and quality checks.
- [ ] Treat reduced-precision recurrent state as a separate experiment. Error can
  accumulate over a long sequence even when a single update appears accurate.
- [ ] Do not introduce a sliding window unless the model's supported behavior or
  an explicit quality tradeoff permits dropping history.

A useful new cache design must reduce time or memory for an identified workload
and preserve the required model state. A different data structure alone is not
evidence of a faster inference engine.

## 9. Benchmark and profiling work

Default comparison command:

```sh
uv run benchmark/offline/compare.py
```

It runs Quickdraw against saved matched baselines. It never starts a baseline
engine because a saved result is missing. Baseline refresh is an explicit action:

```sh
uv run benchmark/offline/compare.py --refresh-baselines
```

Do not run that refresh during ordinary kernel iteration.

- [ ] Keep the saved 128/16 workload as the initial regression measurement.
- [ ] Save each optimized run's settings and samples. Record which source change
  produced it so that equal package version strings do not hide different code.
- [ ] Inspect spread and outliers as well as the median. Do not discard slow runs
  just to improve the displayed result.
- [ ] Add real chat, coding, and long-output workloads based on actual usage.
- [ ] Add longer contexts and prefix reuse as distinct benchmark conditions.
- [ ] When no matched baseline exists, time Quickdraw alone or explicitly arrange
  a new baseline capture. Do not compare unmatched workloads as a percentage gap.
- [ ] Share checkpoint-cache reclaim and RAM-throttling detection with standalone
  timing/profiling paths where needed, outside measured or captured ranges.
- [ ] Record GPU clocks, temperature, power, VRAM, and CPU conditions to help explain
  timing variation. Sampling must not dominate the measured work.
- [ ] Keep load time, cold compilation, warm TTFT, decode time, and total generation
  time separate.

Definitions:

- TTFT: time from request submission to delivery of the first output ID.
- Decode time/token: time between first and final output delivery divided by N-1.
- Decode TPS: reciprocal of decode time/token, not N divided by total time.
- Extra latency: `100 * (quickdraw_time / baseline_time - 1)`.
- TPS deficit: `100 * (1 - quickdraw_tps / baseline_tps)`.

The bandwidth SoL estimate models about 2.39 GB of traffic per decode step for
the saved workload. At [NVIDIA's advertised 1792 GB/s](https://images.nvidia.cn/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), this gives about 1.33 ms of ideal
bandwidth time. This is not a strict latency lower bound or measured utilization.
It ignores L2 reuse, backend layouts, intermediates, and extra passes. Do not use
all stored MoE weights as if every expert runs for every decode token.

- [ ] Use NCU counters for measured kernel compute and memory throughput.
- [ ] Do not treat NVML GPU-busy percentage as compute efficiency or bandwidth SoL.
- [ ] Do not treat one profiled NVFP4 launch as the utilization of the entire engine.
- [ ] Use Nsight Systems to inspect launch gaps, transfers, synchronization, and
  the complete prefill/decode timeline.
- [ ] Use Torch profiling to attribute operations and locate expensive calls.
- [ ] Use unprofiled runs for the comparison timing; profilers can perturb execution.

Commands:

```sh
uv run benchmark/offline/profile_torch.py
uv run benchmark/offline/profile_nsys.py
uv run benchmark/offline/profile_ncu.py --profile-phase decode
```

The NCU wrapper defaults to one matching NVFP4 launch. Change its kernel filter
and launch count deliberately to inspect other operations. Keep saved profiles
and commands alongside the change being evaluated.

## 10. Keep support for other models practical

- [ ] Read dimensions, layer types, and weight formats from configuration.
- [ ] Keep checkpoint interpretation separate from GPU weight layouts.
- [ ] Define kernel inputs, output shapes, layouts, dtypes, and mutation rules.
- [ ] Select specialized kernels from shape, format, device, and execution phase.
- [ ] Keep a correct fallback for unsupported shapes or formats.
- [ ] Bind choices once. Avoid a growing dispatch framework in the token loop.
- [ ] Add another model only after the Qwen path has useful correctness and speed
  evidence. Do not infer support from a shared model name or tokenizer.
- [ ] Add a small second-model fixture and real generation check before claiming
  that the engine is model agnostic.

The stable pieces should be request control, memory ownership, benchmarking,
and operator contracts. Each model still needs its correct layer wiring and
quantization interpretation.

## 11. Later experiments: persistent kernels, MTP, and speculative decode

### 11.1 Persistent kernels and megakernels

- [ ] Establish a graph-based decode measurement first.
- [ ] Use the remaining profile to identify whether launch overhead, traffic,
  compute, or synchronization limits the next improvement.
- [ ] Start with a persistent or fused block such as MoE or GDN before attempting
  a whole-model kernel.
- [ ] Design producer/consumer synchronization explicitly. Ordinary CUDA blocks
  cannot assume a global barrier between arbitrary blocks.
- [ ] Check resident-block limits and forward progress before using a persistent
  work queue or cooperative synchronization scheme.
- [ ] Account for register pressure, shared memory, intermediate storage, and
  parallelism across heads or experts.
- [ ] Compare against the graph implementation with the same precision and history.
- [ ] Keep the experiment only if end-to-end speed improves without excessive
  memory use or a fragile model-specific constraint.

A megakernel is a possible way to submit less work from the CPU and reuse more
intermediate data. It is not a requirement for a useful fast library.

### 11.2 MTP and speculative decoding

MTP means multi-token prediction. A draft mechanism proposes extra tokens; the
target model verifies them. More proposed tokens do not automatically mean more
accepted tokens per second.

- [ ] Finish a fast ordinary decode path first.
- [ ] Verify the checkpoint's MTP architecture and weights, and implement their
  actual computation. The current loader/reference path is not an MTP decoder.
- [ ] Implement target verification and acceptance before pursuing speed.
- [ ] For greedy generation, emit only the sequence accepted by target verification.
- [ ] Preserve history at the start of a speculative block.
- [ ] On rejection, restore the correct KV length, GDN state, convolution state,
  position, and pending-token boundary.
- [ ] Choose between snapshots of intermediate states and replay of accepted
  tokens. Measure their memory and latency costs.
- [ ] Do not rewind only KV: this hybrid model also mutates recurrent and conv state.
- [ ] Check rejection at every draft position, all-accepted blocks, EOS, reset,
  and context limits.
- [ ] Measure acceptance rate, verification cost, rollback cost, and accepted TPS
  on real prompts.
- [ ] Compare speculative engines with explicitly matched speculative conditions,
  separately from the existing no-MTP baseline.

## 12. Delivery order and completion criteria

Work in this order, with a measured checkpoint after each stage:

1. Preserve and investigate numerical correctness; remove unused prefill logits.
2. Build batched/chunked prefill with correct final history.
3. Move routing and selected expert computation onto the GPU.
4. Reuse workspace and capture correct decode graph replay.
5. Fuse and tune the operations that the new profile identifies.
6. Add exact conversation-prefix reuse and real usage benchmarks.
7. Add a second supported model and validate the interfaces.
8. Evaluate persistent kernels, novel cache checkpoints, and MTP/speculation.

Prefill and GPU-routing work can share projection and workspace preparation, but
do not let a broad refactor hide whether either change actually improves speed.

If the deadline is close, prioritize a correct batched prefill path, GPU MoE
routing, and stable graph decode. Defer multi-user scheduling, paged allocation,
whole-model megakernels, and speculative decoding until their measured benefit
justifies the implementation work.

Before calling a milestone complete:

- [ ] Relevant output, history, and engine checks pass at stated tolerances.
- [ ] Saved baseline comparison uses matching conditions and starts no baseline
  engines unless refresh was explicitly requested.
- [ ] Raw samples, medians, source identity, and profiling evidence are saved.
- [ ] Peak GPU and host memory fit the intended limits.
- [ ] Startup and warm execution costs are documented separately.
- [ ] At least one real prompt produces a useful response with normal EOS handling.
- [ ] A fresh request, a repeated request, and a continued conversation behave
  correctly for the supported feature set.
- [ ] Documentation says exactly which model, formats, shapes, and devices work.

Only claim to beat vLLM or SGLang for the workload and conditions that were
actually measured. Use the saved baseline as an iteration target, and use an
explicit later refresh when making a claim about current competing versions.
