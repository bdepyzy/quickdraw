# TODO

BS=1, Qwen3.6-35B-A3B-NVFP4, RTX 5090. [Runtime contracts](docs/architecture.md).
Links point to current edit locations; line numbers will change as code changes.

## Target: CuTe DSL on SM120

Specialize for this checkpoint, one RTX 5090, and BS=1. New GPU kernels use CuTe
DSL. The existing Triton path remains the measured comparison during migration.

- [x] Packed W4A16 decode and selected-expert reads in [cute/matvec.py](quickdraw/kernel/cute/matvec.py), with fused gate/up/SiLU and routed-down BF16 rounding.
- [x] CuTe decode MoE routing and reduction in [cute/moe.py](quickdraw/kernel/cute/moe.py); independent MoE fixtures, routing ties, and nondefault-stream graph replay in [cute_check.py](tests/kernel/cute_check.py). Measure performance only with `benchmark/offline/bench.py`; select experimental kernels with `--cute-decode`.
- [ ] Replace prefill expert and dense projections with native SM120 tensor-core GEMMs. These currently account for about 93% of profiled prefill GPU time. Validate W4A4 activation quantization separately from W4A16.
- [ ] Implement static-FP8 decode projections and fuse residual addition/RMSNorm with consumer-specific activation quantization.
- [ ] Implement split-KV decode attention and fuse DeltaNet stages, preserving FP32 recurrence and BF16 convolution rounding.
- [ ] Bind a complete CuTe backend with no Triton fallback, including sequence prefill, CUDA Graph decode, and the existing state contracts. Compare the same 1024/1024 workload and check checkpoint-level numerical drift.
- [ ] Evaluate persistent whole-block/layer kernels against graph decode using a safe resident grid and explicit synchronization. Retain only measured end-to-end improvements.

## 1. Kernels

- [ ] [moe.py:61](quickdraw/kernel/triton/moe.py#L61), [moe.py:101](quickdraw/kernel/triton/moe.py#L101): combine gate/up expert reads and SiLU multiplication; write activated features directly instead of separate gate/up tensors.
- [ ] [norm.py:9](quickdraw/kernel/triton/norm.py#L9), [qwen.py:213](quickdraw/models/qwen.py#L213): fuse residual addition, RMSNorm, and FP8 activation quantization; preserve BF16 residual storage and each consumer's input scale.
- [ ] [delta.py:38](quickdraw/kernel/triton/delta.py#L38), [delta.py:57](quickdraw/kernel/triton/delta.py#L57), [delta.py:84](quickdraw/kernel/triton/delta.py#L84): evaluate fusion of convolution, controls, recurrence, and output gating; avoid unsafe synchronization between thread blocks.
- [ ] [matvec.py:23](quickdraw/kernel/triton/matvec.py#L23), [matvec.py:49](quickdraw/kernel/triton/matvec.py#L49): tune tile sizes, warps, and reduction layouts for actual projection shapes, including the vocabulary head.
- [ ] [bench.py](benchmark/offline/bench.py): use `--profile ncu --ncu-kernel` for the kernel under investigation; inspect registers, spills, occupancy, and memory throughput.
- [ ] [bench.py:74](benchmark/offline/bench.py#L74), [examples/offline.py:54](examples/offline.py#L54): compare unprofiled generation time, load/repacking time, and peak VRAM before retaining an optimization.

## 2. Conversation reuse

Keep contiguous KV for BS=1.

- [ ] [engine.py:93](quickdraw/engine/engine.py#L93): add a prefix-reuse entry point that checks token IDs and processes only the new suffix; keep fresh prefill's reset behavior.
- [ ] [state.py:9](quickdraw/kvcache/state.py#L9), [engine.py:143](quickdraw/engine/engine.py#L143): snapshot KV validity, recurrent/conv tensors, position, and pending token together; restore by copying into existing buffers.
- [ ] [engine.py:86](quickdraw/engine/engine.py#L86): bind snapshots to checkpoint and execution settings; invalidate when either or the token prefix changes.
- [ ] [integrationcheck.py:90](tests/models/integrationcheck.py#L90): compare reused and fresh prefill logits/state, including divergent suffixes and reset.
- [ ] [state.py:19](quickdraw/kvcache/state.py#L19): measure cache/snapshot bytes and reuse latency before adding bounded snapshots or FP8 KV; FP8 KV also needs scales and attention-kernel support.
- [ ] [attention.py:47](quickdraw/kernel/triton/attention.py#L47), [attention.py:69](quickdraw/kernel/triton/attention.py#L69): add a page table only if measured concurrency or allocation pressure requires it; change both KV writes and attention reads.

## Later

- [ ] [weights.py:126](quickdraw/models/weights.py#L126), [kernel/__init__.py:6](quickdraw/kernel/__init__.py#L6): add explicit model wiring and independent fixtures for a second model, then verify real generation.
- [ ] [triton/__init__.py:4](quickdraw/kernel/triton/__init__.py#L4): bind an experimental persistent block kernel and compare against graph decode, accounting for synchronization and resource limits.
- [ ] [weights.py:187](quickdraw/models/weights.py#L187), [engine.py:130](quickdraw/engine/engine.py#L130): use MTP weights for draft generation, add target verification/acceptance, and restore all history on rejection.
- [ ] [compare.py:307](benchmark/offline/compare.py#L307): add chat/coding prompts, long contexts, and a prefix-reuse workload.
- [ ] [parity.py:162](benchmark/offline/parity.py#L162): revisit full-model numerical drift later; retain existing captures and limits.
