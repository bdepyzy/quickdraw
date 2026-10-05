# Model reference cases

Checks:

```sh
uv run -m tests.models.mathcheck
uv run -m tests.models.mathcheck --only full_attention_block
uv run -m tests.models.mathcheck --strict
```

The checker loads all `.pt` files in this directory automatically and requires
Torch, without Transformers. A pass confirms the case's output and state within
its tolerances. Skipped cases remain unverified. Coverage does not establish
correctness for every input or complete engine parity.

| Function | Cases | What changes |
|---|---:|---|
| `full_attention_block` | 3 | Positions 0, 2, and 4; empty/history/full cache; GQA head counts; partial/full RoPE; epsilon |
| `gated_delta_net_block` | 3 | Zero/nonzero recurrent and convolution history; head dimensions; convolution widths; epsilon |
| `moe_block` | 3 | Top-1, top-2, and top-3 routing; 4 or 6 experts; distinct expert weights and scales |
| `forward_step` | 6 | Two alternating layer orders; positions 0, 1, and 3; accumulated history; all vocabulary logits |

All weights are small, synthetic, and nonzero. FP8 projections have non-unit
activation and weight scales. NVFP4 projections contain independently varied
nibbles, block scales, and per-expert global scales. Unused KV slots contain
finite sentinel values. Expected state includes these slots, so unintended
writes fail the comparison. The default tolerance is `rtol=0.02`, `atol=0.002`
for BF16 output and state comparisons, including FP32 recurrent state. These
tolerances permit rounding differences; they do not require bitwise equality.

Each file contains pre-call inputs, expected output, expected post-call state,
comparison tolerances, and source metadata. `manifest.json` records dependency
versions and the SHA256 hashes of the HF source and generator. Loading uses
`torch.load(weights_only=True)`; project dataclasses are encoded as dictionaries.

The expected values come from the installed **Transformers 5.17.0**
[Qwen3.5 MoE implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).
The local Qwen3.6 checkpoint uses the same `qwen3_5_moe_text` architecture.
The generator does not call `quickdraw.models.qwen` math or checkpoint dequantization.
HF performs the attention, recurrence, routing, residual connections, and norms.
Attention uses the eager backend; experts use the explicit `grouped_mm` backend
for both isolated blocks and complete text models. RoPE frequencies remain FP32:
the generator rebuilds HF's rotary module after casting model weights to BF16.
Independent format adapters implement the project's static FP8 and weight-only
NVFP4 contracts. A cache adapter maps HF's single-token operations onto
quickdraw's buffers. Text-only RoPE uses the same position on all three axes.

These are CPU formula references, not captures of SGLang/vLLM quantized kernels.
They do not exercise the 35B checkpoint, multimodal inputs, batched prefill,
CUDA graphs, or inference performance. Actual-checkpoint engine parity remains
a separate check. No GPU compilation or model download occurs during capture.

## Contracts

- Block functions receive `config` for epsilon, RoPE, dimensions, and routing.
- Query/gate packing is per head.
- Layer, final, and Q/K RMSNorm gamma is `1 + stored_weight`. DeltaNet RMSNorm
  uses its stored weight directly with a SiLU output gate.
- Selected MoE probabilities sum to one. Rounded expert contributions accumulate
  in FP32; output is BF16 and includes the shared expert.
- NVFP4 calibration `input_scale` is metadata only for W4A16.
- `forward_step` mutates tensor history without advancing `state.position`.
  Convolution history retains the last `kernel_size - 1` raw projected QKV vectors.

## Regeneration

The retained environment contains Transformers:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/direct_run.py --cache-tag references -- \
  .venv-vllm/bin/python -m tests.models.referencegen
```

This limits host RAM to 12 GiB, disables workload swap, and uses two CPU cores
with one Torch worker. The generator writes only its named files and manifest.
