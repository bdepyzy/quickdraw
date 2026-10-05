# Runtime interfaces

The engine owns request buffers, position, token selection, EOS, and capacity.
Model functions update history and return logits. GPU implementations bind as
plain functions before generation.

## Kernel binding

`Engine(checkpoint, kernels=overrides)` accepts replacements for `rms_norm`,
`fp8_gemv`, `nvfp4_gemv`, `full_attention_block`, `gated_delta_net_block`, and
`moe_block`. Reference blocks inherit projection replacements. Whole-block
replacements own their internal operations and use the reference signature,
including `config`. Blocks return BF16 hidden vectors and update supplied history
in place. RMSNorm receives gamma directly; Qwen layer, final, and Q/K norms
supply `1 + stored_weight`.

`backend="triton"` binds the included FP8 and NVFP4 projections and requires CUDA.
Custom `kernels` or `step` require `backend="reference"` and are mutually exclusive.
Input contracts use asserts; assertions must remain enabled.

## Token execution

`Engine(checkpoint, step=fn)` calls `fn(checkpoint, token, position, state)` once
per processed token. The token is a scalar `torch.long` tensor. The function
updates history, returns vocabulary logits, and leaves `state.position` unchanged.
The engine advances position after the call.

Prefill resets history and processes prompt IDs in order. Its final logits select
the first output without an additional forward call. A pending token has been
selected but has not entered model state. Decode processes that token and selects
its successor. The final returned token remains pending. An uninterrupted request
with P prompt tokens and N outputs processes P + N - 1 tokens.

A custom graph step owns capture, replay, workspace, and stable device inputs and
outputs. Changing positions require device-side indexing and masking or separate
captures for context ranges. Capture uses scratch history; replay binds to the
checkpoint, device, and request buffer addresses. The reference path's Python
slicing, CPU expert routing, and temporary allocations prevent reusable capture
across changing positions. Generation also reads selected IDs on CPU for EOS and
output delivery. The runtime does not implement graph capture or a megakernel.

## Request history

Buffers are contiguous, allocated once, and retain their addresses across reset.
Capacity remains fixed for the engine's lifetime.

| Buffer | Shape | Dtype |
|---|---|---|
| Keys and values, separate | `[full_layers, capacity, kv_heads, head_dim]` | BF16 |
| DeltaNet recurrence | `[linear_layers, value_heads, key_dim, value_dim]` | FP32 |
| Convolution history | `[linear_layers, channels, kernel_width - 1]` | BF16 |

`channels = 2 * key_heads * key_dim + value_heads * value_dim` for linear attention.
At position p, full attention writes normalized, RoPE-transformed K and projected
V to slot p, then attends through slots 0..p. Queries are temporary. GQA maps query
heads to KV heads; optimized kernels can share KV without materializing expansion.
The reference path materializes repeated heads.

Reset clears recurrence and convolution history, sets position to zero, and clears
the pending token. EOS settings remain. Unused KV bytes remain outside the valid
prefix. `state.memory_bytes()` reports allocated sizes.

For Qwen3.6-35B-A3B, 10 full-attention layers, 2 KV heads, and head dimension 256
require 20,480 KV bytes per token: 160 MiB at capacity 8192 or 640 MiB at 32768.
The 30 linear-attention layers require 60 MiB of FP32 recurrence and about 1.41 MiB
of BF16 convolution history, independent of context length.

The current single-request cache uses direct slot indexing. Paging supports
multiple live requests and shared prefixes. Prefix reuse and speculative rollback
require matching recurrent and convolution snapshots as well as KV history;
rewinding the KV position alone cannot restore this hybrid model's state.

## Weight formats

The loader reads modelopt mixed-precision language weights, excludes vision, and
stacks experts on the leading axis. MTP loading is optional.

| Format | Stored tensors | Reference interpretation |
|---|---|---|
| NVFP4 W4A16 | U8 weight `[out, in/2]`, E4M3 scale `[out, in/16]`, FP32 global scale | Two E2M1 values per byte, low nibble at even input; one block scale per 16 inputs |
| Static FP8 | E4M3 weight `[out, in]`, FP32 weight and input scales | Quantized activation and weight, each with its own scale |
| Raw BF16 | Router, shared gate, embeddings, norms, linear control projections, convolution, decay parameters, MTP | Unquantized tensors |

NVFP4 dequantization multiplies decoded E2M1 values by repeated E4M3 block scales
and the global scale, then casts weights to BF16. Calibration `input_scale` is
metadata only in the W4A16 path. Static FP8 divides activations by `input_scale`,
saturates and converts to E4M3, then applies both activation and weight scales.

[Independent fixtures](../references/model/README.md) cover block outputs and
history mutations. Complete engine parity is measured separately.
