"""Single-request state and greedy generation."""

import torch

from ..kernel import bind_kernels
from ..kvcache.state import DecodeState
from ..models.weights import Checkpoint, load_checkpoint, merge_fp8_scales


class Engine:
    def __init__(self, ckpt: Checkpoint, max_context: int = 32768,
                 device: str = "cuda", backend: str | None = None, *,
                 kernels: dict | None = None, step=None, cuda_graph: bool = True,
                 prefill_chunk_size: int = 128, sequence_prefill: bool = True):
        """Allocate history and bind kernels or step(checkpoint, token, position, state).
        The step updates buffers and returns logits; the engine owns position."""
        custom = kernels is not None or step is not None
        if backend is None:
            backend = "reference" if custom or torch.device(device).type == "cpu" else "triton"
        assert backend in ("reference", "triton")
        assert kernels is None or step is None
        assert backend == "reference" or step is None
        if backend == "triton":
            assert torch.device(device).type == "cuda"
            from ..kernel.triton import kernels as triton_kernels
            kernels = triton_kernels() | (kernels or {})

        self.ckpt = ckpt
        self.max_context = max_context
        self.device = device
        self.backend = "custom" if custom and backend == "reference" else backend
        prepared = None
        if backend == "triton":
            prepared = ([(1 + layer.input_layernorm.float(), 1 + layer.post_attention_layernorm.float()) for layer in ckpt.layers], 1 + ckpt.final_norm.float())

        self.step = step if step is not None else bind_kernels(prepared=prepared, **(kernels or {}))["forward_step"]
        assert max_context > 0
        assert prefill_chunk_size > 0
        self.prefill_chunk_size = prefill_chunk_size
        self.sequence_prefill = backend == "triton" and sequence_prefill
        self.graph = None
        config = ckpt.config
        full = config.layer_types.count("full_attention")
        linear = config.layer_types.count("linear_attention")
        channels = (2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim)

        kv_shape = (full, max_context, config.num_key_value_heads, config.head_dim)
        self.state = DecodeState(
            kv_k=torch.empty(kv_shape, dtype=torch.bfloat16, device=device),
            kv_v=torch.empty(kv_shape, dtype=torch.bfloat16, device=device),
            rec_state=torch.zeros(
                (linear, config.linear_num_value_heads,
                 config.linear_key_head_dim, config.linear_value_head_dim),
                dtype=torch.float32, device=device),
            conv_state=torch.zeros(
                (linear, channels, config.linear_conv_kernel_dim - 1),
                dtype=torch.bfloat16, device=device),
            position=0,
        )
        self.pending_token: torch.Tensor | None = None
        self.eos_token_ids: set[int] = set()
        if backend == "triton" and cuda_graph: self.capture_decode()

    @torch.inference_mode()
    def capture_decode(self):
        """Capture once before requests. Graph allocations retain a private workspace."""
        assert self.backend == "triton" and self.state.position == 0
        self._decode_input = torch.zeros((), dtype=torch.long, device=self.device)
        self._device_position = torch.zeros((), dtype=torch.int32, device=self.device)
        self._decode_output = torch.empty((), dtype=torch.long, device=self.device)
        scratch = DecodeState(torch.empty_like(self.state.kv_k), torch.empty_like(self.state.kv_v), torch.zeros_like(self.state.rec_state), torch.zeros_like(self.state.conv_state), 0)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))

        with torch.cuda.stream(stream):
            for _ in range(3):
                self.step(self.ckpt, self._decode_input, self._device_position, scratch)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            logits = self.step(self.ckpt, self._decode_input, self._device_position, self.state)
            torch.argmax(logits, out=self._decode_output)
        self.reset()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Engine":
        device = kwargs.get("device", "cuda")
        ckpt = load_checkpoint(path, device=device)
        merge_fp8_scales(ckpt)
        return cls(ckpt, **kwargs)

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> torch.Tensor:
        """Start a fresh request; process the prompt and return first-output logits."""
        assert 0 < len(token_ids) <= self.max_context, "prompt must fit within max_context"
        self.reset()
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        if self.sequence_prefill:
            for start in range(0, len(tokens), self.prefill_chunk_size):
                chunk = tokens[start:start + self.prefill_chunk_size]
                final = start + len(chunk) == len(tokens)
                logits = self.step(self.ckpt, chunk, self.state.position, self.state, logits=final)
                self.state.position += len(chunk)
            return logits
        for token in tokens:
            if self.backend == "triton":
                logits = self.step(self.ckpt, token, self.state.position, self.state, logits=self.state.position == len(tokens) - 1)
            else:
                logits = self.step(self.ckpt, token, self.state.position, self.state)
            self.state.position += 1
        return logits

    @torch.inference_mode()
    def decode_step(self) -> torch.Tensor:
        """Process the pending token once; select and return its successor."""
        assert self.pending_token is not None, "select a pending token before decode"
        assert self.state.position < self.max_context, "context capacity reached"
        if self.graph is None:
            logits = self.step(self.ckpt, self.pending_token, self.state.position, self.state)
            self.pending_token = torch.argmax(logits)
        else:
            self._decode_input.copy_(self.pending_token)
            self._device_position.fill_(self.state.position)
            self.graph.replay()
            self.pending_token = self._decode_output.clone()
        self.state.position += 1
        return self.pending_token

    @torch.inference_mode()
    def generate(self, token_ids: list[int], max_tokens: int) -> list[int]:
        """Return new IDs, including EOS. Stop before another step exceeds capacity."""
        assert max_tokens >= 0, "max_tokens must be nonnegative"
        if max_tokens == 0: return []

        logits = self.prefill(token_ids)
        self.pending_token = torch.argmax(logits)
        output = [int(self.pending_token.item())]
        while (len(output) < max_tokens and output[-1] not in self.eos_token_ids and self.state.position < self.max_context):
            output.append(int(self.decode_step().item()))

        return output

    def reset(self) -> None:
        """Clear request history while retaining buffer addresses and EOS settings."""
        self.state.rec_state.zero_()
        self.state.conv_state.zero_()
        self.state.position = 0
        self.pending_token = None
