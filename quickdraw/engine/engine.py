"""Single-request state and greedy generation."""

import torch

from ..kernel import bind_kernels
from ..kvcache.state import DecodeState
from ..models.weights import Checkpoint, load_checkpoint


class Engine:
    def __init__(self, ckpt: Checkpoint, max_context: int = 32768,
                 device: str = "cuda", backend: str = "reference", *,
                 kernels: dict | None = None, step=None):
        """Allocate history and bind kernels or step(checkpoint, token, position, state).
        The step updates buffers and returns logits; the engine owns position."""
        assert backend in ("reference", "triton")
        assert kernels is None or step is None
        assert backend == "reference" or (kernels is None and step is None)
        custom = kernels is not None or step is not None
        if backend == "triton":
            assert torch.device(device).type == "cuda"
            from ..kernel.triton.matvec import fp8_gemv, nvfp4_gemv
            kernels = {"fp8_gemv": fp8_gemv, "nvfp4_gemv": nvfp4_gemv}

        self.ckpt = ckpt
        self.max_context = max_context
        self.device = device
        self.backend = "custom" if custom else backend
        self.step = step if step is not None else bind_kernels(**(kernels or {}))["forward_step"]
        assert max_context > 0
        config = ckpt.config
        full = config.layer_types.count("full_attention")
        linear = config.layer_types.count("linear_attention")
        channels = (2 * config.linear_num_key_heads * config.linear_key_head_dim
                    + config.linear_num_value_heads * config.linear_value_head_dim)
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

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "Engine":
        device = kwargs.get("device", "cuda")
        return cls(load_checkpoint(path, device=device), **kwargs)

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> torch.Tensor:
        """Start a fresh request; process the prompt and return first-output logits."""
        assert 0 < len(token_ids) <= self.max_context, "prompt must fit within max_context"
        self.reset()
        tokens = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        for token in tokens:
            logits = self.step(self.ckpt, token, self.state.position, self.state)
            self.state.position += 1
        return logits

    @torch.inference_mode()
    def decode_step(self) -> torch.Tensor:
        """Process the pending token once; select and return its successor."""
        assert self.pending_token is not None, "select a pending token before decode"
        assert self.state.position < self.max_context, "context capacity reached"
        logits = self.step(self.ckpt, self.pending_token, self.state.position, self.state)
        self.state.position += 1
        self.pending_token = torch.argmax(logits)
        return self.pending_token

    @torch.inference_mode()
    def generate(self, token_ids: list[int], max_tokens: int) -> list[int]:
        """Return new IDs, including EOS. Stop before another step exceeds capacity."""
        assert max_tokens >= 0, "max_tokens must be nonnegative"
        if max_tokens == 0:
            return []

        logits = self.prefill(token_ids)
        self.pending_token = torch.argmax(logits)
        output = [int(self.pending_token.item())]
        while (len(output) < max_tokens and output[-1] not in self.eos_token_ids
               and self.state.position < self.max_context):
            output.append(int(self.decode_step().item()))
        return output

    def reset(self) -> None:
        """Clear request history while retaining buffer addresses and EOS settings."""
        self.state.rec_state.zero_()
        self.state.conv_state.zero_()
        self.state.position = 0
        self.pending_token = None
