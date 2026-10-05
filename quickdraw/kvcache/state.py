"""Fixed request buffers for batch-one decoding."""

from dataclasses import dataclass

import torch


@dataclass
class DecodeState:
    """Fixed KV, recurrent, and convolution history; shapes are in docs/architecture.md.
    position counts processed tokens and bounds valid KV slots."""

    kv_k: torch.Tensor
    kv_v: torch.Tensor
    rec_state: torch.Tensor
    conv_state: torch.Tensor
    position: int

    def memory_bytes(self) -> dict[str, int]:
        return {name: getattr(self, name).nbytes
                for name in ("kv_k", "kv_v", "rec_state", "conv_state")}
