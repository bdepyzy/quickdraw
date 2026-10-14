"""CuTe NVFP4 correctness and nondefault-stream/graph checks.

Run through scripts/direct_run.py. Performance is measured with
benchmark/offline/bench.py; this module checks CuTe against reference math.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from quickdraw.kernel.cute import matvec
from quickdraw.kernel.cute.moe import moe_block, route
from quickdraw.models import qwen
from quickdraw.models.weights import NVFP4Linear
from tests.models.mathcheck import run_reference


def projection(rows, width, experts=None, global_bank=False):
    shape = (rows,) if experts is None else (experts, rows)
    return NVFP4Linear(
        torch.randint(0, 256, (*shape, width // 2), device="cuda", dtype=torch.uint8),
        (torch.randint(1, 9, (*shape, width // 16), device="cuda").float() / 8).to(torch.float8_e4m3fn),
        torch.rand(experts if global_bank else 1, device="cuda", dtype=torch.float32) * .25 + .01)


def close(actual, expected):
    # BF16 outputs and independently accumulated reference reductions.
    torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)


def expert_reference(x, proj, ids, routing=None):
    return torch.stack([
        qwen.nvfp4_gemv(x if x.ndim == 1 else x[slot], qwen._expert_projection(proj, expert),
                       routing_weight=None if routing is None else routing[slot])
        for slot, expert in enumerate(ids.tolist())])


def check():
    torch.manual_seed(7)
    for rows, width in ((13, 16), (17, 48), (512, 2048), (2048, 512)):
        proj = projection(rows, width)
        x = torch.randn(width, device="cuda", dtype=torch.bfloat16)
        for routing in (None, torch.tensor(.317, device="cuda")):
            close(matvec.nvfp4_gemv(x, proj, routing_weight=routing),
                  qwen.nvfp4_gemv(x, proj, routing_weight=routing))
        up = projection(rows, width)
        close(matvec.gate_up(x, proj, up), F.silu(qwen.nvfp4_gemv(x, proj)) * qwen.nvfp4_gemv(x, up))
    for global_bank in (False, True):
        ids = torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32)
        routing = torch.tensor([.2, .3, .5], device="cuda")
        gate = projection(17, 48, 5, global_bank)
        up = projection(17, 48, 5, global_bank)
        x = torch.randn(48, device="cuda", dtype=torch.bfloat16)
        close(matvec.expert_gemv(x, gate, ids), expert_reference(x, gate, ids))
        close(matvec.gate_up(x, gate, up, ids), F.silu(expert_reference(x, gate, ids)) * expert_reference(x, up, ids))
        per_slot = torch.randn(3, 48, device="cuda", dtype=torch.bfloat16)
        close(matvec.expert_gemv(per_slot, gate, ids, routing), expert_reference(per_slot, gate, ids, routing))
        close(matvec.expert_gemv(x, gate, ids, routing), expert_reference(x, gate, ids, routing))
    # Capture on a nondefault stream, then mutate the input and verify replay.
    proj = projection(17, 48)
    x = torch.randn(48, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            matvec.nvfp4_gemv(x, proj)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = matvec.nvfp4_gemv(x, proj)
    for _ in range(2):
        x.normal_()
        graph.replay()
        close(actual, qwen.nvfp4_gemv(x, proj))
    # All 256 byte codes and signed zero, through the scalar-load path.
    proj = NVFP4Linear(torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8).reshape(256, 1),
                      torch.ones((256, 1), device="cuda").to(torch.float8_e4m3fn),
                      torch.ones(1, device="cuda"), group_size=2)
    x = torch.tensor([1., 3.], device="cuda", dtype=torch.bfloat16)
    close(matvec.nvfp4_gemv(x, proj), qwen.nvfp4_gemv(x, proj))
    for experts, topk in ((7, 1), (33, 8), (256, 8), (7, 7)):
        for logits in (torch.zeros(experts, device="cuda", dtype=torch.bfloat16),
                       torch.randn(experts, device="cuda", dtype=torch.bfloat16),
                       torch.arange(experts, device="cuda", dtype=torch.bfloat16) * 100):
            ids, weights = route(logits, topk)
            probabilities = logits.float().softmax(-1)
            expected_ids = probabilities.argsort(descending=True, stable=True)[:topk]
            torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
            expected_weights = probabilities[expected_ids]
            torch.testing.assert_close(weights, expected_weights / expected_weights.sum(), rtol=2e-6, atol=1e-7)
    directory = Path(__file__).resolve().parents[2] / "references/model"
    for path in sorted(directory.glob("moe-*.pt")):
        run_reference(torch.load(path, weights_only=True), torch.device("cuda"), kernels={"moe_block": moe_block})
        print(f"PASS CuTe independent fixture: {path.name}", flush=True)
    print("PASS CuTe: packed FP4, scalar/expert scales, tails, routed rounding, fused gate/up, stream and graph replay", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("benchmark/runs/cute-check.json"))
    args = parser.parse_args()
    with torch.inference_mode():
        check()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__, correctness="passed"), indent=2) + "\n")


if __name__ == "__main__":
    main()
