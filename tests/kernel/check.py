"""Verify fused CUDA projections, then optionally measure their warm latency."""

import argparse
import json
from pathlib import Path

import torch
import triton.testing

from quickdraw.models import qwen as model
from quickdraw.models.weights import FP8Linear, NVFP4Linear
from quickdraw.kernel.triton import matvec
from tests.models.mathcheck import run_reference


def check_saved_math():
    kernels = {"fp8_gemv": matvec.fp8_gemv, "nvfp4_gemv": matvec.nvfp4_gemv}
    directory = Path(__file__).resolve().parents[2] / "references" / "model"
    paths = sorted(directory.glob("*.pt"))
    assert paths, "independent model references are missing"
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        run_reference(data, torch.device("cuda"), kernels=kernels)
        print(f"PASS Triton backend: {path.name}", flush=True)


def projection(kind, rows, width):
    if kind == "fp8":
        return FP8Linear((torch.randn(rows, width, device="cuda")*.5).to(torch.float8_e4m3fn),
                         torch.tensor(.125, device="cuda"), torch.tensor(.25, device="cuda"))
    return NVFP4Linear(torch.randint(0, 256, (rows, width//2), device="cuda", dtype=torch.uint8),
                        (torch.randint(1, 5, (rows, width//16), device="cuda").float()/4).to(torch.float8_e4m3fn),
                        torch.tensor(.25, device="cuda"), torch.tensor(3., device="cuda"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("benchmark/kernels/matvec.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(7)
    with torch.inference_mode():
        check_saved_math()
        shapes = [("nvfp4", 13, 16), ("nvfp4", 512, 2048), ("nvfp4", 2048, 512),
                  ("fp8", 13, 48), ("fp8", 8192, 2048), ("fp8", 2048, 4096)]
        if args.benchmark:
            shapes.append(("nvfp4", 248320, 2048))
        results = []
        for kind, rows, width in shapes:
            proj = projection(kind, rows, width)
            x = torch.randn(width, dtype=torch.bfloat16, device="cuda")
            reference = model.nvfp4_gemv if kind == "nvfp4" else model.fp8_gemv
            fused = matvec.nvfp4_gemv if kind == "nvfp4" else matvec.fp8_gemv
            expected, actual = reference(x, proj), fused(x, proj)
            torch.testing.assert_close(actual, expected, rtol=.02, atol=.002)
            result = dict(kind=kind, rows=rows, input_width=width,
                          max_abs_error=(actual.float()-expected.float()).abs().max().item())
            if args.benchmark:
                result["torch_ms"] = triton.testing.do_bench(lambda: reference(x, proj), warmup=10, rep=50)
                result["fused_ms"] = triton.testing.do_bench(lambda: fused(x, proj), warmup=10, rep=50)
                result["projection_speedup"] = result["torch_ms"] / result["fused_ms"]
            results.append(result)
            print(json.dumps(result), flush=True)
            del proj, x, expected, actual
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__, results=results), indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
