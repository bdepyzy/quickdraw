"""Verify Triton blocks and projections; optionally time warm projections."""

import argparse
import json
from pathlib import Path

import torch
import triton.testing

from quickdraw.models import qwen as model
from quickdraw.models.weights import FP8Linear, NVFP4Linear
from quickdraw.kernel.triton import matvec
from tests.models.mathcheck import run_reference


def check_saved_math(device="cuda"):
    from quickdraw.kernel.triton import kernels as triton_kernels
    kernels = triton_kernels()
    directory = Path(__file__).resolve().parents[2] / "references" / "model"
    paths = sorted(directory.glob("*.pt"))
    assert paths, "independent model references are missing"
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        run_reference(data, torch.device(device), kernels=kernels)
        print(f"PASS Triton backend: {path.name}", flush=True)


def compile_kernels():
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile
    from quickdraw.kernel.triton.delta import _conv_kernel, _delta_kernel, _gate_kernel
    from quickdraw.kernel.triton.moe import _route_kernel, _silu_kernel, _reduce_kernel
    from quickdraw.kernel.triton.norm import _rms_kernel

    cases = [
        (_rms_kernel, dict(X="*bf16", W="*fp32", Out="*bf16"), dict(N=2048, EPS=1e-6, BLOCK=2048)),
        (_conv_kernel, dict(X="*bf16", W="*bf16", State="*bf16", Out="*bf16"), dict(C=8192, WIDTH=4, BLOCK=128)),
        (_delta_kernel, dict(QKV="*bf16", A="*bf16", B="*bf16", ALog="*bf16", Bias="*bf16",
                             State="*fp32", Out="*bf16"), dict(KH=16, HV=32, K=128, V=128, BK=128, BV=16)),
        (_gate_kernel, dict(X="*bf16", Z="*bf16", W="*bf16", Out="*bf16"), dict(V=128, EPS=1e-6, BLOCK=128)),
        (_route_kernel, dict(Logits="*bf16", Ids="*i32", Weights="*fp32"), dict(E=256, TOPK=8, BLOCK=256)),
        (_silu_kernel, dict(Gate="*bf16", Up="*bf16", Out="*bf16"), dict(N=4096, BLOCK=256)),
        (_reduce_kernel, dict(Experts="*bf16", Shared="*bf16", Strength="*bf16", Out="*bf16"),
         dict(H=2048, TOPK=8, SLOTS=8, BLOCK=256)),
        (matvec._fp8_kernel, dict(X="*bf16", W="*fp8e4nv", WeightScale="*fp32", InputScale="*fp32", Y="*bf16"),
         dict(M=8192, K=2048, ROWS=4, WIDTH=2048)),
    ]
    for bank, routed, rows, width in ((False, False, 248320, 2048), (False, True, 2048, 512),
                                      (True, False, 512, 2048), (True, True, 2048, 512)):
        cases.append((matvec._nvfp4_kernel,
                      dict(X="*bf16", W="*u8", S="*fp8e4nv", GlobalScale="*fp32", Y="*bf16",
                           Ids="*i32", Routing="*fp32"),
                      dict(M=rows, K=width, GROUP=16, ROWS=4, PAIRS=width // 2, BANK=bank,
                           PER_SLOT_INPUT=bank and routed, ROUTED=routed, GLOBAL_BANK=bank)))
    for fn, signature, constants in cases:
        compiled = compile(ASTSource(fn, signature=signature, constexprs=constants),
                           target=GPUTarget("cuda", 120, 32),
                           options=dict(num_warps=4, enable_fp_fusion=False))
        print(f"COMPILED SM120: {compiled.name}", flush=True)
    print("Compilation only; kernels have not executed on a GPU.")


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
    parser.add_argument("--compile-only", action="store_true", help="compile SM120 kernels without GPU execution")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report is None:
        args.report = Path("benchmark/runs/kernel-benchmark.json" if args.benchmark else "benchmark/runs/kernel-check.json")
    if args.compile_only:
        compile_kernels()
        return 0
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
