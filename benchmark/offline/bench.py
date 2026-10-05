"""Print warm batch-one inference timings for synthetic token workloads."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
from random import Random
import shlex
import shutil
from statistics import median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Timing:
    prefill_ms: float
    decode_step_ms: float | None
    total_ms: float


def run_once(engine, prompt, output_len, *, synchronize=None, clock=time.perf_counter):
    """Time first-token delivery and the remaining decode steps separately."""
    import torch

    assert output_len > 0
    assert prompt and len(prompt) + output_len - 1 <= engine.max_context
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    with torch.inference_mode():
        synchronize()
        started = clock()
        logits = engine.prefill(prompt)
        engine.pending_token = torch.argmax(logits)
        # Include CPU token delivery in the timings.
        engine.pending_token.item()
        synchronize()
        first = clock()
        for _ in range(output_len - 1):
            engine.decode_step().item()
        synchronize()
        finished = clock()
    return Timing(
        prefill_ms=(first - started) * 1000,
        decode_step_ms=(finished - first) * 1000 / (output_len - 1) if output_len > 1 else None,
        total_ms=(finished - started) * 1000,
    )


def profile_workload(engine, prompt, output_len, phase, *, native=False, synchronize=None):
    """Run the selected phase; decode-only callers must prepare pending_token first."""
    import torch

    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    mark = torch.cuda.nvtx.range if native else torch.profiler.record_function
    if phase != "decode":
        with mark("quickdraw.prefill"):
            engine.pending_token = engine.prefill(prompt).argmax()
            engine.pending_token.item()
            synchronize()
    steps = 1 if phase == "decode" else output_len - 1
    if phase != "prefill" and steps:
        with mark("quickdraw.decode"):
            for _ in range(steps):
                engine.decode_step().item()
            synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path.home() / "data/models/vllm/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--backend", choices=("reference", "triton"), default="reference")
    parser.add_argument("--input-lens", nargs="+", type=int, default=[128])
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", nargs="?", const="all", choices=("all", "torch", "nsys", "ncu"),
                        help="save captures; bare --profile attempts all three formats")
    parser.add_argument("--profile-phase", choices=("generation", "prefill", "decode"), default="generation",
                        help="capture full generation, prefill alone, or one decode step (default: generation)")
    parser.add_argument("--profile-shapes", action="store_true",
                        help="also record tensor shapes; can retain tensors and increase profiling memory")
    parser.add_argument("--profile-dir", type=Path, help="capture directory (default: timestamped benchmark/profiles session)")
    parser.add_argument("--ncu-section", default="LaunchStats",
                        help="Compute section (default: launch configuration statistics)")
    parser.add_argument("--ncu-kernel", default="regex:.*",
                        help="Compute kernel-name filter, e.g. regex:.*_nvfp4_kernel.*")
    parser.add_argument("--ncu-launch-count", type=int, default=10,
                        help="maximum matching kernel launches captured by Compute")
    parser.add_argument("--capture-only", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(*args.input_lens, args.output_len, args.runs, args.ncu_launch_count) < 1 or args.warmup < 0:
        parser.error("lengths, runs, and launch count must be positive; warmup must be nonnegative")
    if args.capture_only is not None and not 0 <= args.capture_only < len(args.input_lens):
        parser.error("capture workload index is out of range")
    if not (args.model / "config.json").is_file():
        parser.error(f"local model config does not exist: {args.model / 'config.json'}")

    if os.environ.get("QUICKDRAW_DIRECT_RUN") != "1":
        os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/direct_run.py"),
                                 "--quiet", "--cache-tag", "quickdraw", "--",
                                 sys.executable, str(Path(__file__).resolve()),
                                 *(sys.argv[1:] if argv is None else argv)])

    modes = ("torch", "nsys", "ncu") if args.profile == "all" else ((args.profile,) if args.profile else ())
    tools = {}
    for name in (mode for mode in modes if mode != "torch"):
        executable = shutil.which(name)
        if executable is None:
            matches = sorted((ROOT / ".runtime/profilers").glob(f"opt/nvidia/nsight-*/**/{name}"))
            executable = next((str(path) for path in sorted(matches, key=lambda p: len(p.parts))
                               if path.is_file() and os.access(path, os.X_OK)), None)
        if executable is None:
            parser.error(f"{name} is not installed; put it on PATH or use --profile torch")
        tools[name] = executable
    session = None
    if modes:
        session = (args.profile_dir or ROOT / "benchmark/profiles" /
                   f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-{os.getpid()}").resolve()
        session.mkdir(parents=True, exist_ok=True)
        (session / "session.txt").write_text(
            f"Command: {sys.executable} {Path(__file__).resolve()} {shlex.join(sys.argv[1:] if argv is None else argv)}\n"
            f"Model: {args.model.resolve()}\nBackend: {args.backend}\n"
            f"BS=1; median of {args.runs} runs; warmup={args.warmup}; seed={args.seed}\n"
            "Synthetic IDs; EOS ignored; loading, warmup, and profiling excluded from generation times.\n"
            f"Capture phase: {args.profile_phase}; tensor shapes: {args.profile_shapes}\n"
            "Generation captures include prefill and all requested output tokens; model loading is excluded.\n"
            f"Compute section: {args.ncu_section}; launch limit: {args.ncu_launch_count}\n")

    import torch
    from quickdraw import Engine

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this benchmark has no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.capture_only is not None:
        # Inherited Nsight preload libraries can hang CPU compiler helpers.
        preload = [path for path in os.environ.get("LD_PRELOAD", "").split(":")
                   if path and "nsight-" not in path]
        if preload:
            os.environ["LD_PRELOAD"] = ":".join(preload)
        else:
            os.environ.pop("LD_PRELOAD", None)
    capacity = max(args.input_lens) + max(args.output_len, 2) - 1
    started = time.perf_counter()
    engine = Engine.from_pretrained(str(args.model), device="cuda",
                                    max_context=capacity, backend=args.backend)
    torch.cuda.synchronize()
    load = time.perf_counter() - started
    engine.eos_token_ids.clear()
    rng = Random(args.seed)
    workloads = [[rng.randrange(engine.ckpt.config.vocab_size) for _ in range(length)]
                 for length in args.input_lens]
    if args.capture_only is not None:
        prompt = workloads[args.capture_only]
        run_once(engine, prompt, max(args.output_len, 2))
        with torch.inference_mode():
            if args.profile_phase == "decode":
                engine.pending_token = engine.prefill(prompt).argmax()
            torch.cuda.synchronize()
            with torch.autograd.profiler.emit_nvtx(record_shapes=args.profile_shapes):
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("quickdraw.capture")
                try:
                    profile_workload(engine, prompt, args.output_len, args.profile_phase, native=True)
                finally:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.cudart().cudaProfilerStop()
        return 0

    report = [f"Load: {load:.3f} s",
              f"\n{'Input':>8} {'Output':>8} {'Prefill ms':>14} {'Decode ms/token':>18} {'Total ms':>14}"]
    print("\n".join(report), flush=True)
    for length, prompt in zip(args.input_lens, workloads):
        for _ in range(args.warmup):
            run_once(engine, prompt, args.output_len)
        timings = [run_once(engine, prompt, args.output_len) for _ in range(args.runs)]
        prefill = median(t.prefill_ms for t in timings)
        total = median(t.total_ms for t in timings)
        decode = f"{median(t.decode_step_ms for t in timings):.3f}" if args.output_len > 1 else "n/a"
        row = f"{length:8d} {args.output_len:8d} {prefill:14.3f} {decode:>18} {total:14.3f}"
        report.append(row)
        print(row, flush=True)
    if session is not None:
        (session / "timings.txt").write_text("\n".join(report) + "\n")
    if "torch" in modes:
        for index, prompt in enumerate(workloads):
            prefix = session / f"{args.profile_phase}-{index}-input{len(prompt)}"
            # Kineto diagnostics bypass Python's stdout/stderr streams.
            sys.stdout.flush()
            sys.stderr.flush()
            saved = [os.dup(fd) for fd in (1, 2)]
            try:
                with prefix.with_suffix(".torch.log").open("w") as log:
                    for fd in (1, 2):
                        os.dup2(log.fileno(), fd)
                    from torch.profiler import ProfilerActivity, profile

                    with torch.inference_mode():
                        if args.profile_phase == "decode":
                            engine.pending_token = engine.prefill(prompt).argmax()
                        torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                     record_shapes=args.profile_shapes) as prof:
                            profile_workload(engine, prompt, args.output_len, args.profile_phase)
                    prof.export_chrome_trace(str(prefix.with_suffix(".trace.json")))
                    with prefix.with_suffix(".operators.txt").open("w") as table:
                        table.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
                    del prof
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                for fd, backup in zip((1, 2), saved):
                    os.dup2(backup, fd)
                    os.close(backup)
    failures = []
    if tools:
        # Native capture processes load their own weights.
        del engine
        torch.cuda.empty_cache()
        for name, executable in tools.items():
            for index, length in enumerate(args.input_lens):
                prefix = session / f"{args.profile_phase}-{index}-input{length}"
                child = [sys.executable, str(Path(__file__).resolve()), "--model", str(args.model),
                         "--backend", args.backend, "--input-lens", *map(str, args.input_lens),
                         "--output-len", str(args.output_len), "--seed", str(args.seed),
                         "--profile-phase", args.profile_phase, "--capture-only", str(index)]
                if args.profile_shapes:
                    child.append("--profile-shapes")
                if name == "nsys":
                    command = [executable, "profile", "--trace=cuda,nvtx", "--sample=none",
                               "--cpuctxsw=none", "--capture-range=cudaProfilerApi",
                               "--capture-range-end=stop", "--force-overwrite=true",
                               "--output", str(prefix), *child]
                else:
                    command = [executable, "--nvtx", "--nvtx-include", "quickdraw.capture/",
                               "--section", args.ncu_section, "--kernel-name", args.ncu_kernel,
                               "--launch-count", str(args.ncu_launch_count), "--force-overwrite",
                               "--export", str(prefix), *child]
                with (session / "commands.txt").open("a") as commands:
                    commands.write(shlex.join([sys.executable, str(ROOT / "scripts/direct_run.py"),
                                               "--quiet", "--cache-tag", "quickdraw", "--", *command]) + "\n")
                log_path = prefix.with_suffix(f".{name}.log")
                # R610 capability-device ACLs can override RmProfilingAdminOnly.
                with log_path.open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                artifact = prefix.with_suffix(".nsys-rep" if name == "nsys" else ".ncu-rep")
                if result.returncode or not artifact.is_file():
                    failures.append((name, prefix.with_suffix(f".{name}.log")))
                    break
    if session is not None:
        print(f"Profiles: {session}", flush=True)
    for name, log in failures:
        reason = ("driver denies GPU performance-counter access" if "ERR_NVGPUCTRPERM" in log.read_text()
                  else "capture failed")
        print(f"{name}: {reason}; see {log}", file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
