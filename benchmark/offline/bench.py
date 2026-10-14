"""Print warm batch-one inference timings for synthetic token workloads."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
from statistics import median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.offline.compare import (
    DEFAULT_WORKLOAD, QUICK_WORKLOAD, bandwidth_estimate, prepare_memory, saved_baselines, traffic_model, workload_manifest,
)


@dataclass
class Timing:
    prefill_ms: float
    decode_step_ms: float | None
    total_ms: float


def enable_cute_decode(engine):
    """Capture experimental CuTe MoE/head decode, retaining the prefill step."""
    from functools import partial
    from quickdraw.kernel.cute.matvec import nvfp4_gemv
    from quickdraw.kernel.cute.moe import moe_block

    assert engine.backend == "triton" and engine.graph is not None
    prefill_step = engine.step
    engine.step = partial(prefill_step.func, **(prefill_step.keywords | {
        "experts": moe_block, "project_fp4": nvfp4_gemv}))
    try:
        engine.capture_decode()
    finally:
        engine.step = prefill_step


def speedup_report(timings, folder, *, manifest=None):
    baselines = ({name: json.loads((folder / f"{name}.json").read_text())["workloads"]
                  for name in ("vllm", "sglang")} if folder else {})
    if manifest is None and folder:
        manifest = json.loads((folder / "manifest.json").read_text())
    lines = ([f"Saved baselines: {folder}"] if folder else
             ["Saved baselines: unavailable for these settings; speed ratios need matching measurements.",
              "Use --quick for the saved 128-input/16-output comparison."])
    lines += [f"{'Input':>7} {'Engine':<10} {'Prefill ms':>12} {'Decode ms/tok':>14} {'Decode tok/s':>14} {'Total ms':>12} {'SoL est':>9}"]
    for length, actual in timings.items():
        cases = {"quickdraw": dict(ttft_ms=actual.prefill_ms, decode_ms_per_token=actual.decode_step_ms,
                                   total_ms=actual.total_ms)}
        for name, workloads in baselines.items():
            cases[name] = next(case["median"] for case in workloads if case["input_len"] == length)
        for name, values in cases.items():
            decode_ms = values["decode_ms_per_token"]
            decode, tps, sol = "n/a", "n/a", "n/a"
            if decode_ms is not None:
                decode, tps = f"{decode_ms:.3f}", f"{1000 / decode_ms:.2f}"
                if manifest:
                    estimate = bandwidth_estimate(manifest["traffic"], length,
                                                  manifest["conditions"]["output_len"], decode_ms,
                                                  manifest["peak_bandwidth_gbs"])
                    sol = f"{estimate['bandwidth_sol_estimate']:.3f}"
            lines.append(f"{length:7d} {name:<10} {values['ttft_ms']:12.3f} {decode:>14} {tps:>14} {values['total_ms']:12.3f} {sol:>9}")
        for name, baseline in cases.items():
            if name == "quickdraw":
                continue
            prefill = baseline["ttft_ms"] / actual.prefill_ms
            total = baseline["total_ms"] / actual.total_ms
            decode = (f"{baseline['decode_ms_per_token'] / actual.decode_step_ms:.2f}x"
                      if actual.decode_step_ms is not None else "n/a")
            lines.append(f"  Quickdraw / {name}: prefill {prefill:.2f}x | decode {decode} | total {total:.2f}x")
            saved_decode = (f"{baseline['decode_ms_per_token'] - actual.decode_step_ms:+.3f} ms/token"
                            if actual.decode_step_ms is not None else "n/a")
            lines.append(f"  Time saved vs {name}: prefill {baseline['ttft_ms'] - actual.prefill_ms:+.3f} ms | "
                         f"decode {saved_decode} | total {baseline['total_ms'] - actual.total_ms:+.3f} ms")
    lines += ["Speed: 1.00x = equal; higher is faster. Time saved: positive = faster, negative = slower.",
              "SoL: 1.000 = modeled peak-bandwidth limit.",
              "SoL is an estimate, not measured GPU utilization. Baseline engines are not started."]
    return "\n".join(lines)


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


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        description="Compare Quickdraw with saved vLLM/SGLang timings and estimated bandwidth SoL.",
        epilog="Defaults: Triton, BS=1, 1024 input / 1024 output, 1 warmup, median of 5 runs. No baseline engines start.")
    parser.set_defaults(**(QUICK_WORKLOAD if "--quick" in argv else DEFAULT_WORKLOAD))
    parser.add_argument("--quick", action="store_true", help="use the saved 128-input/16-output workload")
    parser.add_argument("--profile", nargs="?", const="torch", choices=("torch", "nsys", "ncu", "all"),
                        help="also save separate prefill and decode profiles (default: torch)")
    parser.add_argument("--cute-decode", action="store_true",
                        help="experiment with CuTe MoE/head decode kernels (full-model parity remains unverified)")
    parser.add_argument("--help-all", action="help", help="show workload and profiler controls")
    parser.add_argument("--model", type=Path,
                        default=Path.home() / "data/models/vllm/Qwen3.6-35B-A3B-NVFP4")
    parser.add_argument("--save-dir", "--profile-dir", dest="save_dir", type=Path,
                        help="save destination (default: timestamped run; latest symlink updated)")
    advanced = parser.add_argument_group("Advanced")
    advanced.add_argument("--backend", choices=("reference", "triton"))
    advanced.add_argument("--eager", action="store_true", help="disable decode CUDA Graphs")
    advanced.add_argument("--sequential-prefill", action="store_true", help="use token-at-a-time prefill")
    advanced.add_argument("--prefill-chunk-size", type=int, default=128)
    advanced.add_argument("--input-lens", nargs="+", type=int)
    advanced.add_argument("--output-len", type=int)
    advanced.add_argument("--runs", type=int)
    advanced.add_argument("--warmup", type=int)
    advanced.add_argument("--seed", type=int)
    advanced.add_argument("--baselines", type=Path, help="saved comparison session (default: newest matching session)")
    advanced.add_argument("--peak-bandwidth-gbs", type=float, default=1792,
                          help="advertised peak in decimal GB/s (default: RTX 5090)")
    advanced.add_argument("--profile-phase", choices=("both", "generation", "prefill", "decode"), default="both",
                         help="capture separate prefill and decode profiles, or select one phase")
    advanced.add_argument("--profile-shapes", action="store_true",
                        help="also record tensor shapes; can retain tensors and increase profiling memory")
    advanced.add_argument("--ncu-section", default="SpeedOfLight",
                          help="Compute section (default: SpeedOfLight)")
    advanced.add_argument("--ncu-kernel", default="regex:.*",
                        help="Compute kernel-name filter, e.g. regex:.*_nvfp4_kernel.*")
    advanced.add_argument("--ncu-launch-count", type=int, default=10,
                        help="maximum matching kernel launches captured by Compute")
    parser.add_argument("--capture-only", type=int, help=argparse.SUPPRESS)
    if "--help-all" not in argv:
        for action in advanced._group_actions:
            action.help = argparse.SUPPRESS
    args = parser.parse_args(argv)
    if min(*args.input_lens, args.output_len, args.runs, args.ncu_launch_count, args.prefill_chunk_size) < 1 or args.warmup < 0:
        parser.error("lengths, runs, and launch count must be positive; warmup must be nonnegative")
    if args.capture_only is not None and not 0 <= args.capture_only < len(args.input_lens):
        parser.error("capture workload index is out of range")
    if args.capture_only is not None and args.profile_phase == "both":
        parser.error("native capture worker requires one profile phase")
    if args.peak_bandwidth_gbs <= 0:
        parser.error("peak bandwidth must be positive")
    if args.cute_decode and (args.backend != "triton" or args.eager):
        parser.error("--cute-decode requires --backend triton and CUDA Graphs")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"local model config does not exist: {args.model / 'config.json'}")

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
            raise SystemExit(f"{name} is not installed; put it on PATH or use --profile")
        tools[name] = executable

    import torch
    from quickdraw import Engine

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this benchmark has no CPU fallback")
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
    manifest = workload_manifest(args)
    baseline = (saved_baselines(manifest, args.baselines, required=args.baselines is not None)
                if args.capture_only is None else None)
    phases = ("prefill", "decode") if args.profile_phase == "both" else (args.profile_phase,)
    if args.capture_only is None:
        assert "5090" in manifest["gpu"]["name"] or args.peak_bandwidth_gbs != 1792, "set this GPU's peak bandwidth"
        manifest.update(traffic=traffic_model(args.model), peak_bandwidth_gbs=args.peak_bandwidth_gbs,
                        baseline_session=str(baseline) if baseline else None,
                        execution=dict(cuda_graph=args.backend == "triton" and not args.eager,
                                       sequence_prefill=not args.sequential_prefill,
                                       prefill_chunk_size=args.prefill_chunk_size,
                                       decode_kernels="cute_moe_and_head" if args.cute_decode else "default"),
                        profiles=dict(tools=list(modes), phases=list(phases), record_shapes=args.profile_shapes,
                                      ncu=dict(section=args.ncu_section, kernel=args.ncu_kernel,
                                               launch_count=args.ncu_launch_count)))
        stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-{os.getpid()}"
        session = (args.save_dir or ROOT / "benchmark/runs" / f"bench-{stamp}").resolve()
        session.mkdir(parents=True, exist_ok=True)
        (session / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (session / "session.txt").write_text(
            f"Command: {sys.executable} {Path(__file__).resolve()} {shlex.join(sys.argv[1:] if argv is None else argv)}\n"
            f"Model: {args.model.resolve()}\nBackend: {args.backend}\n"
            f"Decode kernels: {'experimental CuTe MoE/head' if args.cute_decode else 'default'}\n"
            f"BS=1; median of {args.runs} runs; warmup={args.warmup}; seed={args.seed}\n"
            "Synthetic IDs; EOS ignored; loading, warmup, and profiling excluded from timings.\n"
            f"Profiles: {', '.join(modes) or 'none'}; phases: {', '.join(phases)}\n")
    capacity = manifest["conditions"]["max_context"]
    started = time.perf_counter()
    engine = Engine.from_pretrained(str(args.model), device="cuda",
                                    max_context=capacity, backend=args.backend, cuda_graph=not args.eager,
                                    sequence_prefill=not args.sequential_prefill,
                                    prefill_chunk_size=args.prefill_chunk_size)
    if args.cute_decode:
        enable_cute_decode(engine)
    torch.cuda.synchronize()
    load = time.perf_counter() - started
    engine.eos_token_ids.clear()
    workloads = manifest["prompts"]
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

    report = [f"BS=1 | inputs={','.join(map(str, args.input_lens))} | {args.output_len} outputs | median of {args.runs} runs | warmup={args.warmup}",
              f"Load: {load:.3f} s (excluded from timings)\n"]
    if args.cute_decode:
        report.append("Experimental CuTe MoE/head decode; other kernels retain Triton; full-model numerical parity is unverified.\n")
    print("\n".join(report), flush=True)
    measured = {}
    samples = []
    for length, prompt in zip(args.input_lens, workloads):
        prepare_memory()
        for _ in range(args.warmup):
            run_once(engine, prompt, args.output_len)
        before = prepare_memory()
        timings = [run_once(engine, prompt, args.output_len) for _ in range(args.runs)]
        assert prepare_memory()["high_events"] == before["high_events"], "RAM soft-limit throttling invalidates these samples"
        prefill = median(t.prefill_ms for t in timings)
        total = median(t.total_ms for t in timings)
        decode_ms = median(t.decode_step_ms for t in timings) if args.output_len > 1 else None
        measured[length] = Timing(prefill, decode_ms, total)
        samples.append(dict(input_len=length, samples=[asdict(t) for t in timings],
                            median=asdict(measured[length])))
    comparison = speedup_report(measured, baseline, manifest=manifest)
    report.append(comparison)
    print(comparison, flush=True)
    (session / "timings.txt").write_text("\n".join(report) + "\n")
    results = dict(manifest=manifest, load_seconds=load, workloads=samples,
                   baselines={name: json.loads((baseline / f"{name}.json").read_text())
                              for name in ("vllm", "sglang")} if baseline else {})
    (session / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    jobs = [(phase, index, prompt) for phase in phases for index, prompt in enumerate(workloads)]
    if "torch" in modes:
        for phase, index, prompt in jobs:
            prefix = session / f"{phase}-{index}-input{len(prompt)}"
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
                        if phase == "decode":
                            engine.pending_token = engine.prefill(prompt).argmax()
                        torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                     record_shapes=args.profile_shapes) as prof:
                            profile_workload(engine, prompt, args.output_len, phase)
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
            for phase, index, prompt in jobs:
                prefix = session / f"{phase}-{index}-input{len(prompt)}"
                child = [sys.executable, str(Path(__file__).resolve()), "--model", str(args.model),
                         "--backend", args.backend, "--input-lens", *map(str, args.input_lens),
                         "--output-len", str(args.output_len), "--seed", str(args.seed),
                         "--runs", str(args.runs), "--warmup", str(args.warmup),
                         "--prefill-chunk-size", str(args.prefill_chunk_size),
                         "--profile-phase", phase, "--capture-only", str(index)]
                if args.eager:
                    child.append("--eager")
                if args.cute_decode:
                    child.append("--cute-decode")
                if args.sequential_prefill:
                    child.append("--sequential-prefill")
                if args.profile_shapes:
                    child.append("--profile-shapes")
                if name == "nsys":
                    command = [executable, "profile", "--trace=cuda,nvtx", "--sample=none",
                               "--cuda-graph-trace=node",
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
    results["profile_failures"] = [dict(tool=name, log=str(log)) for name, log in failures]
    (session / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    latest = ROOT / "benchmark/runs/latest"
    if not failures:
        latest.parent.mkdir(parents=True, exist_ok=True)
        temporary = latest.with_name(f".latest-{os.getpid()}")
        temporary.symlink_to(session, target_is_directory=True)
        temporary.replace(latest)
    print(f"Saved: {session}", flush=True)
    if not failures:
        print(f"Latest: {latest}", flush=True)
    for name, log in failures:
        reason = ("driver denies GPU performance-counter access" if "ERR_NVGPUCTRPERM" in log.read_text()
                  else "capture failed")
        print(f"{name}: {reason}; see {log}", file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
