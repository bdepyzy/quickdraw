#!/usr/bin/env python3
"""Run a foreground process with RAM/compiler limits, without creating services."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def select_cpus():
    """Prefer two distinct fast cores instead of two SMT siblings."""
    ranked = []
    for cpu in os.sched_getaffinity(0):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        try:
            core = ((base / "topology/physical_package_id").read_text().strip(),
                    (base / "topology/core_id").read_text().strip())
            frequency = int((base / "cpufreq/cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            core, frequency = ("unknown", str(cpu)), 0
        ranked.append((-frequency, cpu, core))
    selected, cores = [], set()
    for _, cpu, core in sorted(ranked):
        if core not in cores:
            selected.append(cpu)
            cores.add(core)
        if len(selected) == 2:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--cache-tag", default="vllm")
    parser.add_argument("--quiet", action="store_true", help="omit launcher status lines")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    cache = ROOT / ".runtime" / args.cache_tag
    env = dict(os.environ)
    for key in ("MAX_JOBS", "NVCC_THREADS", "FLASHINFER_NVCC_THREADS",
                "TORCHINDUCTOR_COMPILE_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL",
                "CARGO_BUILD_JOBS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    env.update(QUICKDRAW_DIRECT_RUN="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               HF_HOME=str(ROOT / ".runtime/huggingface"),
               HF_XET_HIGH_PERFORMANCE="0", HF_XET_NUM_CONCURRENT_RANGE_GETS="2",
               HF_HUB_ETAG_TIMEOUT="60", HF_HUB_DOWNLOAD_TIMEOUT="60",
               UV_CONCURRENT_DOWNLOADS="2", UV_CONCURRENT_INSTALLS="2",
               UV_CONCURRENT_BUILDS="1", XDG_CACHE_HOME=str(cache),
               TRITON_CACHE_DIR=str(cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
               TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
               VLLM_CACHE_ROOT=str(cache / "vllm"),
               FLASHINFER_WORKSPACE_BASE=str(cache), CUDA_CACHE_PATH=str(cache / "cuda"),
               UV_CACHE_DIR=str(ROOT / ".runtime/uv"), TMPDIR=str(ROOT / ".runtime/tmp"))
    for path in (cache, Path(env["TMPDIR"]), Path(env["HF_HOME"])):
        path.mkdir(parents=True, exist_ok=True)
    # Use the user's existing cgroup delegation.
    parent = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service")
    group = parent / f"quickdraw-direct-{os.getpid()}"
    group.mkdir()
    proc = None
    try:
        for key, value in (("memory.high", 10 * 2**30), ("memory.max", 12 * 2**30),
                           ("memory.swap.max", 0), ("memory.oom.group", 1), ("pids.max", 256)):
            (group / key).write_text(str(value))
        cpus = select_cpus()
        if args.performance:
            command = ["powerprofilesctl", "launch", "--profile", "performance",
                       "--reason", "Quickdraw benchmark", "--appid", "quickdraw", *command]
        # Apply limits to the child, not the caller.
        child = ("import os,sys; from pathlib import Path; "
                 "Path(sys.argv[1]).write_text(str(os.getpid())); "
                 "os.sched_setaffinity(0, {int(x) for x in sys.argv[2].split(',')}); "
                 "os.execvpe(sys.argv[3], sys.argv[3:], os.environ)")
        if not args.quiet:
            print(f"Direct run: CPUs {cpus}, host RAM cap 12 GiB, no swap, compiler workers 1", flush=True)
        proc = subprocess.Popen([sys.executable, "-c", child, str(group / "cgroup.procs"),
                                 ",".join(map(str, cpus)), *command], cwd=ROOT, env=env)
        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        return proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            # Include detached engine/compiler children in cleanup.
            (group / "cgroup.kill").write_text("1")
            proc.wait()
        if (group / "cgroup.events").exists():
            for _ in range(100):
                if "populated 0" in (group / "cgroup.events").read_text():
                    break
                time.sleep(0.05)
            else:
                (group / "cgroup.kill").write_text("1")
                time.sleep(0.1)
        if not args.quiet:
            print("Host RAM peak bytes: " + (group / "memory.peak").read_text().strip(), flush=True)
        group.rmdir()


if __name__ == "__main__":
    sys.exit(main())
