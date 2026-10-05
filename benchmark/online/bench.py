#!/usr/bin/env python3
"""Measure serial BS=1 requests and answers through a configured Qwen server.
Prompt IDs use the Qwen benchmark dataset; direct_run.py supplies resource limits."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from datetime import date
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--scenarios", nargs="+", default=["128", "2048", "8192"])
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--startup-timeout", type=int, default=2400)
    p.add_argument("--gpu-metrics", action="store_true", help="Sample NVML at 100 ms")
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    dataset = json.loads((ROOT / "benchmark/dataset.json").read_text())
    if str(Path(cfg["model"]).resolve()) != str(Path(dataset["model"]).resolve()):
        raise ValueError("Dataset token IDs belong to a different model")
    available = {(x["scenario"], x["run"]): x["prompt"] for x in dataset["examples"]}
    for scenario in a.scenarios:
        for run in range(a.runs + 1):
            if (scenario, run) not in available:
                raise ValueError(f"Missing dataset case: {(scenario, run)}")
    output = ROOT / "benchmark/runs" / date.today().isoformat() / cfg["label"] / str(int(time.time()))
    output.mkdir(parents=True, exist_ok=False)
    (ROOT / ".runtime").mkdir(exist_ok=True)
    lock = (ROOT / ".runtime/inference.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Resolving the venv symlink would select the base interpreter.
    python = str(ROOT / cfg["python"])
    module = ("sglang.launch_server" if cfg["engine"] == "sglang"
              else "vllm.entrypoints.openai.api_server")
    model_flag = "--model-path" if cfg["engine"] == "sglang" else "--model"
    command = [python, "-m", module, model_flag, cfg["model"],
               "--host", "127.0.0.1", "--port", str(a.port), *cfg["args"]]
    env = {**os.environ, **cfg.get("env", {})}
    if cfg["engine"] == "vllm":
        env["VLLM_SERVER_DEV_MODE"] = "1"
    for key in ("MAX_JOBS", "NVCC_THREADS", "FLASHINFER_NVCC_THREADS",
                "TORCHINDUCTOR_COMPILE_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        env[key] = "1"
    base = f"http://127.0.0.1:{a.port}"

    def api(route, body=None, *, method=None, timeout=600):
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.urlopen(urllib.request.Request(
            base + route, data=data, method=method,
            headers={"Content-Type": "application/json"}), timeout=timeout)

    result = {"config": cfg, "command": command,
              "compile_workers": 1, "max_cpu_execution": 2,
              "output_tokens": a.tokens, "runs": a.runs, "samples": [],
              "probes": [], "status": "starting"}
    result["dataset_sha256"] = hashlib.sha256(
        (ROOT / "benchmark/dataset.json").read_bytes()).hexdigest()
    result["packages"] = json.loads(subprocess.check_output([
        python, "-c", "import importlib.metadata as m, json; "
        "print(json.dumps({n:m.version(n) for n in "
        "('" + cfg["engine"] + "','torch','flashinfer-python','transformers')}))"
    ], text=True))
    result["gpu"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader"], text=True).strip()
    result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    result["measurement_notes"] = (
        "Serial HTTP requests, cold prefix cache, warmed kernels, temperature 0. "
        "Decode rate uses 511/(last SSE - first text SSE); speculative chunks may "
        "contain multiple tokens. Total latency is the primary comparison. "
        "Answer probes are smoke checks, not numerical engine parity.")

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)

    def save():
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"RESULT_DIR={output}", flush=True)
    print("Starting " + cfg["label"], flush=True)
    started = time.monotonic()
    save()
    log = (output / "server.log").open("w")
    server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while time.monotonic() - started < a.startup_timeout:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited with {server.returncode}; see {output / 'server.log'}")
            try:
                with api("/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Server startup timed out")
        result["startup_seconds"] = time.monotonic() - started
        print(f"Healthy after {result['startup_seconds']:.1f}s", flush=True)
        with api("/v1/models", timeout=10) as r:
            model = json.load(r)["data"][0]["id"]

        def reset_cache():
            route = "/flush_cache" if cfg["engine"] == "sglang" else "/reset_prefix_cache"
            with api(route, {}, method="POST", timeout=30) as r:
                raw = r.read()
                if raw:
                    # Successful SGLang flush responses are plain text.
                    if cfg["engine"] == "sglang" and raw.startswith(b"Cache flushed."):
                        return
                    reset = json.loads(raw)
                    if isinstance(reset, dict) and reset.get("success") is False:
                        raise RuntimeError("Cache reset did not succeed")

        def completion(ids, tokens, *, stream=False):
            body = {"model": model, "prompt": ids, "temperature": 0,
                    "max_tokens": tokens, "stream": stream, "seed": 0}
            if stream:
                body.update(ignore_eos=True, stream_options={"include_usage": True})
            monitor = None
            if stream and a.gpu_metrics:
                if __package__:
                    from .gpu_metrics import GpuMetrics
                else:
                    from gpu_metrics import GpuMetrics
                monitor = GpuMetrics()
                monitor.start()
            begin = time.monotonic()
            with api("/v1/completions", body) as r:
                if not stream:
                    return json.load(r)
                first = None
                usage = None
                text = []
                for line in r:
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        break
                    event = json.loads(raw)
                    if event.get("error"):
                        raise RuntimeError(event["error"])
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        if choice.get("text"):
                            if first is None:
                                first = time.monotonic()
                            text.append(choice["text"])
                end = time.monotonic()
            if first is None or usage is None:
                raise RuntimeError("Missing streaming text or token usage")
            if usage["completion_tokens"] != tokens:
                raise RuntimeError(f"Expected {tokens} generated tokens, got {usage}")
            row = {"ttft_ms": (first - begin) * 1000,
                    "total_ms": (end - begin) * 1000,
                    "decode_tokens_per_second": (tokens - 1) / (end - first),
                    "end_to_end_tokens_per_second": tokens / (end - begin),
                    "time_per_output_token_ms": 1000 * (end - first) / (tokens - 1),
                    "usage": usage, "text": "".join(text)}
            if monitor is not None:
                row["gpu_metrics"] = monitor.finish(begin, first, end)
                row["gpu_metrics"]["joules_per_output_token_estimate"] = (
                    row["gpu_metrics"]["energy_joules_estimate"] / tokens)
            return row

        for probe in dataset["probes"]:
            reset_cache()
            answer = completion(probe["prompt"], 32)
            actual = answer["choices"][0]["text"].strip()
            row = {"expected": probe["expected"], "actual": actual,
                   "passed": actual == probe["expected"]}
            result["probes"].append(row)
            print(f"Probe {probe['expected']!r}: {actual!r}", flush=True)
        if not all(x["passed"] for x in result["probes"]):
            raise RuntimeError("Correctness probe failed")
        for scenario in a.scenarios:
            reset_cache()
            completion(available[(scenario, 0)], 32)
        result["status"] = "benchmarking"
        result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        result["power_profile"] = subprocess.check_output(
            ["powerprofilesctl", "get"], text=True).strip()
        save()
        for scenario in a.scenarios:
            for run in range(1, a.runs + 1):
                reset_cache()
                row = completion(available[(scenario, run)], a.tokens, stream=True)
                row.update(scenario=scenario, run=run)
                result["samples"].append(row)
                save()
                print(f"{scenario} run {run}: TTFT {row['ttft_ms']:.1f}ms, "
                      f"total {row['total_ms']:.1f}ms, "
                      f"decode {row['decode_tokens_per_second']:.1f} tok/s", flush=True)
        result["summary"] = {}
        for scenario in a.scenarios:
            rows = [x for x in result["samples"] if x["scenario"] == scenario]
            result["summary"][scenario] = {key: statistics.median(x[key] for x in rows)
                for key in ("ttft_ms", "total_ms", "decode_tokens_per_second",
                            "end_to_end_tokens_per_second", "time_per_output_token_ms")}
            if a.gpu_metrics:
                result["summary"][scenario]["gpu_metrics"] = {
                    key: statistics.median(x["gpu_metrics"]["request"][key] for x in rows)
                    for key in rows[0]["gpu_metrics"]["request"]}
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        log.close()
        cgroup = next(l.split(":", 2)[2] for l in Path("/proc/self/cgroup").read_text().splitlines()
                      if l.startswith("0:"))
        resource = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        result["resources"] = {key: (resource / key).read_text().strip()
                               for key in ("memory.peak", "memory.max", "memory.swap.max", "memory.events")}
        save()
        print(json.dumps({"status": result["status"], "summary": result.get("summary"),
                          "resources": result["resources"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
