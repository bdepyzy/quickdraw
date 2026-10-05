#!/usr/bin/env python3
"""Summarize the latest completed Qwen engine runs and their measured winners."""

import argparse
from datetime import date
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    day = date.fromisoformat(args.date).isoformat()
    runs = ROOT / "benchmark/runs" / day
    rows = {}
    for config in sorted((ROOT / "benchmark/configs").glob("qwen5090-*.json")):
        label = config.stem
        paths = sorted((runs / label).glob("*/results.json"),
                       key=lambda p: int(p.parent.name))
        if not paths:
            continue
        result = json.loads(paths[-1].read_text())
        if result["status"] in ("starting", "benchmarking"):
            raise RuntimeError("Benchmark still running: " + label)
        result["result_file"] = str(paths[-1])
        rows[label] = result
    passed = {k: v for k, v in rows.items() if v["status"] == "passed"}
    if not passed:
        raise RuntimeError("No passing benchmark runs")
    conditions = {(tuple(v.get("cpu_affinity", [])), v.get("power_profile"),
                   v.get("output_tokens"), v.get("runs"), v.get("dataset_sha256"))
                  for v in passed.values()}
    if len(conditions) != 1 or next(iter(conditions))[1] != "performance":
        raise RuntimeError("Passing runs have inconsistent CPU/power/request settings")
    winners = {context: min(passed, key=lambda k: passed[k]["summary"][context]["total_ms"])
               for context in ("128", "2048", "8192")}
    summary = {"results": {k: {"status": v["status"], "summary": v.get("summary"),
                              "error": v.get("error"), "result_file": v["result_file"]}
                           for k, v in rows.items()},
               "best_by_context": winners,
               "serve_config": f"benchmark/configs/{winners['2048']}.json",
               "selection": "Lowest median total latency at 2048 input tokens"}
    (runs / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [f"# Qwen RTX 5090 runs — {day}", "",
            "BS=1, 512 output tokens, temperature 0, three serial samples per context. "
            "Kernels warmed and prefix caches reset before each sample. "
            "Times below are median complete HTTP response latency.", "",
            "| Profile | 128 input | 2,048 input | 8,192 input |",
            "|---|---:|---:|---:|"]
    for label, result in passed.items():
        times = [f"{result['summary'][c]['total_ms'] / 1000:.3f} s" for c in ("128", "2048", "8192")]
        text.append(f"| {label} | {' | '.join(times)} |")
    text += ["", "## Latency and throughput", "",
             "| Profile | Input tokens | Decode tok/s | TTFT ms | Total ms |",
             "|---|---:|---:|---:|---:|"]
    for label, result in passed.items():
        for context, metrics in result["summary"].items():
            text.append(f"| {label} | {context} | {metrics['decode_tokens_per_second']:.1f} | "
                        f"{metrics['ttft_ms']:.1f} | {metrics['total_ms']:.1f} |")
    telemetry = {label: result for label, result in passed.items()
                 if all("gpu_metrics" in m for m in result["summary"].values())}
    if telemetry:
        text += ["", "## GPU telemetry", "",
                 "Median of each request's sampled mean; VRAM is the median request peak. "
                 "Device-wide NVML samples are collected every 100 ms during generation.", "",
                 "| Profile | Input tokens | GPU busy % | Memory busy % | VRAM GiB | Power W |",
                 "|---|---:|---:|---:|---:|---:|"]
        for label, result in telemetry.items():
            for context, metrics in result["summary"].items():
                gpu = metrics["gpu_metrics"]
                text.append(f"| {label} | {context} | {gpu['gpu_busy_percent_mean']:.1f} | "
                            f"{gpu['memory_controller_busy_percent_mean']:.1f} | "
                            f"{gpu['vram_peak_bytes']/2**30:.2f} | {gpu['power_watts_mean']:.1f} |")
        text += ["", "GPU busy measures time with at least one kernel executing. Memory busy "
                 "measures time during which device memory is read or written. Neither tells "
                 "us how close we are to maximum compute throughput or bandwidth. NVML "
                 "reports rolling windows, so short prefill/decode boundaries are approximate. "
                 "VRAM includes engine allocations and preallocated KV/state pools; driver "
                 "reserved memory is recorded separately when NVML v2 is available. "
                 "Raw JSON also includes clocks, temperature, phase samples, end-to-end TPS, "
                 "average time per output token, and estimated energy per token.", "",
                 "**Measured MBU: unavailable.** Effective memory bandwidth divided by peak "
                 "bandwidth needs hardware traffic counters. This machine restricts GPU "
                 "profiling to administrators (`RmProfilingAdminOnly: 1`); profiler capability "
                 "nodes have no user access. No DRAM counter collection was attempted. A "
                 "model traffic estimate would need to account for MoE routing, speculative "
                 "verification and accepted tokens before it could be compared meaningfully. "
                 "The 5090 uses GDDR7; the project's older HBM traffic terminology is generic.", "",
                 "Metric definitions: [NVIDIA NVML utilization]"
                 "(https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceStructs.html); "
                 "[NVIDIA profiling permissions]"
                 "(https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)."]
    text += ["", "Measured winners: " + "; ".join(f"{c} input: **{k}**" for c, k in winners.items()) + ".", "",
             "The comparison JSON selects a serving configuration at 2,048 input tokens. "
             "This is a workload-specific comparison of these candidates, not a global optimum.", "",
             "Measured jobs have a 12 GiB host RAM cap and cannot use swap. Compilation pools "
             "use one worker, and the entire subprocess tree runs on at most two logical CPUs. "
             "Inference also inherits that CPU limit. CPU performance mode is held while "
             "each job runs and restored afterwards. FlashInfer's official 0.6.18 CUDA 13.0 "
             "kernel wheels are installed in both environments. `direct_run.py` starts a "
             "foreground process in a delegated kernel cgroup without creating a service. "
             "Resource counters in the raw results apply to that cgroup; if multiple "
             "profiles share a wrapper, its peak is cumulative.", "",
             "Three answer probes passed for every included profile. These are smoke checks; "
             "they do not establish numerical parity, model quality, or equality across KV precisions. "
             "The FlashInfer MTP4 candidate requests FP8 KV. SGLang's automatic KV dtype "
             "resolves to FP8 for this checkpoint; vLLM MTP5 uses its automatic default. "
             "Decode rates in the JSON are estimates from streaming chunks, which can include "
             "multiple speculative tokens. Total latency is the selection metric.", "",
             "## Candidate outcomes", ""]
    for label, result in rows.items():
        text.append(f"- {label}: {result['status']}. " +
                    (result.get("error", "") or "") +
                    f" [Raw results](../{Path(result['result_file']).relative_to(ROOT)}).")
    text += ["", "## Model files", "",
             "Qwen weights were already present on the internal disk at "
             "`/home/bdepyzy/data/models/vllm/Qwen3.6-35B-A3B-NVFP4`. "
             "The dock drive contained a different dense Qwen model; no replacement was copied.", "",
             "Engine settings were informed by the [vLLM Qwen recipe]"
             "(https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B) and "
             "[SGLang Qwen cookbook]"
             "(https://github.com/sgl-project/sglang/blob/main/docs_new/cookbook/autoregressive/Qwen/Qwen3.6.mdx).", ""]
    (ROOT / f"analysis/RUNS-{day}.md").write_text("\n".join(text))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
