#!/usr/bin/env python3
"""Run Qwen engine candidates sequentially under the resource guardian."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = [
    "qwen5090-vllm-mtp5",
    "qwen5090-sglang",
    "qwen5090-vllm-mtp4",
    "qwen5090-sglang-mtp3",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=CONFIGS, default=CONFIGS)
    parser.add_argument("--gpu-metrics", action="store_true")
    args = parser.parse_args()
    runs = ROOT / "benchmark/runs" / date.today().isoformat()
    runs.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started": time.time(), "engines": {}}
    previous = runs / "suite.json"
    if previous.exists():
        prior = json.loads(previous.read_text())
        if prior.get("status") == "running":
            raise RuntimeError("Another suite is running")
        state["engines"] = prior.get("engines", {})

    def save():
        temporary = runs / "suite.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(runs / "suite.json")

    for label in args.profiles:
        state["current"] = label
        save()
        before = set((runs / label).glob("*/results.json"))
        code = subprocess.call([sys.executable, str(ROOT / "benchmark/online/bench.py"),
                                str(ROOT / f"benchmark/configs/{label}.json"),
                                *(["--gpu-metrics"] if args.gpu_metrics else [])])
        created = set((runs / label).glob("*/results.json")) - before
        path = max(created, key=lambda p: p.stat().st_mtime) if created else None
        result = json.loads(path.read_text()) if path else {}
        state["engines"][label] = {
            "exit_code": code, "result": str(path) if path else None,
            "status": result.get("status", "failed"),
            "summary": result.get("summary", {}), "error": result.get("error"),
        }
        save()
    passed = {k: v for k, v in state["engines"].items() if v["status"] == "passed"}
    state["best_by_context"] = {
        context: min(passed, key=lambda k: passed[k]["summary"][context]["total_ms"])
        for context in ("128", "2048", "8192")
    } if passed else {}
    state["status"] = "complete" if all(
        state["engines"][label]["status"] == "passed" for label in args.profiles
    ) else "completed_with_failures"
    state["finished"] = time.time()
    save()
    print(json.dumps(state, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
