#!/usr/bin/env python3
"""Serve a validated engine configuration under direct_run.py."""
import argparse
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("config", type=Path)
p.add_argument("--port", type=int, default=8000)
a = p.parse_args()
cfg = json.loads(a.config.read_text())
(ROOT / ".runtime").mkdir(exist_ok=True)
lock = (ROOT / ".runtime/inference.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.set_inheritable(lock.fileno(), True)
python = str(ROOT / cfg["python"])
module = "sglang.launch_server" if cfg["engine"] == "sglang" else "vllm.entrypoints.openai.api_server"
model_flag = "--model-path" if cfg["engine"] == "sglang" else "--model"
os.environ.update(cfg.get("env", {}))
for key in ("MAX_JOBS", "NVCC_THREADS", "FLASHINFER_NVCC_THREADS",
            "TORCHINDUCTOR_COMPILE_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL"):
    os.environ[key] = "1"
os.execv(python, [python, "-m", module, model_flag, cfg["model"],
                 "--host", "127.0.0.1", "--port", str(a.port), *cfg["args"]])
