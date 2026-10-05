#!/usr/bin/env python3
"""Create an isolated SGLang environment, reusing compatible installed wheels."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / ".venv-vllm"
target = ROOT / ".venv-sglang"
if not target.exists():
    subprocess.run(["uv", "venv", "--python", str(source / "bin/python"), str(target)], check=True)
    subprocess.run(["cp", "-a", "--reflink=auto",
                    str(source / "lib/python3.13/site-packages") + "/.",
                    str(target / "lib/python3.13/site-packages")], check=True)
subprocess.run(["uv", "pip", "install", "--python", str(target / "bin/python"),
                "--prerelease=allow", "sglang==0.5.19",
                "cuda-tile @ https://pypi.nvidia.com/cuda-tile/cuda_tile-1.6.0rc5-cp313-cp313-manylinux2014_x86_64.whl"], check=True)
subprocess.run([str(target / "bin/python"), "-c",
                "import importlib.metadata as m; print({p:m.version(p) for p in ('sglang','torch','sglang-kernel','flashinfer-python')})"], check=True)
