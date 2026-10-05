#!/usr/bin/env python3
"""Install matching FlashInfer kernel wheels without changing engine versions."""

from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WHEELS = [
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/"
    "flashinfer_jit_cache-0.6.18%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl",
    "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18/"
    "flashinfer_cubin-0.6.18-py3-none-any.whl",
]
HASHES = [
    "428a47a554ade93c30a818e142b781df58582bf056bab94611fb4c906cc366bf",
    "2dd65c0fcfc6bc44c67f148530de5372979c2e3d260e47935730f94156d4d873",
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    directory = ROOT / ".runtime/kernel-wheels"
    directory.mkdir(parents=True, exist_ok=True)
    wheels = []
    for url, expected in zip(WHEELS, HASHES):
        target = directory / url.rsplit("/", 1)[1].replace("%2B", "+")
        if not target.exists() or digest(target) != expected:
            print("Downloading " + target.name, flush=True)
            subprocess.run(["curl", "--fail", "--location", "--silent",
                            "--show-error", "--retry", "5", "--retry-all-errors",
                            "--connect-timeout", "30", "--continue-at", "-",
                            "--output", str(target), url], check=True)
        if digest(target) != expected:
            raise RuntimeError("Kernel wheel failed SHA256 validation: " + target.name)
        wheels.append(str(target))
    for venv in (".venv-vllm", ".venv-sglang"):
        subprocess.run(["uv", "pip", "install", "--python",
                        str(ROOT / venv / "bin/python"), "--no-deps", *wheels],
                       check=True)
    print("Matching FlashInfer kernel wheels installed in both engine environments.",
          flush=True)


if __name__ == "__main__":
    main()
