"""Print inference times and save a Torch trace and operator timing table."""

import sys

from bench import main


if __name__ == "__main__":
    raise SystemExit(main(["--backend", "triton", "--output-len", "16",
                          *sys.argv[1:], "--profile", "torch"]))
