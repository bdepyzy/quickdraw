"""Print inference times and measure selected launches during generation with Compute."""

import sys

from bench import main


if __name__ == "__main__":
    raise SystemExit(main(["--backend", "triton", "--output-len", "16",
                          "--ncu-section", "SpeedOfLight", "--ncu-kernel", "regex:.*_nvfp4_kernel.*",
                          "--ncu-launch-count", "1", *sys.argv[1:], "--profile", "ncu"]))
