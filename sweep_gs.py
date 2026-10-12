"""Convenience CFG sweep for the rolling planner.

Additional command-line arguments are forwarded to the general ablation tool.
"""
import sys

from experiments.run_ablation import main


if __name__ == "__main__":
    if "--cfgs" not in sys.argv:
        sys.argv.extend(["--cfgs", "1.0,1.5,2.0,2.5,3.0,4.0"])
    if "--samplers" not in sys.argv:
        sys.argv.extend(["--samplers", "ddim:50"])
    if "--candidates" not in sys.argv:
        sys.argv.extend(["--candidates", "5"])
    main()
