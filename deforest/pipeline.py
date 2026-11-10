"""Run every local step after the Earth Engine CSVs are in data/raw/:

    python -m deforest.pipeline [--trials 30]

dataset → train → evaluate → predict → web data.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .config import ROOT

STEPS = ["deforest.dataset", "deforest.train", "deforest.evaluate", "deforest.predict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    common = ["--config", args.config] if args.config else []
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for step in STEPS:
        extra = ["--trials", str(args.trials)] if step == "deforest.train" and args.trials else []
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, "-m", step, *common, *extra], check=True, cwd=ROOT, env=env)
    print("\n=== web data ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_web_data.py"), *common], check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
