from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command):
    print("\n>>>", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    p = argparse.ArgumentParser(description="Train, evaluate, and plot both baseline models.")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", default=None)
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--reports-dir", default="reports")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    for seed in args.seeds:
        for model in ("memory", "transformer"):
            run_dir = Path(args.runs_dir) / f"{model}_seed{seed}"
            train = [py, str(root / "scripts/train.py"), "--model", model,
                     "--steps", str(args.steps), "--batch-size", str(args.batch_size),
                     "--seq-len", str(args.seq_len), "--seed", str(seed),
                     "--run-dir", str(run_dir)]
            if args.device:
                train += ["--device", args.device]
            run(train)
            evaluate = [py, str(root / "scripts/evaluate.py"),
                        "--checkpoint", str(run_dir / "best.pt"),
                        "--lengths", *map(str, args.lengths)]
            if args.device:
                evaluate += ["--device", args.device]
            run(evaluate)

    run([py, str(root / "scripts/plot_results.py"),
         "--runs-dir", args.runs_dir, "--output-dir", args.reports_dir])
    print("\nExperiment suite complete. Inspect runs/ and reports/.")


if __name__ == "__main__":
    main()
