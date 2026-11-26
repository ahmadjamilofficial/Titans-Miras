from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output-dir", default="reports")
    args = p.parse_args()
    runs, out = Path(args.runs_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Line plot: training and validation loss.
    fig, ax = plt.subplots()
    for name in ["memory", "transformer"]:
        path = runs / name / "history.json"
        if not path.exists(): continue
        rows = json.loads(path.read_text())
        steps = [r["step"] for r in rows]
        ax.plot(steps, [r["cross_entropy"] for r in rows], label=f"{name} train")
        val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
        if val: ax.plot(*zip(*val), marker=".", linestyle="--", label=f"{name} validation")
    ax.set(title="Training dynamics", xlabel="Optimization step", ylabel="Cross-entropy")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "loss_curves.png", dpi=160); plt.close(fig)

    # Histogram: distribution of sequence lengths in the benchmark protocol.
    lengths = np.random.default_rng(7).choice([16, 32, 64, 128, 256], 1000,
                                                p=[.1, .2, .35, .25, .1])
    fig, ax = plt.subplots()
    ax.hist(lengths, bins=[0, 24, 48, 96, 192, 288], edgecolor="black")
    ax.set(title="Evaluation sequence-length distribution", xlabel="Sequence length", ylabel="Count")
    fig.tight_layout(); fig.savefig(out / "sequence_length_histogram.png", dpi=160); plt.close(fig)

    # Histogram: synthetic token frequencies.
    tokens = np.random.default_rng(9).choice(32, 10000)
    fig, ax = plt.subplots()
    ax.hist(tokens, bins=np.arange(34)-.5, edgecolor="black")
    ax.set(title="Illustrative token-frequency histogram", xlabel="Token ID", ylabel="Frequency")
    fig.tight_layout(); fig.savefig(out / "token_histogram.png", dpi=160); plt.close(fig)

    # Polygon/radar chart from evaluation files, if both model results exist.
    evals = {}
    for name in ["memory", "transformer"]:
        path = runs / name / "evaluation.json"
        if path.exists(): evals[name] = json.loads(path.read_text())
    if evals:
        lengths_eval = [r["sequence_length"] for r in next(iter(evals.values()))]
        angles = np.linspace(0, 2*np.pi, len(lengths_eval), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(); ax = fig.add_subplot(111, polar=True)
        for name, rows in evals.items():
            values = [r["cross_entropy"] for r in rows]
            values += values[:1]
            ax.plot(angles, values, marker="o", label=name)
            ax.fill(angles, values, alpha=.08)
        ax.set_xticks(angles[:-1], [str(x) for x in lengths_eval])
        ax.set_title("Sequence-length generalization (lower is better)")
        ax.set_xlabel("Sequence length"); ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
        fig.tight_layout(); fig.savefig(out / "sequence_length_polygon.png", dpi=160); plt.close(fig)

    print(f"Plots saved in {out}")


if __name__ == "__main__":
    main()
