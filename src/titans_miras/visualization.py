from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_training(history: list[dict], output_dir: str | Path):
    """Create all training-time diagnostics from measured run history."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = [r["step"] for r in history]

    fig, ax = plt.subplots()
    ax.plot(steps, [r["train_loss"] for r in history], label="training CE")
    val = [(r["step"], r["val_loss"]) for r in history if "val_loss" in r]
    if val:
        ax.plot(*zip(*val), marker="o", linestyle="--", label="validation CE")
    ax.set(title="Training and validation loss", xlabel="Training step", ylabel="Cross-entropy")
    ax.grid(alpha=.25); ax.legend()
    _save(fig, out / "loss_curves.png")

    for key, title, ylabel, filename in [
        ("mean_surprise", "Memory surprise over training", "Mean surprise", "surprise_curve.png"),
        ("mean_write_gate", "Memory write gate over training", "Mean write gate", "write_gate_curve.png"),
        ("memory_norm", "Final memory-state norm", "Mean L2 norm", "memory_norm_curve.png"),
        ("grad_norm", "Gradient norm over training", "Gradient norm", "gradient_norm_curve.png"),
    ]:
        pairs = [(r["step"], r[key]) for r in history if key in r]
        if pairs:
            fig, ax = plt.subplots()
            ax.plot(*zip(*pairs))
            ax.set(title=title, xlabel="Training step", ylabel=ylabel)
            ax.grid(alpha=.25)
            _save(fig, out / filename)

    # Histogram of observed training losses, not a synthetic placeholder.
    fig, ax = plt.subplots()
    ax.hist([r["train_loss"] for r in history], bins=30, edgecolor="black")
    ax.set(title="Distribution of training cross-entropy", xlabel="Cross-entropy", ylabel="Step count")
    _save(fig, out / "training_loss_histogram.png")

    (out / "plots_manifest.json").write_text(json.dumps({
        "generated_from": "history.json",
        "plots": [p.name for p in sorted(out.glob("*.png"))],
    }, indent=2))


def plot_evaluation(results: list[dict], output_dir: str | Path):
    """Create context-length plots from actual evaluation measurements."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    lengths = [r["sequence_length"] for r in results]

    for metric, ylabel, filename, title in [
        ("cross_entropy", "Cross-entropy (lower is better)", "sequence_length_loss.png",
         "Next-token loss versus sequence length"),
        ("perplexity", "Perplexity (lower is better)", "sequence_length_perplexity.png",
         "Perplexity versus sequence length"),
        ("accuracy", "Token accuracy (higher is better)", "sequence_length_accuracy.png",
         "Next-token accuracy versus sequence length"),
    ]:
        fig, ax = plt.subplots()
        ax.plot(lengths, [r[metric] for r in results], marker="o")
        ax.set_xscale("log", base=2)
        ax.set(title=title, xlabel="Sequence length (tokens)", ylabel=ylabel)
        ax.grid(alpha=.25, which="both")
        _save(fig, out / filename)

    # Polygon/radar plot; axes are sequence lengths and values are measured CE.
    if len(results) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(lengths), endpoint=False).tolist()
        angles += angles[:1]
        values = [r["cross_entropy"] for r in results]
        values += values[:1]
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, marker="o", label="model")
        ax.fill(angles, values, alpha=.12)
        ax.set_xticks(angles[:-1], [str(x) for x in lengths])
        ax.set_title("Sequence-length generalization polygon (lower is better)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
        _save(fig, out / "sequence_length_polygon.png")

    if all("mean_surprise" in r for r in results):
        fig, ax = plt.subplots()
        ax.plot(lengths, [r["mean_surprise"] for r in results], marker="o")
        ax.set_xscale("log", base=2)
        ax.set(title="Mean surprise by sequence length", xlabel="Sequence length", ylabel="Surprise")
        ax.grid(alpha=.25, which="both")
        _save(fig, out / "surprise_by_length.png")

    if all("mean_write_gate" in r for r in results):
        fig, ax = plt.subplots()
        ax.plot(lengths, [r["mean_write_gate"] for r in results], marker="o")
        ax.set_xscale("log", base=2)
        ax.set(title="Mean memory write gate by sequence length",
               xlabel="Sequence length", ylabel="Mean write gate")
        ax.grid(alpha=.25, which="both")
        _save(fig, out / "write_gate_by_length.png")


def plot_suite_comparison(runs_dir: str | Path, output_dir: str | Path):
    """Compare model evaluation files and generate cross-model charts."""
    runs, out = Path(runs_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for name in ("memory", "transformer"):
        path = runs / name / "evaluation.json"
        if path.exists():
            evaluations[name] = json.loads(path.read_text())
    if not evaluations:
        print("No evaluation.json files found; run scripts/evaluate.py first.")
        return

    fig, ax = plt.subplots()
    for name, rows in evaluations.items():
        ax.plot([r["sequence_length"] for r in rows],
                [r["cross_entropy"] for r in rows], marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set(title="Model comparison: sequence-length generalization",
           xlabel="Sequence length (tokens)", ylabel="Cross-entropy (lower is better)")
    ax.grid(alpha=.25, which="both"); ax.legend()
    _save(fig, out / "model_comparison_loss.png")

    fig, ax = plt.subplots()
    for name, rows in evaluations.items():
        ax.plot([r["sequence_length"] for r in rows],
                [r["perplexity"] for r in rows], marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set(title="Model comparison: perplexity",
           xlabel="Sequence length (tokens)", ylabel="Perplexity (lower is better)")
    ax.grid(alpha=.25, which="both"); ax.legend()
    _save(fig, out / "model_comparison_perplexity.png")

    # Grouped bar chart for token accuracy.
    names = list(evaluations)
    common = sorted(set.intersection(*[
        {r["sequence_length"] for r in evaluations[n]} for n in names
    ])) if names else []
    if common:
        x = np.arange(len(common))
        width = .8 / len(names)
        fig, ax = plt.subplots()
        for i, name in enumerate(names):
            lookup = {r["sequence_length"]: r["accuracy"] for r in evaluations[name]}
            ax.bar(x + i * width, [lookup[v] for v in common], width, label=name)
        ax.set_xticks(x + width * (len(names) - 1) / 2, [str(v) for v in common])
        ax.set(title="Model comparison: next-token accuracy",
               xlabel="Sequence length", ylabel="Accuracy")
        ax.legend()
        _save(fig, out / "model_comparison_accuracy.png")

    if len(common) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(common), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        for name in names:
            lookup = {r["sequence_length"]: r["cross_entropy"] for r in evaluations[name]}
            vals = [lookup[v] for v in common]
            vals += vals[:1]
            ax.plot(angles, vals, marker="o", label=name)
            ax.fill(angles, vals, alpha=.08)
        ax.set_xticks(angles[:-1], [str(v) for v in common])
        ax.set_title("Polygon comparison across context lengths")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
        _save(fig, out / "model_comparison_polygon.png")
