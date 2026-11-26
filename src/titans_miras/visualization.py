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
    """Create training diagnostics from measured run history."""
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

    fig, ax = plt.subplots()
    ax.hist([r["train_loss"] for r in history], bins=30, edgecolor="black")
    ax.set(title="Distribution of training cross-entropy", xlabel="Cross-entropy", ylabel="Step count")
    _save(fig, out / "training_loss_histogram.png")
    (out / "plots_manifest.json").write_text(json.dumps({
        "generated_from": "history.json",
        "plots": [p.name for p in sorted(out.glob("*.png"))],
    }, indent=2))


def plot_evaluation(results: list[dict], output_dir: str | Path):
    """Create context-length and distribution diagnostics from evaluation metrics."""
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

    if len(results) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(lengths), endpoint=False).tolist()
        angles += angles[:1]
        values = [r["cross_entropy"] for r in results] + [results[0]["cross_entropy"]]
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

    # Distribution chart uses marginal frequencies measured on held-out generated tokens.
    target = next((r.get("target_token_distribution") for r in results
                   if r.get("target_token_distribution") is not None), None)
    predicted = next((r.get("predicted_token_distribution") for r in results
                      if r.get("predicted_token_distribution") is not None), None)
    if target is not None and predicted is not None:
        x = np.arange(len(target))
        fig, ax = plt.subplots()
        ax.plot(x, target, marker="o", label="empirical target")
        ax.plot(x, predicted, marker="x", label="mean predicted probability")
        ax.set(title="Next-token marginal distribution",
               xlabel="Token ID", ylabel="Probability")
        ax.grid(alpha=.25); ax.legend()
        _save(fig, out / "distribution_comparison.png")


def plot_suite_comparison(runs_dir: str | Path, output_dir: str | Path):
    """Aggregate evaluations across all seeds and compare model families."""
    runs, out = Path(runs_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for name in ("memory", "transformer"):
        rows_by_seed = []
        for path in sorted(runs.glob(f"{name}*/evaluation.json")):
            rows_by_seed.append(json.loads(path.read_text()))
        if rows_by_seed:
            evaluations[name] = rows_by_seed
    if not evaluations:
        print("No evaluation.json files found; run scripts/evaluate.py first.")
        return

    aggregate = {}
    for name, seed_runs in evaluations.items():
        lengths = sorted(set.intersection(*[
            {r["sequence_length"] for r in run} for run in seed_runs
        ]))
        aggregate[name] = {}
        for length in lengths:
            rows = [next(r for r in run if r["sequence_length"] == length)
                    for run in seed_runs]
            aggregate[name][length] = {
                metric: (float(np.mean([r[metric] for r in rows])),
                         float(np.std([r[metric] for r in rows])))
                for metric in ("cross_entropy", "perplexity", "accuracy")
            }

    for metric, ylabel, filename, title in [
        ("cross_entropy", "Cross-entropy (lower is better)", "model_comparison_loss.png",
         "Model comparison: sequence-length generalization"),
        ("perplexity", "Perplexity (lower is better)", "model_comparison_perplexity.png",
         "Model comparison: perplexity"),
        ("accuracy", "Accuracy (higher is better)", "model_comparison_accuracy.png",
         "Model comparison: next-token accuracy"),
    ]:
        fig, ax = plt.subplots()
        for name, values in aggregate.items():
            lengths = sorted(values)
            means = [values[k][metric][0] for k in lengths]
            stds = [values[k][metric][1] for k in lengths]
            ax.errorbar(lengths, means, yerr=stds, marker="o", capsize=3, label=name)
        ax.set_xscale("log", base=2)
        ax.set(title=title, xlabel="Sequence length (tokens)", ylabel=ylabel)
        ax.grid(alpha=.25, which="both"); ax.legend()
        _save(fig, out / filename)

    common = sorted(set.intersection(*[
        set(values) for values in aggregate.values()
    ]))
    if len(common) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(common), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        for name, values in aggregate.items():
            vals = [values[k]["cross_entropy"][0] for k in common]
            vals += vals[:1]
            ax.plot(angles, vals, marker="o", label=name)
            ax.fill(angles, vals, alpha=.08)
        ax.set_xticks(angles[:-1], [str(v) for v in common])
        ax.set_title("Model comparison polygon (mean cross-entropy)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))
        _save(fig, out / "model_comparison_polygon.png")

    summary = {
        name: {
            str(length): {
                metric: {"mean": value[0], "std": value[1]}
                for metric, value in metrics.items()
            }
            for length, metrics in values.items()
        }
        for name, values in aggregate.items()
    }
    (out / "aggregate_metrics.json").write_text(json.dumps(summary, indent=2))
