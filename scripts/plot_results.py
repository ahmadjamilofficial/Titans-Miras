from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titans_miras.data import RegimeSequenceGenerator
from titans_miras.visualization import plot_suite_comparison


def main():
    p = argparse.ArgumentParser(description="Generate suite-level comparison graphs.")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--sample-count", type=int, default=12000)
    p.add_argument("--seed", type=int, default=9)
    args = p.parse_args()
    runs, out = Path(args.runs_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Actual synthetic benchmark sample: sequence-length and token-frequency histograms.
    rng = __import__("numpy").random.default_rng(args.seed)
    sampled_lengths = rng.choice([16, 32, 64, 128, 256, 512, 1024],
                                 size=1000, p=[.08, .12, .2, .2, .18, .14, .08])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(sampled_lengths, bins=[0, 24, 48, 96, 192, 384, 768, 1280], edgecolor="black")
    ax.set(title="Sequence-length sampling distribution", xlabel="Sequence length (tokens)",
           ylabel="Sample count")
    fig.tight_layout()
    fig.savefig(out / "sequence_length_histogram.png", dpi=170)
    plt.close(fig)

    generator = RegimeSequenceGenerator(args.vocab_size, seed=args.seed)
    tokens = generator.sample(max(1, args.sample_count // 128), 128).numpy().ravel()
    fig, ax = plt.subplots()
    ax.hist(tokens, bins=range(args.vocab_size + 1), edgecolor="black")
    ax.set(title="Generated token-frequency distribution", xlabel="Token ID", ylabel="Frequency")
    fig.tight_layout()
    fig.savefig(out / "token_frequency_histogram.png", dpi=170)
    plt.close(fig)

    plot_suite_comparison(runs, out)
    print(f"All available suite plots saved to {out}")


if __name__ == "__main__":
    main()
