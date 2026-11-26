from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titans_miras.data import RegimeSequenceGenerator, split_inputs
from titans_miras.models import MemoryLanguageModel, TransformerLanguageModel
from titans_miras.visualization import plot_evaluation


def main():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint over context lengths.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        raise ValueError("batches and batch-size must be positive")

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    cfg = ckpt["args"]
    vocab, d_model = cfg["vocab_size"], cfg["d_model"]
    if cfg["model"] == "memory":
        model = MemoryLanguageModel(vocab, d_model, cfg["memory_size"])
    else:
        model = TransformerLanguageModel(
            vocab, d_model, cfg.get("layers", 2), cfg.get("heads", 4),
            cfg.get("max_seq_len", 2048)
        )
    model.load_state_dict(ckpt["model"])
    model.to(args.device).eval()
    data = RegimeSequenceGenerator(vocab, seed=args.seed)
    results = []

    for length in args.lengths:
        if length < 1:
            raise ValueError("evaluation lengths must be positive")
        if cfg["model"] == "transformer" and length > cfg.get("max_seq_len", 2048):
            print(f"Skipping length={length}: exceeds Transformer max context.")
            continue
        batch_losses, batch_accuracies = [], []
        surprise_values, gate_values = [], []
        predicted_sum = torch.zeros(vocab, dtype=torch.float64)
        target_counts = torch.zeros(vocab, dtype=torch.float64)
        confidence_values = []
        for _ in range(args.batches):
            batch = data.sample(args.batch_size, length, args.device)
            x, y = split_inputs(batch)
            with torch.no_grad():
                logits, diag = model(x)
                probs = logits.softmax(dim=-1)
                batch_losses.append(F.cross_entropy(
                    logits.reshape(-1, vocab), y.reshape(-1)
                ).item())
                batch_accuracies.append((logits.argmax(-1) == y).float().mean().item())
                predicted_sum += probs.sum(dim=(0, 1)).cpu().double()
                target_counts += torch.bincount(y.reshape(-1).cpu(), minlength=vocab).double()
                confidence_values.append(probs.max(dim=-1).values.mean().item())
                if diag:
                    surprise_values.append(diag["surprise"].mean().item())
                    gate_values.append(diag["write_gate"].mean().item())
        ce = sum(batch_losses) / len(batch_losses)
        total_tokens = float(target_counts.sum())
        row = {
            "sequence_length": length,
            "cross_entropy": ce,
            "perplexity": float(torch.exp(torch.tensor(ce))),
            "accuracy": sum(batch_accuracies) / len(batch_accuracies),
            "cross_entropy_std": float(torch.tensor(batch_losses).std(unbiased=False)),
            "mean_confidence": sum(confidence_values) / len(confidence_values),
            "target_token_distribution": (target_counts / total_tokens).tolist(),
            "predicted_token_distribution": (predicted_sum / total_tokens).tolist(),
        }
        if surprise_values:
            row["mean_surprise"] = sum(surprise_values) / len(surprise_values)
            row["mean_write_gate"] = sum(gate_values) / len(gate_values)
        results.append(row)
        print(
            f"length={length:4d} | CE={ce:.4f} | "
            f"PPL={row['perplexity']:.3f} | accuracy={row['accuracy']:.3f}"
        )

    out = checkpoint_path.parent / "evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    plot_evaluation(results, checkpoint_path.parent)
    print(f"Evaluation metrics and plots saved beside checkpoint: {checkpoint_path.parent}")


if __name__ == "__main__":
    main()
