from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titans_miras.data import RegimeSequenceGenerator, split_inputs
from titans_miras.models import MemoryLanguageModel, TransformerLanguageModel
from titans_miras.objectives import sequence_loss
from titans_miras.visualization import plot_training


def parse_args():
    p = argparse.ArgumentParser(description="Train a Titans-Miras sequence model.")
    p.add_argument("--model", choices=["memory", "transformer"], default="memory")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--memory-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retention-weight", type=float, default=1e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--no-plots", action="store_true", help="Skip automatic training plots.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.seq_len < 2 or args.batch_size < 1:
        raise ValueError("steps, batch-size and seq-len must be positive (seq-len >= 2)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    run_dir = Path(args.run_dir or f"runs/{args.model}")
    run_dir.mkdir(parents=True, exist_ok=True)
    data = RegimeSequenceGenerator(args.vocab_size, seed=args.seed)
    if args.model == "memory":
        model = MemoryLanguageModel(args.vocab_size, args.d_model, args.memory_size)
    else:
        model = TransformerLanguageModel(
            args.vocab_size, args.d_model, args.layers, args.heads,
            args.max_seq_len
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    best = float("inf")
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model={args.model} | parameters={parameter_count:,} | device={device}")

    for step in range(1, args.steps + 1):
        model.train()
        batch = data.sample(args.batch_size, args.seq_len, str(device))
        x, y = split_inputs(batch)
        logits, diagnostics = model(x)
        loss, parts = sequence_loss(
            logits, y, args.retention_weight, diagnostics
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        row = {
            "step": step,
            "train_loss": float(parts["cross_entropy"]),
            "total_loss": float(loss.detach()),
            "grad_norm": float(grad_norm),
        }
        if diagnostics:
            row["mean_surprise"] = float(diagnostics["surprise"].mean().detach())
            row["mean_write_gate"] = float(diagnostics["write_gate"].mean().detach())
            row["memory_norm"] = float(diagnostics["final_memory"].norm(dim=-1).mean().detach())
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val = data.sample(args.batch_size, args.seq_len, str(device))
                vx, vy = split_inputs(val)
                vl, _ = model(vx)
                val_loss = torch.nn.functional.cross_entropy(
                    vl.reshape(-1, vl.size(-1)), vy.reshape(-1)
                )
            row["val_loss"] = float(val_loss)
            print(
                f"step {step:5d}/{args.steps} | "
                f"train CE {row['train_loss']:.4f} | val CE {row['val_loss']:.4f}"
            )
            if row["val_loss"] < best:
                best = row["val_loss"]
                torch.save(
                    {"model": model.state_dict(), "args": vars(args),
                     "step": step, "parameter_count": parameter_count},
                    run_dir / "best.pt",
                )
        history.append(row)

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save(
        {"model": model.state_dict(), "args": vars(args),
         "step": args.steps, "parameter_count": parameter_count},
        run_dir / "last.pt",
    )
    metadata = {
        "model": args.model, "seed": args.seed, "steps": args.steps,
        "train_sequence_length": args.seq_len, "batch_size": args.batch_size,
        "parameter_count": parameter_count, "best_validation_cross_entropy": best,
        "device": str(device),
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    if not args.no_plots:
        plot_training(history, run_dir)
    print(f"Finished. Best validation CE={best:.4f}. Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
