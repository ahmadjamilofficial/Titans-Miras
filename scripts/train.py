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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["memory", "transformer"], default="memory")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--memory-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retention-weight", type=float, default=1e-5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.run_dir or f"runs/{args.model}")
    run_dir.mkdir(parents=True, exist_ok=True)
    data = RegimeSequenceGenerator(args.vocab_size, seed=args.seed)
    model = (MemoryLanguageModel(args.vocab_size, args.d_model, args.memory_size)
             if args.model == "memory" else
             TransformerLanguageModel(args.vocab_size, args.d_model))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    history = []
    best = float("inf")

    for step in range(1, args.steps + 1):
        model.train()
        batch = data.sample(args.batch_size, args.seq_len, str(device))
        x, y = split_inputs(batch)
        logits, diagnostics = model(x)
        loss, parts = sequence_loss(logits, y, args.retention_weight, diagnostics)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "cross_entropy": float(parts["cross_entropy"])}
        if diagnostics:
            row["mean_surprise"] = float(diagnostics["surprise"].mean().detach())
            row["mean_write_gate"] = float(diagnostics["write_gate"].mean().detach())
        history.append(row)

        if step % 50 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                val = data.sample(args.batch_size, args.seq_len, str(device))
                vx, vy = split_inputs(val)
                vl, vd = model(vx)
                val_loss = torch.nn.functional.cross_entropy(vl.flatten(0, 1), vy.flatten())
            row["val_loss"] = float(val_loss)
            print(f"step {step:5d} | train {row['cross_entropy']:.4f} | val {row['val_loss']:.4f}")
            if row["val_loss"] < best:
                best = row["val_loss"]
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, run_dir / "best.pt")

    with (run_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    torch.save({"model": model.state_dict(), "args": vars(args), "step": args.steps}, run_dir / "last.pt")
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
