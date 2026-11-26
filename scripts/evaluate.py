from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titans_miras.data import RegimeSequenceGenerator, split_inputs
from titans_miras.models import MemoryLanguageModel, TransformerLanguageModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["args"]; vocab = cfg["vocab_size"]; d = cfg["d_model"]
    model = (MemoryLanguageModel(vocab, d, cfg["memory_size"]) if cfg["model"] == "memory"
             else TransformerLanguageModel(vocab, d))
    model.load_state_dict(ckpt["model"]); model.to(args.device).eval()
    data = RegimeSequenceGenerator(vocab, seed=1234)
    results = []
    for length in args.lengths:
        losses = []
        for _ in range(args.batches):
            batch = data.sample(args.batch_size, length, args.device)
            x, y = split_inputs(batch)
            with torch.no_grad():
                logits, _ = model(x)
                losses.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
        mean = sum(losses) / len(losses)
        results.append({"sequence_length": length, "cross_entropy": mean, "perplexity": float(torch.exp(torch.tensor(mean)))})
        print(f"length={length:4d} | CE={mean:.4f} | PPL={results[-1]['perplexity']:.3f}")
    out = Path(args.checkpoint).parent / "evaluation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
