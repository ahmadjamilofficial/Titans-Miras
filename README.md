# Titans-Miras

**A small, reproducible PyTorch laboratory for neural memory and sequence modelling.**

This repository is an educational implementation inspired by the Titans and MIRAS research direction. It is designed to run on a CPU or a modest GPU and to produce interpretable demonstrations—not to claim a faithful reproduction of every detail in the papers.

## What is implemented

- A compact recurrent neural-memory language model with a surprise-gated write mechanism.
- A causal Transformer baseline.
- A synthetic sequence-distribution task with configurable sequence lengths.
- Training, evaluation, checkpointing, and plotting scripts.
- Line plots, sequence-length histograms, token-frequency histograms, and polygon/radar comparisons.

The neural-memory module is **Titans-inspired**, not an exact Titans reproduction. The configurable write gate and retention penalty are **MIRAS-inspired experimental abstractions**, not a complete implementation of the MIRAS framework.

## Quick start

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/train.py --model memory --steps 1200
python scripts/train.py --model transformer --steps 1200
python scripts/evaluate.py --checkpoint runs/memory/best.pt
python scripts/plot_results.py --runs-dir runs --output-dir reports
```

For a quick smoke test:

```bash
python scripts/train.py --model memory --steps 30 --batch-size 8 --seq-len 32
pytest
```

## Task

The generator samples sequences from a latent regime-dependent Markov process. Each regime has a different transition distribution. This makes next-token prediction measurable and lets us test whether a model can retain useful information over longer contexts. It is synthetic: results should not be interpreted as natural-language capability.

## Model sketch

At each token, the memory model reads from a recurrent state, estimates prediction surprise, and uses a learned gate to decide how much new information to write. The state is carried through the sequence. The output head predicts the next token.

```text
token -> embedding -> memory read -> surprise/gate -> memory write
                                      |                    |
                                      +---- hidden state --+ -> next-token logits
```

The baseline uses causal self-attention over the current context window.

## Experiments

Compare models at multiple training and evaluation sequence lengths. Report:
- validation cross-entropy and perplexity;
- loss by sequence length;
- token-frequency distributions;
- memory write-gate and surprise statistics;
- parameter count and approximate inference throughput (optional extension).

Use multiple random seeds before drawing conclusions. Keep model size, data volume, optimizer, and training budget comparable.

## Limitations and research hygiene

1. This is a teaching prototype, not a state-of-the-art language model.
2. The memory update is differentiable within the forward pass; it is not the full Titans gradient-based associative-memory algorithm.
3. MIRAS is treated as a design lens for memory update, retention, and objective choices.
4. Synthetic benchmark performance does not establish generalization to real language.
5. Cite the original Titans and MIRAS papers when using this repository in research.

## Roadmap

- [ ] Add a paper-faithful Titans associative memory with gradient-based updates.
- [ ] Add controlled MIRAS memory-objective ablations.
- [ ] Add a real text dataset adapter and tokenizer.
- [ ] Add long-context and retrieval benchmarks.
- [ ] Add multi-seed confidence intervals.

## License

MIT. See [LICENSE](LICENSE).
