# Titans-Miras

**A reproducible PyTorch research lab for neural memory, test-time adaptation ideas, and sequence modelling.**

This repository is a hands-on implementation and experiment suite inspired by Titans and MIRAS. It trains small models, evaluates next-token distributions over different context lengths, and automatically generates diagnostic graphs.

> **Research honesty:** the current recurrent memory is a compact Titans-inspired teaching abstraction. It is not an exact implementation of Titans' gradient-based associative memory. The write gate and retention regularizer are MIRAS-inspired experimental controls, not a complete MIRAS implementation. The Transformer is a baseline. Synthetic results are not evidence of natural-language capability.

## 1. What the project does

- Trains a surprise-gated recurrent neural-memory language model in PyTorch.
- Trains a causal Transformer baseline with the same synthetic task.
- Learns next-token probability distributions from a latent regime-conditioned Markov sequence generator.
- Evaluates cross-entropy, perplexity, and token accuracy across multiple sequence lengths.
- Records memory surprise, write-gate values, state norms, and gradient norms.
- Automatically creates line charts, histograms, polygon/radar charts, and cross-model comparisons.
- Saves checkpoints, JSON metrics, and run metadata for reproducibility.

No pretrained model download is required for the default demonstration. The synthetic distribution task is deliberately small, deterministic by seed, and runs on CPU. A Hugging Face pretrained-model adapter is a future extension; the present experiments isolate memory behaviour without confounding it with a large pretrained model.

## 2. Installation

Python 3.10 or newer:

```bash
git clone https://github.com/ahmadjamilofficial/Titans-Miras.git
cd Titans-Miras

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 3. Fast demonstration

Train a small memory model (CPU-friendly smoke run):

```bash
python scripts/train.py --model memory --steps 100 --batch-size 8 --seq-len 32
```

Train the Transformer baseline:

```bash
python scripts/train.py --model transformer --steps 100 --batch-size 8 --seq-len 32
```

Each training run writes:
- `best.pt` and `last.pt` checkpoints;
- `history.json` with measured training/validation metrics;
- `run.json` with configuration, seed, parameter count, and best validation loss;
- training plots such as `loss_curves.png`, `surprise_curve.png`, and `write_gate_curve.png` (memory model).

## 4. Full experiment suite

Train both architectures, evaluate multiple sequence lengths, and build comparison graphs:

```bash
python scripts/run_suite.py --steps 1200 --batch-size 32 --seq-len 64
```

For a more credible small study, use multiple seeds:

```bash
python scripts/run_suite.py --steps 1200 --batch-size 32 --seq-len 64 --seeds 11 22 33
```

On a CPU, begin with 100–300 steps and batch size 8. On CUDA, increase steps and batch size as resources allow. Transformer attention cost rises with sequence length, so the longest evaluation lengths can be slower.

## 5. Individual training and evaluation

Memory model:

```bash
python scripts/train.py --model memory --steps 1200 --seq-len 64 --memory-size 128 --retention-weight 0.00001
python scripts/evaluate.py --checkpoint runs/memory/best.pt --lengths 16 32 64 128 256 512 1024
```

Transformer:

```bash
python scripts/train.py --model transformer --steps 1200 --seq-len 64 --layers 2 --heads 4 --max-seq-len 2048
python scripts/evaluate.py --checkpoint runs/transformer/best.pt --lengths 16 32 64 128 256 512 1024
```

Create suite-wide plots from existing runs:

```bash
python scripts/plot_results.py --runs-dir runs --output-dir reports
```

## 6. Graph catalogue

Graphs are generated from measured run histories and evaluation JSON files, except the sequence-length sampling histogram which visualizes the benchmark's configured sampling protocol.

| Graph | Output | What it shows |
|---|---|---|
| Training/validation line chart | `loss_curves.png` | Cross-entropy versus optimization step |
| Surprise line chart | `surprise_curve.png` | Mean prediction surprise during memory training |
| Write-gate line chart | `write_gate_curve.png` | How strongly the memory writes over training |
| Memory norm line chart | `memory_norm_curve.png` | Magnitude of final recurrent memory state |
| Gradient norm line chart | `gradient_norm_curve.png` | Gradient clipping/training stability diagnostic |
| Training-loss histogram | `training_loss_histogram.png` | Distribution of observed training losses |
| Sequence-length histogram | `sequence_length_histogram.png` | Benchmark sequence-length sampling protocol |
| Token-frequency histogram | `token_frequency_histogram.png` | Empirical frequencies in generated synthetic tokens |
| CE vs sequence length | `sequence_length_loss.png` | Context-length generalization |
| Perplexity vs sequence length | `sequence_length_perplexity.png` | Distribution prediction quality |
| Accuracy vs sequence length | `sequence_length_accuracy.png` | Next-token accuracy |
| Polygon/radar chart | `sequence_length_polygon.png` | One model's CE profile across context lengths |
| Surprise vs sequence length | `surprise_by_length.png` | Memory surprise at different evaluation lengths |
| Write gate vs sequence length | `write_gate_by_length.png` | Memory write activity at different lengths |
| Model comparison lines | `model_comparison_loss.png`, `model_comparison_perplexity.png` | Memory model versus Transformer |
| Model comparison bars | `model_comparison_accuracy.png` | Token accuracy by model and context length |
| Comparison polygon | `model_comparison_polygon.png` | Cross-model CE profiles |

Some plots are model-specific and appear only when their corresponding measurements exist. The plotting scripts never fabricate model scores.

## 7. Understanding the synthetic task

The generator samples a latent regime for each sequence. Each regime has a different token-transition probability matrix. The model receives tokens and must predict the next token. It does not receive the regime label.

This is a distribution-learning benchmark, not a language corpus. It lets us ask whether a model can infer statistical structure from a context and maintain useful state. It is especially useful for debugging memory mechanisms before moving to real text.

## 8. Model equations (prototype)

For token embedding (x_t) and memory state (m_{t-1}):

[
r_t = W_r m_{t-1}
]

[
s_t = \operatorname{RMS}(x_t-r_t)
]

[
c_t = \tanh(W_c[x_t;m_{t-1}]),\quad
g_t = \sigma(W_g[x_t;m_{t-1};s_t])
]

[
m_t=m_{t-1}+g_t\,\sigma(\alpha s_t)(c_t-m_{t-1})
]

The prediction head maps (x_t+r_t) to next-token logits. The optional retention term penalizes final-state energy. See [architecture notes](docs/ARCHITECTURE.md).

## 9. Experiment design

Recommended ablations:
1. Retention weight: 0, 1e-5, 1e-4, 1e-3.
2. Memory size: 64, 128, 256.
3. Training context: 32, 64, 128.
4. Training seeds: at least 3.
5. Compare memory and Transformer with identical data generation, training steps, and batch sizes.
6. Add a change-point generator to test adaptation when the latent regime changes mid-sequence.

Record the commit, hardware, seed, configuration, parameter count, runtime, and checkpoint. Report means and variation across seeds. Do not claim one architecture is better based on a single run.

## 10. Project structure

```text
Titans-Miras/
├── configs/                 # Experiment configurations
├── docs/                    # Architecture and research protocols
├── src/titans_miras/
│   ├── data.py              # Synthetic sequence distribution
│   ├── memory.py            # Surprise-gated recurrent memory
│   ├── models.py            # Memory LM and Transformer baseline
│   ├── objectives.py        # Prediction and retention losses
│   └── visualization.py     # All training/evaluation plots
├── scripts/
│   ├── train.py             # Single-model training + automatic plots
│   ├── evaluate.py          # Context-length evaluation + plots
│   ├── run_suite.py         # End-to-end experiments
│   └── plot_results.py     # Suite-wide graph generation
├── tests/
└── README.md
```

## 11. Roadmap

- [x] Synthetic next-token distribution task.
- [x] PyTorch recurrent-memory model and Transformer baseline.
- [x] Training/evaluation scripts and automatic plots.
- [x] Multi-length evaluation and run metadata.
- [ ] Paper-faithful Titans associative memory with gradient-based fast-weight updates.
- [ ] MIRAS memory architecture/objective/retention ablation matrix.
- [ ] Change-point and long-range associative-recall benchmarks.
- [ ] Optional pretrained Hugging Face model adapter and real-text evaluation.
- [ ] Multi-seed aggregate plots with confidence intervals and compute accounting.

## 12. Tests

```bash
pytest
```

## License and citation

MIT License. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff). Cite the original Titans and MIRAS papers for the research mechanisms; this repository is an independent educational implementation.

