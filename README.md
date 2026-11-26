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
- Automatically creates diagnostic line charts, histograms, polygon/radar charts, distribution comparisons, and cross-model comparisons.
- Saves checkpoints, JSON metrics, and run metadata for reproducibility.

The default demonstration does not require a pretrained model download. The synthetic distribution task isolates memory behaviour and runs on CPU. A pretrained Hugging Face adapter and real-text evaluation are planned extensions, not current features.

## 2. Installation

Python 3.10 or newer:

```bash
git clone https://github.com/ahmadjamilofficial/Titans-Miras.git
cd Titans-Miras
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 3. Fast demonstration

```bash
python scripts/train.py --model memory --steps 100 --batch-size 8 --seq-len 32
python scripts/train.py --model transformer --steps 100 --batch-size 8 --seq-len 32
```

Training automatically saves `best.pt`, `last.pt`, `history.json`, `run.json`, and training plots in `runs/<model>/`. Use `--no-plots` to skip plot generation.

## 4. Full experiment suite

Train both architectures, evaluate sequence lengths, and generate comparison charts:

```bash
python scripts/run_suite.py --steps 1200 --batch-size 32 --seq-len 64
```

For multi-seed comparison with mean and standard-deviation error bars:

```bash
python scripts/run_suite.py --steps 1200 --batch-size 32 --seq-len 64 --seeds 11 22 33
```

CPU users should start with 100–300 steps and batch size 8. Transformer attention cost grows with sequence length. CUDA is selected automatically when available.

## 5. Individual commands

```bash
# Memory model
python scripts/train.py --model memory --steps 1200 --seq-len 64 --memory-size 128
python scripts/evaluate.py --checkpoint runs/memory/best.pt --lengths 16 32 64 128 256 512 1024

# Transformer baseline
python scripts/train.py --model transformer --steps 1200 --seq-len 64 --layers 2 --heads 4 --max-seq-len 2048
python scripts/evaluate.py --checkpoint runs/transformer/best.pt --lengths 16 32 64 128 256 512 1024

# Rebuild suite-level plots from existing runs
python scripts/plot_results.py --runs-dir runs --output-dir reports

# Tests
pytest
```

## 6. Graph catalogue

Training plots are created automatically by `train.py`; evaluation plots by `evaluate.py`; cross-run comparisons by `plot_results.py`.

| Graph | File | Interpretation |
|---|---|---|
| Training and validation line chart | `loss_curves.png` | Cross-entropy over optimization steps |
| Surprise line chart | `surprise_curve.png` | Memory prediction surprise over training |
| Write-gate line chart | `write_gate_curve.png` | Average strength of memory writes |
| Memory norm line chart | `memory_norm_curve.png` | Magnitude of final recurrent state |
| Gradient norm line chart | `gradient_norm_curve.png` | Optimization stability |
| Training loss histogram | `training_loss_histogram.png` | Distribution of observed training losses |
| Sequence-length histogram | `sequence_length_histogram.png` | Configured evaluation-length sampling protocol |
| Token-frequency histogram | `token_frequency_histogram.png` | Empirical generated-token frequencies |
| CE vs context length | `sequence_length_loss.png` | Next-token distribution quality |
| Perplexity vs context length | `sequence_length_perplexity.png` | Exponentiated cross-entropy |
| Accuracy vs context length | `sequence_length_accuracy.png` | Argmax next-token accuracy |
| Polygon/radar chart | `sequence_length_polygon.png` | One model's loss profile over lengths |
| Predicted vs empirical distribution | `distribution_comparison.png` | Marginal token probability calibration |
| Surprise vs context length | `surprise_by_length.png` | Memory surprise by context length |
| Write gate vs context length | `write_gate_by_length.png` | Memory write activity by context length |
| Model comparison line charts | `model_comparison_loss.png`, `model_comparison_perplexity.png` | Memory versus Transformer |
| Model comparison accuracy | `model_comparison_accuracy.png` | Accuracy across model families |
| Multi-seed polygon | `model_comparison_polygon.png` | Mean CE comparison across context lengths |

Plots are based on actual measured histories/evaluations, except the sequence-length histogram, which depicts the benchmark sampling protocol. Multi-seed comparison plots include standard-deviation error bars when multiple seed runs are available. No model scores are fabricated.

## 7. Synthetic distribution-prediction task

Each sequence is sampled from a latent regime-conditioned Markov process. Each regime has a different token-transition probability matrix, and the regime label is hidden from the model. The model sees token IDs and predicts a probability distribution for the next token.

Evaluation reports:
- **Cross-entropy:** primary probability prediction metric.
- **Perplexity:** exponential of cross-entropy.
- **Accuracy:** fraction of next-token argmax predictions that are correct.
- **Mean confidence:** average maximum predicted token probability.
- **Marginal token distributions:** empirical target frequencies versus model-predicted probabilities.

This is a controlled synthetic benchmark, not a real-language corpus or pretrained-model benchmark.

## 8. Memory model (prototype equations)

For token embedding (x_t) and memory state (m_{t-1}):

[
r_t = W_r m_{t-1},\quad s_t = \operatorname{RMS}(x_t-r_t)
]
[
c_t = \tanh(W_c[x_t;m_{t-1}]),\quad
g_t = \sigma(W_g[x_t;m_{t-1};s_t])
]
[
m_t=m_{t-1}+g_t\,\sigma(\alpha s_t)(c_t-m_{t-1})
]

The prediction head maps (x_t+r_t) to next-token logits. The optional retention term penalizes final-state energy. See [architecture notes](docs/ARCHITECTURE.md).

## 9. Suggested experiments

1. Retention weight: 0, 1e-5, 1e-4, 1e-3.
2. Memory size: 64, 128, 256.
3. Training context: 32, 64, 128.
4. Seeds: at least 3.
5. Add change points and distractors to test adaptation.
6. Match parameter counts and compute budgets before making architecture claims.

Record commit, hardware, seed, config, parameter count, runtime, and checkpoint. Report variation across seeds; do not conclude superiority from a single run.

## 10. Project structure

```text
Titans-Miras/
├── configs/
├── docs/
├── src/titans_miras/
│   ├── data.py
│   ├── memory.py
│   ├── models.py
│   ├── objectives.py
│   └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_suite.py
│   └── plot_results.py
├── tests/
└── README.md
```

## 11. Roadmap

- [x] Synthetic next-token distribution task.
- [x] PyTorch recurrent-memory model and Transformer baseline.
- [x] Training/evaluation scripts and automatic plots.
- [x] Multi-length evaluation and run metadata.
- [x] Predicted-versus-empirical token distribution plots.
- [ ] Paper-faithful Titans associative memory with gradient-based fast-weight updates.
- [ ] MIRAS memory architecture/objective/retention ablation matrix.
- [ ] Change-point and long-range associative-recall benchmarks.
- [ ] Optional pretrained Hugging Face adapter and real-text evaluation.
- [ ] Confidence intervals and compute-matched multi-seed studies.

## 12. License and citation

MIT License. See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff). Cite the original Titans and MIRAS papers for their research mechanisms. This repository is an independent educational implementation.
