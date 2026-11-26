# Research roadmap

## Phase 1 — Working demonstration (current)

- Synthetic regime-conditioned next-token prediction.
- Recurrent surprise-gated memory.
- Causal Transformer baseline.
- Training curves and sequence-length evaluation.

## Phase 2 — Better experimental controls

- Use identical train/validation/test streams for both models.
- Add multiple seeds and confidence intervals.
- Match parameter counts and training FLOPs more closely.
- Measure tokens/second and peak memory.
- Add abrupt regime switches and distractor sequences.

## Phase 3 — Closer Titans mechanism

- Implement a separate associative-memory module with an explicit inner loss.
- Implement gradient-based fast-weight updates with controlled update frequency.
- Compare chunkwise and tokenwise updates.
- Validate against paper equations and published reference implementations.

## Phase 4 — MIRAS ablation matrix

- Vary memory architecture.
- Vary update objective and attentional bias.
- Compare retention/forgetting rules.
- Evaluate retrieval, recall, adaptation, and interference separately.

## Phase 5 — Language modelling

- Add a tokenizer and a small public text corpus.
- Compare held-out perplexity at matched compute.
- Keep synthetic-memory benchmarks: natural text alone makes it difficult to diagnose why a memory mechanism succeeds or fails.

## Scientific standard

Record configuration, seed, commit hash, hardware, runtime, and checkpoint for every run. Report negative results. Distinguish a mechanism that is implemented from a hypothesis that is merely proposed.
