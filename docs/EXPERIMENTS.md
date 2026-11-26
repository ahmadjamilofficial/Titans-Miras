# Experiment protocol

## Baselines and fairness

Train the memory model and Transformer baseline with the same vocabulary, training sequence length, batch size, number of optimization steps, optimizer, and random seed. Repeat with at least three seeds for claims beyond a demonstration.

## Sequence-length sweep

Train at 64 tokens, then evaluate at 16, 32, 64, 128, and 256 tokens. The Transformer positional embedding is configured for up to 512 tokens. The memory model carries state sequentially.

## Ablations

1. Set `--retention-weight 0` versus a small positive value.
2. Replace surprise-conditioned writes with a constant write strength.
3. Vary memory size (64, 128, 256).
4. Compare training only at short lengths with mixed-length training.
5. Record both predictive quality and runtime.

## Interpret plots carefully

The included sequence-length histogram is a visualization of the evaluation protocol, not measured model performance. The token histogram in the plotting script is illustrative. For scientific reporting, replace it with counts from actual generated or held-out tokens. Polygon charts compare cross-entropy across sequence lengths; they should be accompanied by the underlying numeric table.

## Hypotheses to test

- H1: recurrent memory may degrade less sharply than a fixed-window attention baseline as test sequences grow.
- H2: surprise-gated writes may help when the latent regime changes.
- H3: a retention penalty may trade short-term adaptation against stable memory.

These are hypotheses, not expected outcomes. This first generator uses a fixed regime per sequence, so a later change-point benchmark is needed to test H2 properly.
