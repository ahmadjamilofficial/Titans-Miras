# Architecture notes

## Scope

This first release compares a small recurrent memory model with causal self-attention on a synthetic next-token distribution task. It is an experimental scaffold for learning and debugging ideas from Titans and MIRAS.

## Synthetic task

A hidden regime is sampled for each sequence. Every regime owns a different token-transition matrix. The model receives token IDs but not the regime label, and predicts the next-token distribution. The task is intentionally controlled so we can vary sequence lengths and inspect memory behaviour.

## Recurrent memory prototype

For token embedding x_t and memory state m_(t-1), the module computes a read vector r_t, surprise s_t, candidate memory c_t, and write gate g_t:

- r_t = W_r m_(t-1)
- s_t = RMS(x_t - r_t)
- c_t = tanh(W_c [x_t ; m_(t-1)])
- g_t = sigmoid(W_g [x_t ; m_(t-1) ; s_t])
- m_t = m_(t-1) + g_t * sigmoid(alpha s_t) * (c_t - m_(t-1))

The output representation is x_t + r_t. The prediction head maps this representation to vocabulary logits.

This state is differentiable through the sequence and trained with next-token cross-entropy. The optional retention regularizer penalizes final-state energy. This is a teaching abstraction, not Titans' exact gradient-based associative-memory module or full chunkwise update algorithm.

## MIRAS-inspired design dimensions

The broader MIRAS perspective motivates separate experimental choices:
1. **Memory structure:** vector state versus matrix/associative state.
2. **Attentional bias:** what information the memory reads and writes.
3. **Retention:** what information persists, decays, or is overwritten.
4. **Online optimizer/objective:** how memory updates are computed.

The current prototype explores a surprise-conditioned write gate and a simple retention penalty only. It must not be described as a complete MIRAS implementation.

## Transformer baseline

The baseline uses causal self-attention and learned positional embeddings. It is a useful reference but is not automatically compute- or parameter-matched to the recurrent memory model.

## Generated diagnostics

Training histories produce loss, surprise, write-gate, memory norm, gradient norm and loss-histogram plots. Evaluation histories produce context-length loss/perplexity/accuracy, polygon, memory diagnostics, and empirical-versus-predicted token marginal plots. Suite-level plots aggregate across seeds and display standard-deviation error bars when multiple runs exist.
