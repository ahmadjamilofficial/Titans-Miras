# Architecture notes

## Scope

The first implementation isolates one idea: a sequence model can carry a learned state forward rather than recomputing all pairwise token interactions at every step.

## Memory model

For input embedding x_t and memory state m_(t-1), the module computes a read vector r_t, a discrepancy/surprise scalar s_t, a candidate memory c_t, and a write gate g_t:

- r_t = W_r m_(t-1)
- s_t = RMS(x_t - r_t)
- c_t = tanh(W_c [x_t ; m_(t-1)])
- g_t = sigmoid(W_g [x_t ; m_(t-1) ; s_t])
- m_t = m_(t-1) + g_t * sigmoid(alpha s_t) * (c_t - m_(t-1))

The output representation is x_t + r_t. The prediction head maps this representation to vocabulary logits.

This recurrent state is differentiable through the sequence and is trained with next-token cross entropy. The optional retention regularizer penalizes final-state energy. This is a deliberately small teaching model; it is not the Titans paper's exact neural memory module and does not implement its full associative-memory objective or chunkwise update procedure.

## MIRAS-inspired design dimensions

The prototype exposes separable questions for later experiments:
1. **Memory structure:** vector state versus matrix/associative state.
2. **Attentional bias:** which token features are read and written.
3. **Retention:** what information persists or decays.
4. **Online optimizer/objective:** how memory updates are computed.

The current code explores a surprise-conditioned write gate and a simple retention penalty only. Do not describe it as a complete MIRAS implementation.

## Baseline

The Transformer baseline uses a causal attention mask and learned positional embeddings. It serves as a familiar reference point, not a perfectly compute-matched benchmark.
