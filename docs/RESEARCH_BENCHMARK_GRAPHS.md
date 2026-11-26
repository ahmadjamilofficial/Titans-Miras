# Published research graphs: Titans and long-context memory

This page collects figures published by Google Research for the Titans + MIRAS work. They are **external research results**, not measurements from this repository. They are included as reference benchmarks to guide reproduction.

Source article: [Titans + MIRAS: Helping AI have long-term memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) (Google Research, 4 December 2025).

## 1. Perplexity as sequence length grows

![Effect of memory depth on perplexity at 360M and 760M parameter scales](https://storage.googleapis.com/gweb-research2023-media/images/Titans-3-Performance1.width-1250.png)

The figure compares Mamba with long-term memory (LMM) and neural memory (MM) variants at two parameter scales. Lower perplexity is better. It motivates testing whether deeper memory improves language modelling as context length increases.

**Reproduction target for this repository:** train matched models, evaluate identical held-out sequences at increasing context lengths, and plot cross-entropy/perplexity with multiple-seed uncertainty. The current synthetic benchmark is not directly comparable to the paper's language-modelling experiment.

## 2. Extreme long-context recall

![BABILong accuracy versus sequence length for Titans and baselines](https://storage.googleapis.com/gweb-research2023-media/images/Titans-4-Performance2.width-1250.png)

This Google Research figure reports accuracy versus sequence length on an extreme long-context reasoning benchmark. It compares Titans (MAC)-FT with GPT-4, Mamba-FT, RMT-FT, Qwen2.5-72B, and other baselines. It illustrates the long-context recall question: can a model retrieve relevant facts as distractor context grows?

**Reproduction target for this repository:** implement a controlled needle-in-a-haystack / associative-recall task with exact-match accuracy, distractor count, retrieval position, and sequence length recorded for each run. Do not compare our current synthetic next-token accuracy to these published BABILong scores.

## 3. How to interpret short-term and long-term memory

Google's description distinguishes attention's precise in-context access (short-term working context) from Titans' learned neural memory that accumulates information over a stream (long-term memory). This is a useful conceptual distinction, not a claim that every Transformer has a separate human-like short-term memory module.

For a fair local comparison, this repository should report at least:

- **Short context:** next-token loss/accuracy at 128–512 tokens.
- **Long context:** loss/accuracy and retrieval at 1K–32K+ tokens, where supported.
- **Recall:** exact-match retrieval under increasing distractors.
- **Retention:** whether early facts remain retrievable after later updates.
- **Efficiency:** tokens/second, latency, peak memory, and parameter count.

## Attribution

Figures are reproduced by embedding the original Google Research-hosted image assets. All original figure content and research claims remain the property of their respective authors/publishers. See the [Google Research article](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) and the [Titans paper](https://arxiv.org/abs/2501.00663) for full methodology and citations.
