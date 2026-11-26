import torch
from titans_miras.data import RegimeSequenceGenerator, split_inputs
from titans_miras.models import MemoryLanguageModel, TransformerLanguageModel


def test_memory_model_shapes_and_gradients():
    model = MemoryLanguageModel(vocab_size=16, d_model=32, memory_size=40)
    batch = RegimeSequenceGenerator(16).sample(3, 12)
    x, y = split_inputs(batch)
    logits, diag = model(x)
    assert logits.shape == (3, 12, 16)
    assert diag["write_gate"].shape == (3, 12)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
    assert model.memory.write_gate.weight.grad is not None


def test_transformer_causal_model_shapes():
    model = TransformerLanguageModel(vocab_size=16, d_model=32, num_layers=1, num_heads=4)
    logits, _ = model(torch.randint(0, 16, (2, 10)))
    assert logits.shape == (2, 10, 16)
