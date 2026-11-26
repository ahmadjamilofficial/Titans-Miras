import torch
from titans_miras.data import RegimeSequenceGenerator, split_inputs
from titans_miras.models import MemoryLanguageModel, TransformerLanguageModel
from titans_miras.visualization import plot_training, plot_evaluation


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


def test_visualizations_create_training_and_evaluation_plots(tmp_path):
    history = [
        {"step": 1, "train_loss": 2.0, "val_loss": 2.1, "grad_norm": 0.8,
         "mean_surprise": 0.4, "mean_write_gate": 0.5, "memory_norm": 1.2},
        {"step": 2, "train_loss": 1.8, "val_loss": 1.9, "grad_norm": 0.7,
         "mean_surprise": 0.3, "mean_write_gate": 0.6, "memory_norm": 1.3},
    ]
    plot_training(history, tmp_path)
    assert (tmp_path / "loss_curves.png").exists()
    eval_rows = [
        {"sequence_length": 16, "cross_entropy": 1.2, "perplexity": 3.32,
         "accuracy": 0.4, "mean_surprise": 0.2, "mean_write_gate": 0.5,
         "target_token_distribution": [0.5, 0.5],
         "predicted_token_distribution": [0.4, 0.6]},
        {"sequence_length": 32, "cross_entropy": 1.3, "perplexity": 3.67,
         "accuracy": 0.35, "mean_surprise": 0.25, "mean_write_gate": 0.55},
        {"sequence_length": 64, "cross_entropy": 1.4, "perplexity": 4.05,
         "accuracy": 0.3, "mean_surprise": 0.3, "mean_write_gate": 0.6},
    ]
    plot_evaluation(eval_rows, tmp_path)
    assert (tmp_path / "sequence_length_polygon.png").exists()
    assert (tmp_path / "distribution_comparison.png").exists()
