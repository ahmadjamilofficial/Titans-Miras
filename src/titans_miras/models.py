from __future__ import annotations

import torch
from torch import nn
from .memory import SurpriseGatedMemory


class MemoryLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 96, memory_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.memory = SurpriseGatedMemory(d_model, memory_size)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = self.embedding(tokens)
        hidden, diagnostics = self.memory(x)
        return self.head(self.norm(hidden)), diagnostics


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 32, d_model: int = 96, num_layers: int = 2,
                 num_heads: int = 4, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch, length = tokens.shape
        if length > self.position.num_embeddings:
            raise ValueError(f"sequence length {length} exceeds max {self.position.num_embeddings}")
        positions = torch.arange(length, device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)[None, :, :]
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        hidden = self.encoder(x, mask=mask)
        return self.head(self.norm(hidden)), {}
