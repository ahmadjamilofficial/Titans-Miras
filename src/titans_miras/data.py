from __future__ import annotations

import torch


class RegimeSequenceGenerator:
    """Synthetic next-token task with a persistent latent transition regime."""

    def __init__(self, vocab_size: int = 32, regimes: int = 4, seed: int = 42):
        if vocab_size < regimes or regimes < 2:
            raise ValueError("vocab_size must be >= regimes >= 2")
        self.vocab_size = vocab_size
        self.regimes = regimes
        generator = torch.Generator().manual_seed(seed)
        self.transitions = torch.rand(regimes, vocab_size, vocab_size, generator=generator)
        self.transitions /= self.transitions.sum(dim=-1, keepdim=True)
        # Each regime has a distinctive token preference, making it inferable from context.
        for r in range(regimes):
            self.transitions[r, :, r::regimes] *= 2.5
        self.transitions /= self.transitions.sum(dim=-1, keepdim=True)
        self.generator = generator

    def sample(self, batch_size: int, seq_len: int, device: str = "cpu") -> torch.Tensor:
        regime = torch.randint(self.regimes, (batch_size,), generator=self.generator)
        current = torch.randint(self.vocab_size, (batch_size,), generator=self.generator)
        result = torch.empty(batch_size, seq_len + 1, dtype=torch.long)
        result[:, 0] = current
        for t in range(1, seq_len + 1):
            probs = self.transitions[regime, current]
            current = torch.multinomial(probs, 1, generator=self.generator).squeeze(-1)
            result[:, t] = current
        return result.to(device)


def split_inputs(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return batch[:, :-1], batch[:, 1:]
