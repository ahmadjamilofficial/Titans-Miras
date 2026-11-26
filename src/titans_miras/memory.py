from __future__ import annotations

import torch
from torch import nn


class SurpriseGatedMemory(nn.Module):
    """Compact recurrent memory with learned, surprise-conditioned writes.

    This is an educational abstraction inspired by neural long-term memory.
    It is not the exact Titans associative-memory update algorithm.
    """

    def __init__(self, d_model: int, memory_size: int):
        super().__init__()
        self.read = nn.Linear(memory_size, d_model)
        self.write_candidate = nn.Linear(d_model + memory_size, memory_size)
        self.write_gate = nn.Linear(d_model + memory_size + 1, memory_size)
        self.surprise_scale = nn.Parameter(torch.tensor(1.0))
        self.memory_size = memory_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, _ = x.shape
        state = x.new_zeros(batch, self.memory_size)
        outputs, surprises, gates = [], [], []
        for t in range(length):
            token = x[:, t]
            retrieved = self.read(state)
            surprise = (token - retrieved).pow(2).mean(dim=-1, keepdim=True).sqrt()
            write_input = torch.cat([token, state], dim=-1)
            gate_input = torch.cat([token, state, surprise], dim=-1)
            gate = torch.sigmoid(self.write_gate(gate_input))
            candidate = torch.tanh(self.write_candidate(write_input))
            # Surprise raises the write strength while the learned gate selects dimensions.
            strength = torch.sigmoid(self.surprise_scale * surprise)
            state = state + gate * strength * (candidate - state)
            outputs.append(retrieved + token)
            surprises.append(surprise.squeeze(-1))
            gates.append(gate.mean(dim=-1))
        return torch.stack(outputs, dim=1), {
            "surprise": torch.stack(surprises, dim=1),
            "write_gate": torch.stack(gates, dim=1),
            "final_memory": state,
        }
