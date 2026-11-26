from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor,
                  retention_weight: float = 0.0,
                  diagnostics: dict | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Next-token cross entropy plus an optional memory-retention regularizer."""
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    retention = ce.new_zeros(())
    if diagnostics and "final_memory" in diagnostics and retention_weight:
        # Small state-energy penalty is a simple, explicit retention ablation.
        retention = diagnostics["final_memory"].pow(2).mean()
    total = ce + retention_weight * retention
    return total, {"cross_entropy": ce.detach(), "retention": retention.detach()}
