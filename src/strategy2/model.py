from __future__ import annotations

import torch
from torch import nn

from src.strategy2.config import Strategy2Config


class EdgeScoringMLP(nn.Module):
    """
    Rețea MLP pentru edge scoring.
    Intrare: vector de feature-uri pentru o muchie candidat.
    Ieșire: logit scalar pentru clasificare binară (muchie bună / nu).
    """

    def __init__(self, input_dim: int, config: Strategy2Config) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_1, config.hidden_dim_2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returnează logits de formă [batch_size].
        """
        logits = self.network(x).squeeze(-1)
        return logits