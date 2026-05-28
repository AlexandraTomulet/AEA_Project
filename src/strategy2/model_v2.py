from __future__ import annotations

import torch
from torch import nn


class EdgeScoringNetworkV2(nn.Module):
    """
    Rețea mai puternică pentru edge scoring.
    Folosește:
    - straturi dense mai adânci
    - batch normalization
    - dropout
    - activări ReLU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 128, 64),
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returnează logits de formă [batch_size].
        """
        return self.network(x).squeeze(-1)