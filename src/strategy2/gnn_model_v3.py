from __future__ import annotations

import torch
from torch import nn


class EdgeNodeScoringNetworkV3(nn.Module):
    """
    Model v3 cu două head-uri:
    - edge head pentru edge scores
    - node head pentru node penalties

    Intrări:
    - edge_features: [num_edges, edge_input_dim]
    - node_features: [num_nodes, node_input_dim]

    Ieșiri:
    - edge_logits: [num_edges]
    - node_values: [num_nodes]
    """

    def __init__(
        self,
        edge_input_dim: int,
        node_input_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.edge_head = nn.Linear(hidden_dim, 1)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        edge_sources: torch.Tensor,
        edge_targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        edge_features: [E, F_e]
        node_features: [N, F_n]
        edge_sources: [E]
        edge_targets: [E]
        """
        node_embeddings = self.node_encoder(node_features)          # [N, H]
        edge_embeddings = self.edge_encoder(edge_features)          # [E, H]

        src_emb = node_embeddings[edge_sources]                     # [E, H]
        tgt_emb = node_embeddings[edge_targets]                     # [E, H]

        fused = torch.cat([edge_embeddings, src_emb, tgt_emb], dim=1)
        fused = self.fusion_layer(fused)

        edge_logits = self.edge_head(fused).squeeze(-1)             # [E]
        node_values = self.node_head(node_embeddings).squeeze(-1)   # [N]

        return edge_logits, node_values