from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.strategy2.gnn_model_v3 import EdgeNodeScoringNetworkV3


class EdgeNodePredictorV3:
    """
    Predictor pentru modelul v3:
    - edge scores
    - node penalties
    """

    def __init__(
        self,
        model_dir: str,
        edge_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        device: str = "cpu"
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        self.edge_feature_mean = np.load(
            self.model_dir / "edge_feature_mean_v3.npy"
        ).astype(np.float32)
        self.edge_feature_std = np.load(
            self.model_dir / "edge_feature_std_v3.npy"
        ).astype(np.float32)

        self.model = EdgeNodeScoringNetworkV3(
            edge_input_dim=edge_input_dim,
            node_input_dim=2,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)

        state_dict = torch.load(
            self.model_dir / "gnn_model_v3.pt",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _normalize_edge_features(self, edge_features: np.ndarray) -> np.ndarray:
        return ((edge_features - self.edge_feature_mean) / self.edge_feature_std).astype(np.float32)

    def predict(
        self,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        edge_sources: np.ndarray,
        edge_targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returnează:
        - edge probabilities
        - node penalties
        """
        edge_features_norm = self._normalize_edge_features(edge_features)

        node_tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        edge_tensor = torch.tensor(edge_features_norm, dtype=torch.float32, device=self.device)
        src_tensor = torch.tensor(edge_sources, dtype=torch.long, device=self.device)
        tgt_tensor = torch.tensor(edge_targets, dtype=torch.long, device=self.device)

        with torch.no_grad():
            edge_logits, node_values = self.model(
                edge_features=edge_tensor,
                node_features=node_tensor,
                edge_sources=src_tensor,
                edge_targets=tgt_tensor
            )

            edge_probs = torch.sigmoid(edge_logits)

        return (
            edge_probs.cpu().numpy().astype(np.float32),
            node_values.cpu().numpy().astype(np.float32)
        )