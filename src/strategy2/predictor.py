from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.strategy2.config import Strategy2Config
from src.strategy2.model import EdgeScoringMLP


class EdgeScorePredictor:
    """
    Predictor pentru scorurile muchiilor candidate.
    Încarcă modelul antrenat și statisticile de normalizare.
    """

    def __init__(
        self,
        model_dir: str,
        input_dim: int,
        config: Strategy2Config
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config

        self.feature_mean = np.load(self.model_dir / "feature_mean.npy").astype(np.float32)
        self.feature_std = np.load(self.model_dir / "feature_std.npy").astype(np.float32)

        self.model = EdgeScoringMLP(input_dim=input_dim, config=config)
        state_dict = torch.load(self.model_dir / "edge_model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Normalizează feature-urile cu statisticile din train.
        """
        return ((X - self.feature_mean) / self.feature_std).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returnează probabilitățile pentru clasa pozitivă.
        """
        X_norm = self._normalize(X)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)

        return probs.numpy().astype(np.float32)