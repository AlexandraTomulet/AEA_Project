from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from src.strategy2.model_v2 import EdgeScoringNetworkV2


class EdgeScorePredictorV2:
    """
    Predictor pentru modelul v2 de edge scoring.
    """

    def __init__(
        self,
        model_dir: str,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256, 128, 64),
        dropout: float = 0.2,
        device: str = "cpu"
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        self.feature_mean = np.load(self.model_dir / "feature_mean_v2.npy").astype(np.float32)
        self.feature_std = np.load(self.model_dir / "feature_std_v2.npy").astype(np.float32)

        self.model = EdgeScoringNetworkV2(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        state_dict = torch.load(
            self.model_dir / "edge_model_v2.pt",
            map_location=self.device
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.feature_mean) / self.feature_std).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returnează probabilități pentru muchiile candidate.
        """
        X_norm = self._normalize(X)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)

        return probs.cpu().numpy().astype(np.float32)