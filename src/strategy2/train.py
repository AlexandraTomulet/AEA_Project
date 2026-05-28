from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.strategy2.config import Strategy2Config
from src.strategy2.model import EdgeScoringMLP


@dataclass
class TrainingArtifacts:
    """
    Obiect care conține tot ce trebuie salvat după antrenare.
    """
    model: EdgeScoringMLP
    feature_mean: np.ndarray
    feature_std: np.ndarray
    history: Dict[str, list[float]]


def _compute_class_weight(y: np.ndarray) -> float:
    """
    Calculează pos_weight pentru BCEWithLogitsLoss.
    """
    num_positive = float((y == 1).sum())
    num_negative = float((y == 0).sum())

    if num_positive == 0:
        return 1.0

    return num_negative / num_positive


def _normalize_features(
    X_train: np.ndarray,
    X_other: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizează folosind media și deviația standard din train.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std = np.where(std < 1e-8, 1.0, std)

    X_train_norm = (X_train - mean) / std
    X_other_norm = (X_other - mean) / std

    return X_train_norm.astype(np.float32), X_other_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _compute_binary_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[float, float]:
    """
    Returnează accuracy și F1-score.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    correct = (preds == targets).float().mean().item()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return float(correct), float(f1)


def train_edge_model(
    X: np.ndarray,
    y: np.ndarray,
    config: Strategy2Config,
    validation_ratio: float = 0.2,
    random_seed: int = 42
) -> TrainingArtifacts:
    """
    Antrenează modelul de edge scoring.
    """
    if X.ndim != 2:
        raise ValueError("X trebuie să fie matrice 2D.")
    if y.ndim != 1:
        raise ValueError("y trebuie să fie vector 1D.")
    if len(X) != len(y):
        raise ValueError("X și y trebuie să aibă același număr de exemple.")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split_idx = int((1.0 - validation_ratio) * len(X))
    X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    X_train, X_val, feature_mean, feature_std = _normalize_features(X_train_raw, X_val_raw)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = EdgeScoringMLP(input_dim=X.shape[1], config=config)

    pos_weight_value = _compute_class_weight(y_train)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            batch_size = batch_X.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        all_val_logits = []
        all_val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                batch_size = batch_X.size(0)
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size

                all_val_logits.append(logits)
                all_val_targets.append(batch_y)

        val_loss = val_loss_sum / max(val_count, 1)

        val_logits = torch.cat(all_val_logits, dim=0)
        val_targets = torch.cat(all_val_targets, dim=0)

        val_accuracy, val_f1 = _compute_binary_metrics_from_logits(val_logits, val_targets)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))
        history["val_f1"].append(float(val_f1))

        print(
            f"Epoch {epoch:02d}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

    return TrainingArtifacts(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        history=history
    )


def save_training_artifacts(
    artifacts: TrainingArtifacts,
    output_dir: str
) -> None:
    """
    Salvează modelul, normalizarea și istoricul antrenării.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(artifacts.model.state_dict(), output_path / "edge_model.pt")
    np.save(output_path / "feature_mean.npy", artifacts.feature_mean)
    np.save(output_path / "feature_std.npy", artifacts.feature_std)

    import json
    with (output_path / "history.json").open("w", encoding="utf-8") as f:
        json.dump(artifacts.history, f, indent=2)