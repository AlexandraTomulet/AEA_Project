from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.strategy2.model_v2 import EdgeScoringNetworkV2


@dataclass
class TrainingConfigV2:
    """
    Config pentru antrenarea modelului v2.
    """
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    hidden_dims: tuple[int, ...] = (256, 256, 128, 64)
    dropout: float = 0.2
    device: str = "cpu"


@dataclass
class TrainingArtifactsV2:
    """
    Artefactele rezultate din antrenare.
    """
    model: EdgeScoringNetworkV2
    feature_mean: np.ndarray
    feature_std: np.ndarray
    history: Dict[str, list[float]]
    best_val_loss: float
    best_epoch: int


def _normalize_train_val(
    X_train: np.ndarray,
    X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizează train și val folosind statisticile din train.
    """
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0)
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)

    X_train_norm = ((X_train - feature_mean) / feature_std).astype(np.float32)
    X_val_norm = ((X_val - feature_mean) / feature_std).astype(np.float32)

    return X_train_norm, X_val_norm, feature_mean.astype(np.float32), feature_std.astype(np.float32)


def _compute_pos_weight(y_train: np.ndarray) -> float:
    """
    Calculează pos_weight pentru BCEWithLogitsLoss.
    """
    positives = float((y_train == 1).sum())
    negatives = float((y_train == 0).sum())

    if positives == 0:
        return 1.0

    return negatives / positives


def _compute_binary_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> tuple[float, float]:
    """
    Calculează accuracy și F1.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    accuracy = (preds == targets).float().mean().item()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return float(accuracy), float(f1)


def train_edge_model_v2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfigV2
) -> TrainingArtifactsV2:
    """
    Antrenează modelul v2 pe split explicit train/validation.
    """
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X_train și X_val trebuie să fie matrici 2D.")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("y_train și y_val trebuie să fie vectori 1D.")

    X_train_norm, X_val_norm, feature_mean, feature_std = _normalize_train_val(X_train, X_val)

    device = torch.device(config.device)

    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_norm, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    model = EdgeScoringNetworkV2(
        input_dim=X_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout
    ).to(device)

    pos_weight_value = _compute_pos_weight(y_train)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        # train
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_X.size(0)
            train_loss_sum += loss.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        all_val_logits = []
        all_val_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_X)
                loss = criterion(logits, batch_y)

                batch_size = batch_X.size(0)
                val_loss_sum += loss.item() * batch_size
                val_count += batch_size

                all_val_logits.append(logits.cpu())
                all_val_targets.append(batch_y.cpu())

        val_loss = val_loss_sum / max(val_count, 1)

        val_logits = torch.cat(all_val_logits, dim=0)
        val_targets = torch.cat(all_val_targets, dim=0)
        val_accuracy, val_f1 = _compute_binary_metrics_from_logits(val_logits, val_targets)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_accuracy))
        history["val_f1"].append(float(val_f1))

        print(
            f"Epoch {epoch:03d}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping activat la epoca {epoch}.")
            break

    if best_state_dict is None:
        raise ValueError("Nu s-a salvat niciun best_state_dict.")

    model.load_state_dict(best_state_dict)

    return TrainingArtifactsV2(
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        history=history,
        best_val_loss=float(best_val_loss),
        best_epoch=int(best_epoch)
    )


def evaluate_edge_model_v2(
    model: EdgeScoringNetworkV2,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: str = "cpu"
) -> dict[str, float]:
    """
    Evaluează modelul pe test set.
    """
    X_test_norm = ((X_test - feature_mean) / feature_std).astype(np.float32)

    model = model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(X_tensor)
        loss_fn = nn.BCEWithLogitsLoss()
        test_loss = loss_fn(logits, y_tensor).item()

        accuracy, f1 = _compute_binary_metrics_from_logits(logits.cpu(), y_tensor.cpu())

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(accuracy),
        "test_f1": float(f1),
    }


def save_training_artifacts_v2(
    artifacts: TrainingArtifactsV2,
    output_dir: str
) -> None:
    """
    Salvează modelul, normalizarea și istoricul antrenării.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(artifacts.model.state_dict(), output_path / "edge_model_v2.pt")
    np.save(output_path / "feature_mean_v2.npy", artifacts.feature_mean)
    np.save(output_path / "feature_std_v2.npy", artifacts.feature_std)

    summary = {
        "best_val_loss": artifacts.best_val_loss,
        "best_epoch": artifacts.best_epoch,
        "history": artifacts.history,
    }

    with (output_path / "training_summary_v2.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)