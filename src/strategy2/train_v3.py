from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import torch
from torch import nn

from src.strategy2.data_loader_v3 import InstanceTrainingExampleV3
from src.strategy2.gnn_model_v3 import EdgeNodeScoringNetworkV3


@dataclass
class TrainingConfigV3:
    """
    Config pentru training-ul modelului v3.
    """
    hidden_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 12
    edge_loss_weight: float = 1.0
    node_loss_weight: float = 0.5
    device: str = "cpu"


@dataclass
class TrainingArtifactsV3:
    """
    Artefactele finale ale training-ului v3.
    """
    model: EdgeNodeScoringNetworkV3
    edge_feature_mean: np.ndarray
    edge_feature_std: np.ndarray
    history: Dict[str, list[float]]
    best_val_loss: float
    best_epoch: int


def _compute_edge_normalization(
    train_examples: List[InstanceTrainingExampleV3]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalizează doar edge features.
    Node features (coordonatele) le lăsăm brute.
    """
    all_edge_features = np.vstack([ex.edge_features for ex in train_examples]).astype(np.float32)
    mean = all_edge_features.mean(axis=0)
    std = all_edge_features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _normalize_examples(
    examples: List[InstanceTrainingExampleV3],
    mean: np.ndarray,
    std: np.ndarray
) -> List[InstanceTrainingExampleV3]:
    normalized_examples: List[InstanceTrainingExampleV3] = []

    for ex in examples:
        edge_features_norm = ((ex.edge_features - mean) / std).astype(np.float32)

        normalized_examples.append(
            InstanceTrainingExampleV3(
                instance_name=ex.instance_name,
                node_features=ex.node_features,
                edge_features=edge_features_norm,
                edge_labels=ex.edge_labels,
                edge_sources=ex.edge_sources,
                edge_targets=ex.edge_targets,
                node_targets=ex.node_targets
            )
        )

    return normalized_examples


def _compute_pos_weight(train_examples: List[InstanceTrainingExampleV3]) -> float:
    all_labels = np.concatenate([ex.edge_labels for ex in train_examples])
    positives = float((all_labels == 1).sum())
    negatives = float((all_labels == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def _evaluate_examples(
    model: EdgeNodeScoringNetworkV3,
    examples: List[InstanceTrainingExampleV3],
    edge_loss_fn,
    node_loss_fn,
    config: TrainingConfigV3,
) -> tuple[float, float, float]:
    """
    Returnează:
    - total loss
    - edge loss
    - node loss
    """
    device = torch.device(config.device)
    model.eval()

    total_loss_sum = 0.0
    edge_loss_sum = 0.0
    node_loss_sum = 0.0
    count = 0

    with torch.no_grad():
        for ex in examples:
            node_features = torch.tensor(ex.node_features, dtype=torch.float32, device=device)
            edge_features = torch.tensor(ex.edge_features, dtype=torch.float32, device=device)
            edge_sources = torch.tensor(ex.edge_sources, dtype=torch.long, device=device)
            edge_targets = torch.tensor(ex.edge_targets, dtype=torch.long, device=device)
            edge_labels = torch.tensor(ex.edge_labels, dtype=torch.float32, device=device)
            node_targets = torch.tensor(ex.node_targets, dtype=torch.float32, device=device)

            edge_logits, node_values = model(
                edge_features=edge_features,
                node_features=node_features,
                edge_sources=edge_sources,
                edge_targets=edge_targets
            )

            edge_loss = edge_loss_fn(edge_logits, edge_labels)
            node_loss = node_loss_fn(node_values, node_targets)
            total_loss = (
                config.edge_loss_weight * edge_loss
                + config.node_loss_weight * node_loss
            )

            total_loss_sum += total_loss.item()
            edge_loss_sum += edge_loss.item()
            node_loss_sum += node_loss.item()
            count += 1

    return (
        total_loss_sum / max(count, 1),
        edge_loss_sum / max(count, 1),
        node_loss_sum / max(count, 1),
    )


def train_model_v3(
    train_examples: List[InstanceTrainingExampleV3],
    val_examples: List[InstanceTrainingExampleV3],
    config: TrainingConfigV3
) -> TrainingArtifactsV3:
    device = torch.device(config.device)

    edge_feature_mean, edge_feature_std = _compute_edge_normalization(train_examples)
    train_examples = _normalize_examples(train_examples, edge_feature_mean, edge_feature_std)
    val_examples = _normalize_examples(val_examples, edge_feature_mean, edge_feature_std)

    input_edge_dim = train_examples[0].edge_features.shape[1]
    model = EdgeNodeScoringNetworkV3(
        edge_input_dim=input_edge_dim,
        node_input_dim=2,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout
    ).to(device)

    pos_weight_value = _compute_pos_weight(train_examples)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    edge_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    node_loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    history = {
        "train_total_loss": [],
        "train_edge_loss": [],
        "train_node_loss": [],
        "val_total_loss": [],
        "val_edge_loss": [],
        "val_node_loss": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()

        train_total_sum = 0.0
        train_edge_sum = 0.0
        train_node_sum = 0.0
        train_count = 0

        for ex in train_examples:
            node_features = torch.tensor(ex.node_features, dtype=torch.float32, device=device)
            edge_features = torch.tensor(ex.edge_features, dtype=torch.float32, device=device)
            edge_sources = torch.tensor(ex.edge_sources, dtype=torch.long, device=device)
            edge_targets = torch.tensor(ex.edge_targets, dtype=torch.long, device=device)
            edge_labels = torch.tensor(ex.edge_labels, dtype=torch.float32, device=device)
            node_targets = torch.tensor(ex.node_targets, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            edge_logits, node_values = model(
                edge_features=edge_features,
                node_features=node_features,
                edge_sources=edge_sources,
                edge_targets=edge_targets
            )

            edge_loss = edge_loss_fn(edge_logits, edge_labels)
            node_loss = node_loss_fn(node_values, node_targets)
            total_loss = (
                config.edge_loss_weight * edge_loss
                + config.node_loss_weight * node_loss
            )

            total_loss.backward()
            optimizer.step()

            train_total_sum += total_loss.item()
            train_edge_sum += edge_loss.item()
            train_node_sum += node_loss.item()
            train_count += 1

        train_total_loss = train_total_sum / max(train_count, 1)
        train_edge_loss = train_edge_sum / max(train_count, 1)
        train_node_loss = train_node_sum / max(train_count, 1)

        val_total_loss, val_edge_loss, val_node_loss = _evaluate_examples(
            model=model,
            examples=val_examples,
            edge_loss_fn=edge_loss_fn,
            node_loss_fn=node_loss_fn,
            config=config
        )

        history["train_total_loss"].append(float(train_total_loss))
        history["train_edge_loss"].append(float(train_edge_loss))
        history["train_node_loss"].append(float(train_node_loss))
        history["val_total_loss"].append(float(val_total_loss))
        history["val_edge_loss"].append(float(val_edge_loss))
        history["val_node_loss"].append(float(val_node_loss))

        print(
            f"Epoch {epoch:03d}/{config.num_epochs} | "
            f"train_total={train_total_loss:.4f} | "
            f"train_edge={train_edge_loss:.4f} | "
            f"train_node={train_node_loss:.4f} | "
            f"val_total={val_total_loss:.4f} | "
            f"val_edge={val_edge_loss:.4f} | "
            f"val_node={val_node_loss:.4f}"
        )

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping activat la epoca {epoch}.")
            break

    if best_state_dict is None:
        raise ValueError("Nu s-a salvat niciun checkpoint bun.")

    model.load_state_dict(best_state_dict)

    return TrainingArtifactsV3(
        model=model,
        edge_feature_mean=edge_feature_mean,
        edge_feature_std=edge_feature_std,
        history=history,
        best_val_loss=float(best_val_loss),
        best_epoch=int(best_epoch)
    )


def evaluate_model_v3(
    model: EdgeNodeScoringNetworkV3,
    test_examples: List[InstanceTrainingExampleV3],
    edge_feature_mean: np.ndarray,
    edge_feature_std: np.ndarray,
    config: TrainingConfigV3
) -> dict[str, float]:
    test_examples = _normalize_examples(test_examples, edge_feature_mean, edge_feature_std)

    pos_weight = torch.tensor(1.0, dtype=torch.float32, device=config.device)
    edge_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    node_loss_fn = nn.MSELoss()

    test_total_loss, test_edge_loss, test_node_loss = _evaluate_examples(
        model=model,
        examples=test_examples,
        edge_loss_fn=edge_loss_fn,
        node_loss_fn=node_loss_fn,
        config=config
    )

    return {
        "test_total_loss": float(test_total_loss),
        "test_edge_loss": float(test_edge_loss),
        "test_node_loss": float(test_node_loss),
    }


def save_training_artifacts_v3(
    artifacts: TrainingArtifactsV3,
    output_dir: str
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(artifacts.model.state_dict(), output_path / "gnn_model_v3.pt")
    np.save(output_path / "edge_feature_mean_v3.npy", artifacts.edge_feature_mean)
    np.save(output_path / "edge_feature_std_v3.npy", artifacts.edge_feature_std)

    summary = {
        "best_val_loss": artifacts.best_val_loss,
        "best_epoch": artifacts.best_epoch,
        "history": artifacts.history,
    }

    with (output_path / "training_summary_v3.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)