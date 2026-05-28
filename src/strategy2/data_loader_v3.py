from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.models.mtsp_instance import MTSPInstance
from src.strategy2.advanced_data import (
    SyntheticDatasetConfig,
    build_synthetic_instance_split,
)


@dataclass
class InstanceTrainingExampleV3:
    """
    Exemplu de training pentru o instanță:
    - node_features
    - edge_features
    - edge labels
    - edge indices (source, target)
    - node targets
    """
    instance_name: str
    node_features: np.ndarray          # [N, 2]
    edge_features: np.ndarray          # [E, F]
    edge_labels: np.ndarray            # [E]
    edge_sources: np.ndarray           # [E]
    edge_targets: np.ndarray           # [E]
    node_targets: np.ndarray           # [N]


def _build_instance_lookup() -> dict[str, MTSPInstance]:
    """
    Reconstruiește aceleași instanțe sintetice folosite la build-ul dataset-ului v3.
    """
    synthetic_config = SyntheticDatasetConfig(
        train_instances_per_setting=8,
        val_instances_per_setting=3,
        test_instances_per_setting=3,
        customer_sizes=(30, 50, 75, 100),
        num_salesmen_values=(2, 3, 5),
    )

    train_instances, val_instances, test_instances = build_synthetic_instance_split(
        config=synthetic_config,
        random_seed=42
    )

    all_instances = train_instances + val_instances + test_instances
    return {instance.name: instance for instance in all_instances}


def load_instance_examples_v3(prefix: str) -> List[InstanceTrainingExampleV3]:
    """
    Încarcă dataset-ul v3 salvat și îl transformă în exemple per instanță.
    prefix exemplu:
    data/processed/strategy2_v3/train_dataset_v3
    """
    base_path = Path(prefix)

    X_edges = np.load(f"{prefix}_X_edges.npy")
    y_edges = np.load(f"{prefix}_y_edges.npy")
    edge_meta = pd.read_csv(f"{prefix}_edge_meta.csv")

    node_targets_arr = np.load(f"{prefix}_node_targets.npy", allow_pickle=True)
    node_meta = pd.read_csv(f"{prefix}_node_meta.csv")

    instance_lookup = _build_instance_lookup()

    examples: List[InstanceTrainingExampleV3] = []

    # map instance_name -> node_targets
    node_target_map = {
        row["instance_name"]: node_targets_arr[idx].astype(np.float32)
        for idx, row in node_meta.iterrows()
    }

    grouped = edge_meta.groupby("instance_name", sort=False)

    for instance_name, group_df in grouped:
        if instance_name not in instance_lookup:
            raise ValueError(f"Instanța {instance_name} nu există în lookup.")

        instance = instance_lookup[instance_name]
        node_features = instance.coordinates.astype(np.float32)

        row_indices = group_df.index.to_numpy()

        edge_features = X_edges[row_indices].astype(np.float32)
        edge_labels = y_edges[row_indices].astype(np.float32)
        edge_sources = group_df["source"].to_numpy(dtype=np.int64)
        edge_targets = group_df["target"].to_numpy(dtype=np.int64)

        if instance_name not in node_target_map:
            raise ValueError(f"Nu există node targets pentru instanța {instance_name}.")

        node_targets = node_target_map[instance_name].astype(np.float32)

        examples.append(
            InstanceTrainingExampleV3(
                instance_name=instance_name,
                node_features=node_features,
                edge_features=edge_features,
                edge_labels=edge_labels,
                edge_sources=edge_sources,
                edge_targets=edge_targets,
                node_targets=node_targets
            )
        )

    return examples