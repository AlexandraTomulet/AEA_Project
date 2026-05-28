from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple
import numpy as np
import pandas as pd

from src.models.mtsp_instance import MTSPInstance
from src.strategy2.candidate_graph import build_candidate_graph
from src.strategy2.config import Strategy2Config
from src.strategy2.features import extract_edge_features
from src.strategy2.teacher_solver import MultiRestartTeacherSolver


@dataclass
class EdgeDataset:
    X: np.ndarray
    y: np.ndarray
    edge_index: List[Tuple[int, int]]
    instance_names: List[str]


def _extract_positive_edges_from_routes(routes: List[List[int]]) -> Set[Tuple[int, int]]:
    positive_edges: Set[Tuple[int, int]] = set()
    for route in routes:
        for i in range(len(route) - 1):
            positive_edges.add((route[i], route[i + 1]))
    return positive_edges


def _sample_negative_edges(
    edge_index: List[Tuple[int, int]],
    positive_edges: Set[Tuple[int, int]],
    negative_sampling_ratio: int,
    rng: np.random.Generator
) -> Set[Tuple[int, int]]:
    all_candidate_edges = set(edge_index)
    negative_candidates = list(all_candidate_edges - positive_edges)

    num_positive = len(positive_edges)
    num_negative = min(len(negative_candidates), negative_sampling_ratio * num_positive)

    if num_negative <= 0:
        return set()

    chosen_indices = rng.choice(len(negative_candidates), size=num_negative, replace=False)
    return {negative_candidates[idx] for idx in chosen_indices}


def build_edge_dataset_for_instances_v2(
    instances: List[MTSPInstance],
    strategy2_config: Strategy2Config,
    teacher_solver: MultiRestartTeacherSolver,
    random_seed: int = 42
) -> EdgeDataset:
    """
    Construiește dataset-ul de muchii pentru un set de instanțe.
    """
    rng = np.random.default_rng(random_seed)

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    edge_index_all: List[Tuple[int, int]] = []
    instance_names_all: List[str] = []

    for idx, instance in enumerate(instances, start=1):
        print(f"[{idx}/{len(instances)}] Building labels for {instance.name}")

        teacher_result = teacher_solver.solve(instance)
        positive_edges = _extract_positive_edges_from_routes(teacher_result.routes)

        candidate_graph = build_candidate_graph(
            instance=instance,
            num_neighbors=strategy2_config.num_candidate_neighbors
        )

        X_all, edge_index = extract_edge_features(
            instance=instance,
            candidate_graph=candidate_graph
        )

        negative_edges = _sample_negative_edges(
            edge_index=edge_index,
            positive_edges=positive_edges,
            negative_sampling_ratio=strategy2_config.negative_sampling_ratio,
            rng=rng
        )

        selected_edges = positive_edges | negative_edges

        selected_rows: List[int] = []
        labels: List[int] = []

        for row_idx, edge in enumerate(edge_index):
            if edge in selected_edges:
                selected_rows.append(row_idx)
                labels.append(1 if edge in positive_edges else 0)

        X_selected = X_all[selected_rows]
        y_selected = np.array(labels, dtype=np.float32)

        X_parts.append(X_selected)
        y_parts.append(y_selected)
        edge_index_all.extend([edge_index[row_idx] for row_idx in selected_rows])
        instance_names_all.extend([instance.name] * len(selected_rows))

    if not X_parts:
        raise ValueError("Dataset-ul nu conține exemple.")

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts).astype(np.float32)

    return EdgeDataset(
        X=X,
        y=y,
        edge_index=edge_index_all,
        instance_names=instance_names_all
    )


def save_edge_dataset(dataset: EdgeDataset, output_prefix: str) -> None:
    """
    Salvează dataset-ul în format .npy + metadata CSV.
    """
    np.save(f"{output_prefix}_X.npy", dataset.X)
    np.save(f"{output_prefix}_y.npy", dataset.y)

    metadata_df = pd.DataFrame({
        "instance_name": dataset.instance_names,
        "source": [edge[0] for edge in dataset.edge_index],
        "target": [edge[1] for edge in dataset.edge_index],
        "label": dataset.y.astype(int),
    })
    metadata_df.to_csv(f"{output_prefix}_meta.csv", index=False)