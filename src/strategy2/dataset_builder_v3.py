from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple
import numpy as np
import pandas as pd

from src.models.mtsp_instance import MTSPInstance
from src.strategy2.candidate_graph import build_candidate_graph
from src.strategy2.config import Strategy2Config
from src.strategy2.features import extract_edge_features
from src.strategy2.targets_v3 import build_node_penalty_targets
from src.strategy2.teacher_solver import MultiRestartTeacherSolver


@dataclass
class EdgeNodeDatasetV3:
    """
    Dataset pentru modelul v3:
    - exemple pe muchii
    - targets pe noduri
    """
    X_edges: np.ndarray
    y_edges: np.ndarray
    edge_index: List[Tuple[int, int]]
    edge_instance_names: List[str]

    node_targets: List[np.ndarray]
    node_instance_names: List[str]


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


def build_edge_node_dataset_v3(
    instances: List[MTSPInstance],
    strategy2_config: Strategy2Config,
    teacher_solver: MultiRestartTeacherSolver,
    random_seed: int = 42
) -> EdgeNodeDatasetV3:
    rng = np.random.default_rng(random_seed)

    X_edge_parts: List[np.ndarray] = []
    y_edge_parts: List[np.ndarray] = []
    edge_index_all: List[Tuple[int, int]] = []
    edge_instance_names_all: List[str] = []

    node_targets_all: List[np.ndarray] = []
    node_instance_names_all: List[str] = []

    for idx, instance in enumerate(instances, start=1):
        print(f"[{idx}/{len(instances)}] Building v3 targets for {instance.name}")

        teacher_result = teacher_solver.solve(instance)
        positive_edges = _extract_positive_edges_from_routes(teacher_result.routes)

        candidate_graph = build_candidate_graph(
            instance=instance,
            num_neighbors=strategy2_config.num_candidate_neighbors
        )

        X_edges_all, edge_index = extract_edge_features(
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

        X_selected = X_edges_all[selected_rows]
        y_selected = np.array(labels, dtype=np.float32)

        X_edge_parts.append(X_selected)
        y_edge_parts.append(y_selected)
        edge_index_all.extend([edge_index[row_idx] for row_idx in selected_rows])
        edge_instance_names_all.extend([instance.name] * len(selected_rows))

        node_targets = build_node_penalty_targets(
            instance=instance,
            candidate_graph=candidate_graph,
            positive_edges=positive_edges
        )
        node_targets_all.append(node_targets)
        node_instance_names_all.append(instance.name)

    if not X_edge_parts:
        raise ValueError("Dataset v3 nu conține exemple de muchii.")

    X_edges = np.vstack(X_edge_parts).astype(np.float32)
    y_edges = np.concatenate(y_edge_parts).astype(np.float32)

    return EdgeNodeDatasetV3(
        X_edges=X_edges,
        y_edges=y_edges,
        edge_index=edge_index_all,
        edge_instance_names=edge_instance_names_all,
        node_targets=node_targets_all,
        node_instance_names=node_instance_names_all
    )


def save_edge_node_dataset_v3(dataset: EdgeNodeDatasetV3, output_prefix: str) -> None:
    """
    Salvează partea de muchii + target-urile pe noduri.
    """
    np.save(f"{output_prefix}_X_edges.npy", dataset.X_edges)
    np.save(f"{output_prefix}_y_edges.npy", dataset.y_edges)

    edge_meta_df = pd.DataFrame({
        "instance_name": dataset.edge_instance_names,
        "source": [edge[0] for edge in dataset.edge_index],
        "target": [edge[1] for edge in dataset.edge_index],
        "label": dataset.y_edges.astype(int),
    })
    edge_meta_df.to_csv(f"{output_prefix}_edge_meta.csv", index=False)

    np.save(
        f"{output_prefix}_node_targets.npy",
        np.array(dataset.node_targets, dtype=object),
        allow_pickle=True
    )

    node_meta_df = pd.DataFrame({
        "instance_name": dataset.node_instance_names,
        "num_nodes": [len(targets) for targets in dataset.node_targets],
    })
    node_meta_df.to_csv(f"{output_prefix}_node_meta.csv", index=False)