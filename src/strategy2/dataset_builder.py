from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np

from src.models.mtsp_instance import MTSPInstance
from src.strategy1.solver import Strategy1Solver
from src.strategy2.candidate_graph import CandidateGraph, build_candidate_graph
from src.strategy2.config import Strategy2Config
from src.strategy2.features import extract_edge_features


@dataclass
class EdgeDataset:
    """
    Dataset pentru antrenarea modelului de edge scoring.
    """
    X: np.ndarray
    y: np.ndarray
    edge_index: List[Tuple[int, int]]
    instance_names: List[str]


def _extract_positive_edges_from_routes(routes: List[List[int]]) -> Set[Tuple[int, int]]:
    """
    Extrage muchiile orientate din rutele soluției.
    Exemplu:
    [0, 4, 7, 0] -> (0,4), (4,7), (7,0)
    """
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
    """
    Selectează muchii negative dintre muchiile candidate care nu apar în soluție.
    Numărul de negative va fi aproximativ:
    negative_sampling_ratio * num_positive
    """
    all_candidate_edges = set(edge_index)
    negative_candidates = list(all_candidate_edges - positive_edges)

    num_positive = len(positive_edges)
    num_negative = min(len(negative_candidates), negative_sampling_ratio * num_positive)

    if num_negative <= 0:
        return set()

    chosen_indices = rng.choice(len(negative_candidates), size=num_negative, replace=False)
    selected_negative_edges = {negative_candidates[idx] for idx in chosen_indices}

    return selected_negative_edges


def build_edge_dataset_for_instance(
    instance: MTSPInstance,
    strategy1_solver: Strategy1Solver,
    config: Strategy2Config,
    rng: np.random.Generator
) -> EdgeDataset:
    """
    Construiește dataset-ul de muchii pentru o singură instanță.
    """
    strategy1_result = strategy1_solver.solve(instance)

    if not strategy1_result.is_valid:
        raise ValueError(
            f"Strategia 1 a produs o soluție invalidă pentru instanța {instance.name}: "
            f"{strategy1_result.validation_errors}"
        )

    positive_edges = _extract_positive_edges_from_routes(strategy1_result.routes)

    candidate_graph = build_candidate_graph(
        instance=instance,
        num_neighbors=config.num_candidate_neighbors
    )

    X_all, edge_index = extract_edge_features(
        instance=instance,
        candidate_graph=candidate_graph
    )

    negative_edges = _sample_negative_edges(
        edge_index=edge_index,
        positive_edges=positive_edges,
        negative_sampling_ratio=config.negative_sampling_ratio,
        rng=rng
    )

    selected_edges = positive_edges | negative_edges

    selected_rows: List[int] = []
    labels: List[int] = []

    for row_idx, edge in enumerate(edge_index):
        if edge in selected_edges:
            selected_rows.append(row_idx)
            labels.append(1 if edge in positive_edges else 0)

    X = X_all[selected_rows]
    y = np.array(labels, dtype=np.float32)
    selected_edge_index = [edge_index[idx] for idx in selected_rows]
    instance_names = [instance.name] * len(selected_rows)

    return EdgeDataset(
        X=X,
        y=y,
        edge_index=selected_edge_index,
        instance_names=instance_names
    )


def build_edge_dataset_for_instances(
    instances: List[MTSPInstance],
    config: Strategy2Config,
    random_seed: int = 42
) -> EdgeDataset:
    """
    Construiește un dataset agregat pentru mai multe instanțe.
    """
    rng = np.random.default_rng(random_seed)
    strategy1_solver = Strategy1Solver()

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    edge_index_all: List[Tuple[int, int]] = []
    instance_names_all: List[str] = []

    for instance in instances:
        dataset = build_edge_dataset_for_instance(
            instance=instance,
            strategy1_solver=strategy1_solver,
            config=config,
            rng=rng
        )

        X_parts.append(dataset.X)
        y_parts.append(dataset.y)
        edge_index_all.extend(dataset.edge_index)
        instance_names_all.extend(dataset.instance_names)

    if not X_parts:
        raise ValueError("Nu au fost generate exemple pentru dataset.")

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
    Salvează dataset-ul în fișiere .npy și metadatele într-un CSV.
    """
    import pandas as pd

    np.save(f"{output_prefix}_X.npy", dataset.X)
    np.save(f"{output_prefix}_y.npy", dataset.y)

    metadata_df = pd.DataFrame({
        "instance_name": dataset.instance_names,
        "source": [edge[0] for edge in dataset.edge_index],
        "target": [edge[1] for edge in dataset.edge_index],
        "label": dataset.y.astype(int),
    })
    metadata_df.to_csv(f"{output_prefix}_meta.csv", index=False)