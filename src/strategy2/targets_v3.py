from __future__ import annotations

from typing import Dict, List, Set, Tuple
import numpy as np

from src.models.mtsp_instance import MTSPInstance
from src.strategy2.candidate_graph import CandidateGraph


def build_node_penalty_targets(
    instance: MTSPInstance,
    candidate_graph: CandidateGraph,
    positive_edges: Set[Tuple[int, int]]
) -> np.ndarray:
    """
    Construiește un target continuu pentru fiecare nod, folosit ca
    aproximație inspirată pentru node penalties.

    Scorul combină:
    - distanța la depozit
    - densitatea locală (via kNN average distance)
    - semnalul de utilizare în muchii pozitive
    """
    n = instance.num_nodes
    depot = instance.depot_index
    distance_matrix = instance.distance_matrix

    dist_to_depot = distance_matrix[:, depot].astype(np.float32)

    # media distanțelor către vecinii din candidate graph
    avg_neighbor_dist = np.zeros(n, dtype=np.float32)
    for node in range(n):
        outgoing = candidate_graph.adjacency[node]
        if outgoing:
            avg_neighbor_dist[node] = float(
                np.mean([edge.distance for edge in outgoing])
            )
        else:
            avg_neighbor_dist[node] = 0.0

    # cât de des participă nodul în muchii pozitive
    edge_usage = np.zeros(n, dtype=np.float32)
    for i, j in positive_edges:
        edge_usage[i] += 1.0
        edge_usage[j] += 1.0

    def normalize(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        min_val = float(arr.min())
        max_val = float(arr.max())
        if max_val - min_val < 1e-8:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    dist_norm = normalize(dist_to_depot)
    neigh_norm = normalize(avg_neighbor_dist)
    usage_norm = normalize(edge_usage)

    # combinație ponderată
    targets = (
        0.45 * dist_norm
        + 0.35 * neigh_norm
        + 0.20 * usage_norm
    ).astype(np.float32)

    # depot-ul nu vrem să fie penalizat puternic
    targets[depot] = 0.0

    return targets