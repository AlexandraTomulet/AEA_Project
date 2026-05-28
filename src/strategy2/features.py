from __future__ import annotations

from math import atan2, pi
from typing import Dict, List, Tuple
import numpy as np

from src.models.mtsp_instance import MTSPInstance
from src.strategy2.candidate_graph import CandidateEdge, CandidateGraph


def _polar_angle(cx: float, cy: float, x: float, y: float) -> float:
    """
    Unghiul polar al punctului (x, y) față de centrul (cx, cy).
    """
    return atan2(y - cy, x - cx)


def _normalized_angle_difference(angle_a: float, angle_b: float) -> float:
    """
    Diferența unghiulară normalizată în [0, 1].
    """
    diff = abs(angle_a - angle_b)
    diff = min(diff, 2 * pi - diff)
    return diff / pi


def _build_reverse_edge_lookup(candidate_graph: CandidateGraph) -> set[Tuple[int, int]]:
    """
    Creează un set pentru verificare rapidă a existenței muchiei inverse.
    """
    return {(edge.source, edge.target) for edge in candidate_graph.edges}


def extract_edge_features(
    instance: MTSPInstance,
    candidate_graph: CandidateGraph
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extrage matricea de feature-uri pentru toate muchiile candidate.
    Returnează:
    - X: array [num_edges, num_features]
    - edge_index: listă de perechi (source, target) în aceeași ordine
    """
    if instance.coordinates is None:
        raise ValueError("Strategia 2 necesită coordonate disponibile.")

    coords = instance.coordinates
    depot = instance.depot_index
    depot_x, depot_y = coords[depot]
    distance_matrix = instance.distance_matrix

    reverse_lookup = _build_reverse_edge_lookup(candidate_graph)

    node_angles: Dict[int, float] = {}
    for node in range(instance.num_nodes):
        x, y = coords[node]
        node_angles[node] = _polar_angle(depot_x, depot_y, x, y)

    features: List[List[float]] = []
    edge_index: List[Tuple[int, int]] = []

    for edge in candidate_graph.edges:
        i = edge.source
        j = edge.target

        xi, yi = coords[i]
        xj, yj = coords[j]

        dist_ij = distance_matrix[i, j]
        dist_i_depot = distance_matrix[i, depot]
        dist_j_depot = distance_matrix[j, depot]

        angle_i = node_angles[i]
        angle_j = node_angles[j]
        angle_diff = _normalized_angle_difference(angle_i, angle_j)

        reverse_exists = 1.0 if (j, i) in reverse_lookup else 0.0

        feature_vector = [
            float(xi),
            float(yi),
            float(xj),
            float(yj),
            float(dist_ij),
            float(dist_i_depot),
            float(dist_j_depot),
            float(angle_diff),
            float(edge.rank),
            float(reverse_exists),
        ]

        features.append(feature_vector)
        edge_index.append((i, j))

    X = np.array(features, dtype=np.float32)
    return X, edge_index