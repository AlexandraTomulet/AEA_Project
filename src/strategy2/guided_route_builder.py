from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def _get_edge_score(
    edge_scores: Dict[Tuple[int, int], float],
    source: int,
    target: int
) -> float:
    """
    Returnează scorul muchiei orientate source -> target.
    Dacă muchia nu există în candidate graph, returnează 0.
    """
    return float(edge_scores.get((source, target), 0.0))


def guided_nearest_insertion_route(
    assigned_nodes: List[int],
    depot_index: int,
    distance_matrix: np.ndarray,
    edge_scores: Dict[Tuple[int, int], float],
    alpha_insertion_cost: float,
    beta_edge_score: float
) -> List[int]:
    """
    Construiește o rută folosind nearest insertion ghidat de model.

    La fiecare inserare, se optimizează un scor combinat:
    combined_score = alpha * insertion_cost - beta * edge_bonus

    unde:
    edge_bonus = score(a, candidate) + score(candidate, b)
    """
    if not assigned_nodes:
        return [depot_index, depot_index]

    if len(assigned_nodes) == 1:
        return [depot_index, assigned_nodes[0], depot_index]

    unvisited = assigned_nodes.copy()

    # Primul nod: cel mai apropiat de depozit, ajustat cu scor model
    first_node = min(
        unvisited,
        key=lambda node: (
            alpha_insertion_cost * distance_matrix[depot_index, node]
            - beta_edge_score * (
                _get_edge_score(edge_scores, depot_index, node)
                + _get_edge_score(edge_scores, node, depot_index)
            )
        )
    )
    unvisited.remove(first_node)

    route = [depot_index, first_node, depot_index]

    while unvisited:
        best_candidate = None
        best_position = None
        best_combined_score = float("inf")

        for candidate in unvisited:
            for i in range(len(route) - 1):
                a = route[i]
                b = route[i + 1]

                insertion_cost = (
                    distance_matrix[a, candidate]
                    + distance_matrix[candidate, b]
                    - distance_matrix[a, b]
                )

                edge_bonus = (
                    _get_edge_score(edge_scores, a, candidate)
                    + _get_edge_score(edge_scores, candidate, b)
                )

                combined_score = (
                    alpha_insertion_cost * insertion_cost
                    - beta_edge_score * edge_bonus
                )

                if combined_score < best_combined_score:
                    best_combined_score = combined_score
                    best_candidate = candidate
                    best_position = i + 1

        route.insert(best_position, best_candidate)
        unvisited.remove(best_candidate)

    return route