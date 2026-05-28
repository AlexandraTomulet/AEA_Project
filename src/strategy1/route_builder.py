from typing import List
import numpy as np


def nearest_neighbor_route(
    assigned_nodes: List[int],
    depot_index: int,
    distance_matrix: np.ndarray
) -> List[int]:
    """
    Construiește o rută folosind euristica nearest neighbor.
    """
    if not assigned_nodes:
        return [depot_index, depot_index]

    unvisited = assigned_nodes.copy()
    route = [depot_index]
    current = depot_index

    while unvisited:
        next_node = min(unvisited, key=lambda node: distance_matrix[current, node])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    route.append(depot_index)
    return route


def nearest_insertion_route(
    assigned_nodes: List[int],
    depot_index: int,
    distance_matrix: np.ndarray
) -> List[int]:
    """
    Construiește o rută prin nearest insertion.
    Este de regulă mai stabilă decât nearest neighbor pentru multe instanțe.
    """
    if not assigned_nodes:
        return [depot_index, depot_index]

    if len(assigned_nodes) == 1:
        return [depot_index, assigned_nodes[0], depot_index]

    unvisited = assigned_nodes.copy()

    # Pornim de la nodul cel mai apropiat de depozit
    first_node = min(unvisited, key=lambda node: distance_matrix[depot_index, node])
    unvisited.remove(first_node)

    route = [depot_index, first_node, depot_index]

    while unvisited:
        # Alegem nodul cel mai apropiat de orice nod deja în rută
        candidate = min(
            unvisited,
            key=lambda node: min(distance_matrix[node, route_pos] for route_pos in route[:-1])
        )

        best_position = None
        best_increase = float("inf")

        for i in range(len(route) - 1):
            a = route[i]
            b = route[i + 1]
            increase = (
                distance_matrix[a, candidate]
                + distance_matrix[candidate, b]
                - distance_matrix[a, b]
            )
            if increase < best_increase:
                best_increase = increase
                best_position = i + 1

        route.insert(best_position, candidate)
        unvisited.remove(candidate)

    return route