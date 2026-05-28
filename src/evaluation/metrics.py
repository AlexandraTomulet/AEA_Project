from typing import List
import numpy as np


def route_cost(route: List[int], distance_matrix: np.ndarray) -> float:
    """
    Calculează costul unei rute complete.
    Ruta trebuie să includă și depozitul la început și la final.
    Ex: [0, 4, 2, 7, 0]
    """
    cost = 0.0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i], route[i + 1]]
    return cost


def total_cost(routes: List[List[int]], distance_matrix: np.ndarray) -> float:
    """
    Calculează costul total pentru toate rutele.
    """
    return sum(route_cost(route, distance_matrix) for route in routes)


def longest_route_cost(routes: List[List[int]], distance_matrix: np.ndarray) -> float:
    """
    Returnează costul celei mai lungi rute.
    """
    if not routes:
        return 0.0
    return max(route_cost(route, distance_matrix) for route in routes)