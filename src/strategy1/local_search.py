from typing import List
import numpy as np
from src.evaluation.metrics import route_cost


def two_opt(route: List[int], distance_matrix: np.ndarray) -> List[int]:
    """
    Îmbunătățește o rută folosind euristica 2-opt.
    Ruta trebuie să înceapă și să se termine în depozit.
    """
    if len(route) <= 4:
        return route

    best_route = route[:]
    best_cost = route_cost(best_route, distance_matrix)

    improved = True
    while improved:
        improved = False

        # Evităm pozițiile 0 și ultima, pentru că sunt depozitul
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue

                candidate_route = (
                    best_route[:i]
                    + best_route[i:j][::-1]
                    + best_route[j:]
                )

                candidate_cost = route_cost(candidate_route, distance_matrix)

                if candidate_cost + 1e-9 < best_cost:
                    best_route = candidate_route
                    best_cost = candidate_cost
                    improved = True

        # continuă până nu mai există îmbunătățiri

    return best_route