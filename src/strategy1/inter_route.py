from typing import List
import numpy as np
from src.evaluation.metrics import total_cost


def relocate_between_routes(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    depot_index: int
) -> List[List[int]]:
    """
    Încearcă să mute câte un client dintr-o rută în alta dacă scade costul total.
    Este o versiune simplă, dar foarte utilă pentru a rafina soluția globală.
    """
    best_routes = [route[:] for route in routes]
    best_total = total_cost(best_routes, distance_matrix)

    improved = True
    while improved:
        improved = False

        for from_idx in range(len(best_routes)):
            for to_idx in range(len(best_routes)):
                if from_idx == to_idx:
                    continue

                source_route = best_routes[from_idx]
                target_route = best_routes[to_idx]

                # nu mutăm depozitul; clienții sunt între prima și ultima poziție
                if len(source_route) <= 3:
                    continue  # ruta are deja 0 sau 1 client

                for source_pos in range(1, len(source_route) - 1):
                    customer = source_route[source_pos]

                    reduced_source = source_route[:source_pos] + source_route[source_pos + 1:]

                    for insert_pos in range(1, len(target_route)):
                        expanded_target = (
                            target_route[:insert_pos]
                            + [customer]
                            + target_route[insert_pos:]
                        )

                        candidate_routes = [route[:] for route in best_routes]
                        candidate_routes[from_idx] = reduced_source
                        candidate_routes[to_idx] = expanded_target

                        candidate_total = total_cost(candidate_routes, distance_matrix)

                        if candidate_total + 1e-9 < best_total:
                            best_routes = candidate_routes
                            best_total = candidate_total
                            improved = True
                            break

                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return best_routes