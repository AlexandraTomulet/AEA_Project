from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi
from typing import List, Tuple
import numpy as np

from src.models.mtsp_instance import MTSPInstance
from src.evaluation.metrics import total_cost
from src.evaluation.validator import validate_solution
from src.strategy1.inter_route import relocate_between_routes
from src.strategy1.local_search import two_opt
from src.strategy1.result import Strategy1Result
from src.strategy1.route_builder import nearest_insertion_route


def _polar_angle(cx: float, cy: float, x: float, y: float) -> float:
    return atan2(y - cy, x - cx)


def _rotated_sweep_assignment(
    instance: MTSPInstance,
    angle_offset: float
) -> List[List[int]]:
    """
    Sweep assignment cu rotație de unghi, pentru a obține soluții diferite.
    """
    if instance.coordinates is None:
        raise ValueError("Teacher solver necesită coordonate.")

    depot = instance.depot_index
    depot_x, depot_y = instance.coordinates[depot]

    customers_with_angles: List[Tuple[int, float]] = []
    for node in range(instance.num_nodes):
        if node == depot:
            continue

        x, y = instance.coordinates[node]
        angle = _polar_angle(depot_x, depot_y, x, y) + angle_offset

        while angle < -pi:
            angle += 2 * pi
        while angle > pi:
            angle -= 2 * pi

        customers_with_angles.append((node, angle))

    customers_with_angles.sort(key=lambda item: item[1])

    assignments = [[] for _ in range(instance.num_salesmen)]
    target_size = int(np.ceil((instance.num_nodes - 1) / instance.num_salesmen))

    salesman_idx = 0
    for node, _ in customers_with_angles:
        assignments[salesman_idx].append(node)
        if len(assignments[salesman_idx]) >= target_size and salesman_idx < instance.num_salesmen - 1:
            salesman_idx += 1

    return assignments


@dataclass
class TeacherConfig:
    """
    Config pentru teacher solver.
    """
    num_restarts: int = 8


class MultiRestartTeacherSolver:
    """
    Teacher solver bazat pe mai multe rulări ale unei euristici similare Strategiei 1,
    dar cu rotații diferite ale sweep-ului.
    """

    def __init__(self, config: TeacherConfig | None = None) -> None:
        self.config = config or TeacherConfig()

    def solve(self, instance: MTSPInstance) -> Strategy1Result:
        if self.config.num_restarts < 1:
            raise ValueError("num_restarts trebuie să fie >= 1.")

        best_result: Strategy1Result | None = None

        for restart_idx in range(self.config.num_restarts):
            angle_offset = (2.0 * pi * restart_idx) / self.config.num_restarts

            assignments = _rotated_sweep_assignment(
                instance=instance,
                angle_offset=angle_offset
            )

            routes = []
            for assigned_nodes in assignments:
                route = nearest_insertion_route(
                    assigned_nodes=assigned_nodes,
                    depot_index=instance.depot_index,
                    distance_matrix=instance.distance_matrix
                )
                route = two_opt(route, instance.distance_matrix)
                routes.append(route)

            routes = relocate_between_routes(
                routes=routes,
                distance_matrix=instance.distance_matrix,
                depot_index=instance.depot_index
            )

            routes = [two_opt(route, instance.distance_matrix) for route in routes]

            is_valid, validation_errors = validate_solution(instance, routes)

            if is_valid:
                solution_total_cost = total_cost(routes, instance.distance_matrix)
                from src.evaluation.metrics import longest_route_cost
                solution_longest_route_cost = longest_route_cost(routes, instance.distance_matrix)
            else:
                solution_total_cost = -1.0
                solution_longest_route_cost = -1.0

            result = Strategy1Result(
                routes=routes,
                total_cost=solution_total_cost,
                longest_route_cost=solution_longest_route_cost,
                is_valid=is_valid,
                validation_errors=validation_errors
            )

            if not result.is_valid:
                continue

            if best_result is None or result.total_cost < best_result.total_cost:
                best_result = result

        if best_result is None:
            raise ValueError(f"Teacher solver nu a produs nicio soluție validă pentru {instance.name}.")

        return best_result