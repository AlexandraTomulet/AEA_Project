from typing import List, Tuple
import math
from src.models.mtsp_instance import MTSPInstance


def _polar_angle(cx: float, cy: float, x: float, y: float) -> float:
    """
    Calculează unghiul polar al punctului (x, y) față de centrul (cx, cy).
    """
    return math.atan2(y - cy, x - cx)


def sweep_balanced_assignment(instance: MTSPInstance) -> List[List[int]]:
    """
    Împarte nodurile non-depozit între salesmeni folosind o euristică de tip sweep.
    Nodurile sunt sortate după unghiul polar față de depozit și apoi distribuite
    cât mai echilibrat între salesmeni.

    Este mult mai relevantă pentru mTSP decât round-robin, deoarece păstrează
    o structură geometrică naturală a rutelor.
    """
    if instance.coordinates is None:
        raise ValueError("Sweep assignment necesită coordonate disponibile.")

    depot = instance.depot_index
    depot_x, depot_y = instance.coordinates[depot]

    customers_with_angles: List[Tuple[int, float]] = []
    for node in range(instance.num_nodes):
        if node == depot:
            continue
        x, y = instance.coordinates[node]
        angle = _polar_angle(depot_x, depot_y, x, y)
        customers_with_angles.append((node, angle))

    customers_with_angles.sort(key=lambda item: item[1])

    assignments = [[] for _ in range(instance.num_salesmen)]
    target_size = math.ceil((instance.num_nodes - 1) / instance.num_salesmen)

    salesman_idx = 0
    for node, _ in customers_with_angles:
        assignments[salesman_idx].append(node)

        if len(assignments[salesman_idx]) >= target_size and salesman_idx < instance.num_salesmen - 1:
            salesman_idx += 1

    return assignments