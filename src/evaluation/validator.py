from typing import List
from src.models.mtsp_instance import MTSPInstance


def validate_solution(instance: MTSPInstance, routes: List[List[int]]) -> tuple[bool, list[str]]:
    """
    Verifică dacă soluția mTSP este validă.
    Returnează:
    - bool: dacă soluția este validă
    - list[str]: lista problemelor găsite
    """
    errors: list[str] = []
    depot = instance.depot_index

    if len(routes) != instance.num_salesmen:
        errors.append(
            f"Numărul de rute ({len(routes)}) diferă de numărul de salesmeni ({instance.num_salesmen})."
        )

    visited_customers: list[int] = []

    for route_idx, route in enumerate(routes, start=1):
        if len(route) < 2:
            errors.append(f"Ruta {route_idx} este prea scurtă.")
            continue

        if route[0] != depot:
            errors.append(f"Ruta {route_idx} nu începe în depozit.")
        if route[-1] != depot:
            errors.append(f"Ruta {route_idx} nu se termină în depozit.")

        internal_nodes = route[1:-1]

        for node in internal_nodes:
            if node == depot:
                errors.append(f"Ruta {route_idx} conține depozitul în interiorul traseului.")

        visited_customers.extend(internal_nodes)

    expected_customers = {node for node in range(instance.num_nodes) if node != depot}
    visited_set = set(visited_customers)

    missing = expected_customers - visited_set
    duplicated = [node for node in visited_set if visited_customers.count(node) > 1]
    invalid_nodes = [node for node in visited_customers if node < 0 or node >= instance.num_nodes]

    if missing:
        errors.append(f"Lipsesc clienți din soluție: {sorted(missing)}")
    if duplicated:
        errors.append(f"Există clienți vizitați de mai multe ori: {sorted(set(duplicated))}")
    if invalid_nodes:
        errors.append(f"Există noduri invalide în soluție: {sorted(set(invalid_nodes))}")

    is_valid = len(errors) == 0
    return is_valid, errors