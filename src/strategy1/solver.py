from src.models.mtsp_instance import MTSPInstance
from src.strategy1.allocator import sweep_balanced_assignment
from src.strategy1.route_builder import nearest_insertion_route
from src.strategy1.local_search import two_opt
from src.strategy1.inter_route import relocate_between_routes
from src.strategy1.result import Strategy1Result
from src.evaluation.metrics import total_cost, longest_route_cost
from src.evaluation.validator import validate_solution


class Strategy1Solver:
    """
    Strategia 1:
    1. Alocare geometrică de tip sweep
    2. Construcție inițială cu nearest insertion
    3. Optimizare locală 2-opt pe fiecare rută
    4. Îmbunătățire globală inter-rute prin relocate
    5. Rafinare finală cu 2-opt
    """

    def solve(self, instance: MTSPInstance) -> Strategy1Result:
        """
        Rulează Strategia 1 pe o instanță mTSP și returnează
        rezultatul complet al solverului.
        """
        assignments = sweep_balanced_assignment(instance)

        routes = []
        for assigned_nodes in assignments:
            route = nearest_insertion_route(
                assigned_nodes=assigned_nodes,
                depot_index=instance.depot_index,
                distance_matrix=instance.distance_matrix
            )
            improved_route = two_opt(route, instance.distance_matrix)
            routes.append(improved_route)

        routes = relocate_between_routes(
            routes=routes,
            distance_matrix=instance.distance_matrix,
            depot_index=instance.depot_index
        )

        routes = [two_opt(route, instance.distance_matrix) for route in routes]

        is_valid, validation_errors = validate_solution(instance, routes)

        if is_valid:
            solution_total_cost = total_cost(routes, instance.distance_matrix)
            solution_longest_route_cost = longest_route_cost(routes, instance.distance_matrix)
        else:
            solution_total_cost = -1.0
            solution_longest_route_cost = -1.0

        return Strategy1Result(
            routes=routes,
            total_cost=solution_total_cost,
            longest_route_cost=solution_longest_route_cost,
            is_valid=is_valid,
            validation_errors=validation_errors
        )