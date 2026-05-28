from __future__ import annotations

from typing import Dict, Tuple

from src.models.mtsp_instance import MTSPInstance
from src.evaluation.metrics import total_cost, longest_route_cost
from src.evaluation.validator import validate_solution
from src.strategy1.allocator import sweep_balanced_assignment
from src.strategy1.inter_route import relocate_between_routes
from src.strategy1.local_search import two_opt
from src.strategy2.candidate_graph import build_candidate_graph
from src.strategy2.config import Strategy2Config
from src.strategy2.features import extract_edge_features
from src.strategy2.guided_route_builder import guided_nearest_insertion_route
from src.strategy2.predictor import EdgeScorePredictor
from src.strategy2.result import Strategy2Result


class Strategy2Solver:
    """
    Strategia 2:
    1. Alocare sweep balanced
    2. Candidate graph rar
    3. Edge scoring cu model neural
    4. Guided nearest insertion
    5. 2-opt per route
    6. Inter-route relocate
    7. 2-opt refinement
    """

    def __init__(self, model_dir: str, config: Strategy2Config) -> None:
        self.model_dir = model_dir
        self.config = config
        self.predictor: EdgeScorePredictor | None = None

    def _ensure_predictor(self, input_dim: int) -> None:
        """
        Inițializează predictorul o singură dată.
        """
        if self.predictor is None:
            self.predictor = EdgeScorePredictor(
                model_dir=self.model_dir,
                input_dim=input_dim,
                config=self.config
            )

    def solve(self, instance: MTSPInstance) -> Strategy2Result:
        """
        Rulează Strategia 2 pe o instanță mTSP și returnează rezultatul complet.
        """
        assignments = sweep_balanced_assignment(instance)

        candidate_graph = build_candidate_graph(
            instance=instance,
            num_neighbors=self.config.num_candidate_neighbors
        )

        X_edges, edge_index = extract_edge_features(
            instance=instance,
            candidate_graph=candidate_graph
        )

        self._ensure_predictor(input_dim=X_edges.shape[1])
        edge_probabilities = self.predictor.predict_proba(X_edges)

        edge_scores: Dict[Tuple[int, int], float] = {
            edge: float(score)
            for edge, score in zip(edge_index, edge_probabilities)
        }

        routes = []
        for assigned_nodes in assignments:
            route = guided_nearest_insertion_route(
                assigned_nodes=assigned_nodes,
                depot_index=instance.depot_index,
                distance_matrix=instance.distance_matrix,
                edge_scores=edge_scores,
                alpha_insertion_cost=self.config.alpha_insertion_cost,
                beta_edge_score=self.config.beta_edge_score
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

        return Strategy2Result(
            routes=routes,
            total_cost=solution_total_cost,
            longest_route_cost=solution_longest_route_cost,
            is_valid=is_valid,
            validation_errors=validation_errors
        )