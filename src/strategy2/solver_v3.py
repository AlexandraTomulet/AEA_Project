from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.models.mtsp_instance import MTSPInstance
from src.evaluation.metrics import total_cost, longest_route_cost
from src.evaluation.validator import validate_solution
from src.strategy1.allocator import sweep_balanced_assignment
from src.strategy1.inter_route import relocate_between_routes
from src.strategy1.local_search import two_opt
from src.strategy2.candidate_graph import build_candidate_graph
from src.strategy2.features import extract_edge_features
from src.strategy2.guided_route_builder import guided_nearest_insertion_route
from src.strategy2.predictor_v3 import EdgeNodePredictorV3
from src.strategy2.result import Strategy2Result


@dataclass(frozen=True)
class Strategy2SolverV3Config:
    """
    Config pentru solverul v3.
    """
    num_candidate_neighbors: int = 15
    alpha_insertion_cost: float = 1.0
    beta_edge_score: float = 1.5
    node_penalty_lambda: float = 0.5
    hidden_dim: int = 128
    dropout: float = 0.2
    device: str = "cpu"


class Strategy2SolverV3:
    """
    Strategia 2 v3:
    - edge scores
    - node penalties
    - transformed distances
    - guided heuristic
    """

    def __init__(self, model_dir: str, config: Strategy2SolverV3Config) -> None:
        self.model_dir = model_dir
        self.config = config
        self.predictor: EdgeNodePredictorV3 | None = None

    def _ensure_predictor(self, edge_input_dim: int) -> None:
        if self.predictor is None:
            self.predictor = EdgeNodePredictorV3(
                model_dir=self.model_dir,
                edge_input_dim=edge_input_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
                device=self.config.device
            )

    def _build_transformed_distance_matrix(
        self,
        original_distance_matrix: np.ndarray,
        node_penalties: np.ndarray
    ) -> np.ndarray:
        """
        Construiește matricea de distanțe transformată:
        c_ij = d_ij + lambda * (pi_i + pi_j)
        """
        n = original_distance_matrix.shape[0]
        transformed = original_distance_matrix.copy().astype(np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    transformed[i, j] = 0.0
                else:
                    transformed[i, j] = (
                        original_distance_matrix[i, j]
                        + self.config.node_penalty_lambda * (node_penalties[i] + node_penalties[j])
                    )

        return transformed

    def solve(self, instance: MTSPInstance) -> Strategy2Result:
        assignments = sweep_balanced_assignment(instance)

        candidate_graph = build_candidate_graph(
            instance=instance,
            num_neighbors=self.config.num_candidate_neighbors
        )

        edge_features, edge_index = extract_edge_features(
            instance=instance,
            candidate_graph=candidate_graph
        )

        node_features = instance.coordinates.astype(np.float32)
        edge_sources = np.array([src for src, _ in edge_index], dtype=np.int64)
        edge_targets = np.array([tgt for _, tgt in edge_index], dtype=np.int64)

        self._ensure_predictor(edge_input_dim=edge_features.shape[1])

        edge_probabilities, node_penalties = self.predictor.predict(
            node_features=node_features,
            edge_features=edge_features,
            edge_sources=edge_sources,
            edge_targets=edge_targets
        )

        edge_scores: Dict[Tuple[int, int], float] = {
            edge: float(score)
            for edge, score in zip(edge_index, edge_probabilities)
        }

        transformed_distance_matrix = self._build_transformed_distance_matrix(
            original_distance_matrix=instance.distance_matrix,
            node_penalties=node_penalties
        )

        routes = []
        for assigned_nodes in assignments:
            route = guided_nearest_insertion_route(
                assigned_nodes=assigned_nodes,
                depot_index=instance.depot_index,
                distance_matrix=transformed_distance_matrix,
                edge_scores=edge_scores,
                alpha_insertion_cost=self.config.alpha_insertion_cost,
                beta_edge_score=self.config.beta_edge_score
            )

            # rafinarea o facem pe costul real, nu pe cel transformat
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