from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from src.models.mtsp_instance import MTSPInstance


@dataclass
class CandidateEdge:
    """
    Reprezintă o muchie candidat orientată i -> j.
    """
    source: int
    target: int
    rank: int
    distance: float


@dataclass
class CandidateGraph:
    """
    Graf rar orientat de muchii candidate.
    """
    edges: List[CandidateEdge]
    adjacency: Dict[int, List[CandidateEdge]]


def build_candidate_graph(
    instance: MTSPInstance,
    num_neighbors: int
) -> CandidateGraph:
    """
    Construiește un graf rar orientat pentru instanța dată.
    Pentru fiecare nod i, se păstrează top-k vecini cei mai apropiați.
    """
    n = instance.num_nodes
    distance_matrix = instance.distance_matrix

    edges: List[CandidateEdge] = []
    adjacency: Dict[int, List[CandidateEdge]] = {node: [] for node in range(n)}

    for source in range(n):
        candidate_distances: List[Tuple[int, float]] = []

        for target in range(n):
            if source == target:
                continue
            candidate_distances.append((target, distance_matrix[source, target]))

        candidate_distances.sort(key=lambda item: item[1])
        nearest_neighbors = candidate_distances[:num_neighbors]

        for rank, (target, distance) in enumerate(nearest_neighbors, start=1):
            edge = CandidateEdge(
                source=source,
                target=target,
                rank=rank,
                distance=float(distance)
            )
            edges.append(edge)
            adjacency[source].append(edge)

    return CandidateGraph(edges=edges, adjacency=adjacency)