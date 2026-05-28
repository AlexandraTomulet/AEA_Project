import numpy as np
from src.models.mtsp_instance import MTSPInstance
from src.utils.distance import build_distance_matrix


def create_sample_instance() -> MTSPInstance:
    """
    Creează o instanță mică artificială pentru testare.
    Nodul 0 este depozitul.
    """
    coordinates = np.array([
        [50.0, 50.0],  # depozit
        [20.0, 60.0],
        [25.0, 30.0],
        [60.0, 20.0],
        [80.0, 40.0],
        [70.0, 70.0],
        [40.0, 80.0],
        [15.0, 45.0],
    ], dtype=float)

    distance_matrix = build_distance_matrix(coordinates)

    return MTSPInstance(
        name="sample_mtsp",
        num_nodes=len(coordinates),
        num_salesmen=2,
        depot_index=0,
        coordinates=coordinates,
        distance_matrix=distance_matrix,
        best_known_cost=None,
    )