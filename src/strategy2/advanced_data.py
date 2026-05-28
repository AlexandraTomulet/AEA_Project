from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from src.models.mtsp_instance import MTSPInstance
from src.utils.distance import build_distance_matrix


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    """
    Config pentru generarea instanțelor sintetice de mTSP.
    """
    train_instances_per_setting: int = 80
    val_instances_per_setting: int = 20
    test_instances_per_setting: int = 20

    customer_sizes: tuple[int, ...] = (30, 50, 75, 100, 150)
    num_salesmen_values: tuple[int, ...] = (2, 3, 5, 7)

    coordinate_min: float = 0.0
    coordinate_max: float = 100.0


def _create_random_mtsp_instance(
    name: str,
    num_customers: int,
    num_salesmen: int,
    rng: np.random.Generator,
    coordinate_min: float,
    coordinate_max: float
) -> MTSPInstance:
    """
    Creează o instanță euclidiană random pentru mTSP.
    Nodul 0 este depozitul.
    """
    num_nodes = num_customers + 1
    coordinates = rng.uniform(
        low=coordinate_min,
        high=coordinate_max,
        size=(num_nodes, 2)
    ).astype(float)

    distance_matrix = build_distance_matrix(coordinates)

    return MTSPInstance(
        name=name,
        num_nodes=num_nodes,
        num_salesmen=num_salesmen,
        depot_index=0,
        coordinates=coordinates,
        distance_matrix=distance_matrix,
        best_known_cost=None
    )


def build_synthetic_instance_split(
    config: SyntheticDatasetConfig,
    random_seed: int = 42
) -> tuple[List[MTSPInstance], List[MTSPInstance], List[MTSPInstance]]:
    """
    Generează trei liste de instanțe: train, validation, test.
    Split-ul este făcut la nivel de instanță, nu la nivel de muchie.
    """
    rng = np.random.default_rng(random_seed)

    train_instances: List[MTSPInstance] = []
    val_instances: List[MTSPInstance] = []
    test_instances: List[MTSPInstance] = []

    for num_customers in config.customer_sizes:
        for num_salesmen in config.num_salesmen_values:
            for idx in range(config.train_instances_per_setting):
                instance = _create_random_mtsp_instance(
                    name=f"train_n{num_customers}_m{num_salesmen}_{idx}",
                    num_customers=num_customers,
                    num_salesmen=num_salesmen,
                    rng=rng,
                    coordinate_min=config.coordinate_min,
                    coordinate_max=config.coordinate_max
                )
                train_instances.append(instance)

            for idx in range(config.val_instances_per_setting):
                instance = _create_random_mtsp_instance(
                    name=f"val_n{num_customers}_m{num_salesmen}_{idx}",
                    num_customers=num_customers,
                    num_salesmen=num_salesmen,
                    rng=rng,
                    coordinate_min=config.coordinate_min,
                    coordinate_max=config.coordinate_max
                )
                val_instances.append(instance)

            for idx in range(config.test_instances_per_setting):
                instance = _create_random_mtsp_instance(
                    name=f"test_n{num_customers}_m{num_salesmen}_{idx}",
                    num_customers=num_customers,
                    num_salesmen=num_salesmen,
                    rng=rng,
                    coordinate_min=config.coordinate_min,
                    coordinate_max=config.coordinate_max
                )
                test_instances.append(instance)

    return train_instances, val_instances, test_instances