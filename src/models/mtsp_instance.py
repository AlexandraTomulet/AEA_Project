from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MTSPInstance:
    """
    Reprezentare internă unificată pentru o instanță mTSP.
    Toți solverii vor lucra cu acest obiect, indiferent de sursa benchmark-ului.
    """
    name: str
    num_nodes: int
    num_salesmen: int
    depot_index: int
    coordinates: Optional[np.ndarray]
    distance_matrix: np.ndarray
    best_known_cost: Optional[float] = None