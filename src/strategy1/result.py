from dataclasses import dataclass
from typing import List


@dataclass
class Strategy1Result:
    """
    Rezultatul complet al rulării Strategiei 1.
    """
    routes: List[List[int]]
    total_cost: float
    longest_route_cost: float
    is_valid: bool
    validation_errors: List[str]