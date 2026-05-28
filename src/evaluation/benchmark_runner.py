from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import List
import pandas as pd

from src.models.mtsp_instance import MTSPInstance


@dataclass
class BenchmarkResult:
    instance_name: str
    num_nodes: int
    num_salesmen: int
    is_valid: bool
    total_cost: float
    longest_route_cost: float
    runtime_seconds: float
    error_message: str


def run_benchmark(instances: List[MTSPInstance], solver) -> List[BenchmarkResult]:
    """
    Rulează solverul pe o listă de instanțe și întoarce rezultatele benchmark-ului.
    Solverul trebuie să returneze un obiect rezultat cu:
    - is_valid
    - total_cost
    - longest_route_cost
    - validation_errors
    """
    results: List[BenchmarkResult] = []

    for instance in instances:
        start_time = perf_counter()
        solution = solver.solve(instance)
        end_time = perf_counter()

        result = BenchmarkResult(
            instance_name=instance.name,
            num_nodes=instance.num_nodes,
            num_salesmen=instance.num_salesmen,
            is_valid=solution.is_valid,
            total_cost=solution.total_cost,
            longest_route_cost=solution.longest_route_cost,
            runtime_seconds=end_time - start_time,
            error_message=" | ".join(solution.validation_errors)
        )
        results.append(result)

    return results


def save_benchmark_results(results: List[BenchmarkResult], output_path: str) -> None:
    """
    Salvează rezultatele benchmark-ului într-un fișier CSV.
    """
    df = pd.DataFrame([asdict(result) for result in results])
    df.to_csv(output_path, index=False)