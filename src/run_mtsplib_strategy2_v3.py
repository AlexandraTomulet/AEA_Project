from src.data.mtsplib_loader import load_mtsplib_instances
from src.evaluation.benchmark_runner import run_benchmark, save_benchmark_results
from src.evaluation.result_summary import summarize_benchmark_results
from src.strategy2.solver_v3 import Strategy2SolverV3, Strategy2SolverV3Config


def main() -> None:
    print("Încarc instanțele mTSPLIB...")
    instances = load_mtsplib_instances()
    print(f"Au fost încărcate {len(instances)} instanțe.\n")

    config = Strategy2SolverV3Config(
        num_candidate_neighbors=15,
        alpha_insertion_cost=1.0,
        beta_edge_score=1.5,
        node_penalty_lambda=0.5,
        hidden_dim=128,
        dropout=0.2,
        device="cpu"
    )

    solver = Strategy2SolverV3(
        model_dir="data/processed/strategy2_v3/model_artifacts_v3",
        config=config
    )

    print("Rulez Strategia 2 v3 pe benchmark-ul mTSPLIB...")
    results = run_benchmark(instances, solver)

    raw_output_path = "results/strategy2/mtsplib_results_v3.csv"
    summary_output_path = "results/strategy2/mtsplib_summary_v3.csv"

    save_benchmark_results(results, raw_output_path)
    summarize_benchmark_results(raw_output_path, summary_output_path)

    print("\nBenchmark finalizat.")
    print(f"Rezultate brute salvate în: {raw_output_path}")
    print(f"Rezultate agregate salvate în: {summary_output_path}")

    print("\nPrimele rezultate:")
    for result in results[:8]:
        print(
            f"{result.instance_name} | "
            f"valid={result.is_valid} | "
            f"total={result.total_cost:.2f} | "
            f"longest={result.longest_route_cost:.2f} | "
            f"time={result.runtime_seconds:.4f}s"
        )


if __name__ == "__main__":
    main()