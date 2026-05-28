from src.data.mtsplib_loader import load_mtsplib_instances
from src.evaluation.benchmark_runner import run_benchmark, save_benchmark_results
from src.evaluation.result_summary import summarize_benchmark_results
from src.strategy2.config import Strategy2Config
from src.strategy2.solver import Strategy2Solver


def main() -> None:
    print("Încarc instanțele mTSPLIB...")
    instances = load_mtsplib_instances()
    print(f"Au fost încărcate {len(instances)} instanțe.\n")

    config = Strategy2Config()
    solver = Strategy2Solver(
        model_dir="data/processed/strategy2/model_artifacts",
        config=config
    )

    print("Rulez Strategia 2 pe benchmark-ul mTSPLIB...")
    results = run_benchmark(instances, solver)

    raw_output_path = "results/strategy2/mtsplib_results.csv"
    summary_output_path = "results/strategy2/mtsplib_summary.csv"

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