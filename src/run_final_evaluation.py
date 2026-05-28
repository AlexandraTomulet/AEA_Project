from __future__ import annotations

from src.evaluation.final_comparison import (
    build_final_comparison_table,
    build_final_summary,
)
from src.evaluation.plot_results import (
    plot_total_cost_comparison,
    plot_longest_route_comparison,
    plot_runtime_comparison,
)


def main() -> None:
    strategy1_csv = "results/strategy1/mtsplib_results.csv"
    strategy2_csv = "results/strategy2/mtsplib_results_v3.csv"

    comparison_csv = "results/final_evaluation/final_comparison.csv"
    summary_csv = "results/final_evaluation/final_summary.csv"
    plots_dir = "results/final_evaluation"

    print("Construiesc tabelul comparativ final...")
    comparison_df = build_final_comparison_table(
        strategy1_csv_path=strategy1_csv,
        strategy2_csv_path=strategy2_csv,
        output_csv_path=comparison_csv
    )

    print("Construiesc rezumatul agregat...")
    summary_df = build_final_summary(
        comparison_df=comparison_df,
        output_csv_path=summary_csv
    )

    print("Generez graficele...")
    plot_total_cost_comparison(comparison_df, plots_dir)
    plot_longest_route_comparison(comparison_df, plots_dir)
    plot_runtime_comparison(comparison_df, plots_dir)

    print("\nEvaluarea finală a fost generată cu succes.")
    print(f"Tabel comparativ: {comparison_csv}")
    print(f"Rezumat agregat: {summary_csv}")
    print(f"Grafice salvate în: {plots_dir}")

    print("\nRezumat rapid:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()