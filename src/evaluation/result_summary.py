import pandas as pd


def summarize_benchmark_results(input_csv_path: str, output_csv_path: str) -> None:
    """
    Creează un sumar agregat al rezultatelor benchmark-ului,
    grupat după numărul de noduri și numărul de salesmeni.
    """
    df = pd.read_csv(input_csv_path)

    summary = (
        df.groupby(["num_nodes", "num_salesmen"], as_index=False)
        .agg(
            num_instances=("instance_name", "count"),
            valid_count=("is_valid", "sum"),
            avg_total_cost=("total_cost", "mean"),
            avg_longest_route_cost=("longest_route_cost", "mean"),
            avg_runtime_seconds=("runtime_seconds", "mean"),
            min_total_cost=("total_cost", "min"),
            max_total_cost=("total_cost", "max"),
        )
    )

    summary.to_csv(output_csv_path, index=False)