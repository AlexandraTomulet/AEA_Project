from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_final_comparison_table(
    strategy1_csv_path: str,
    strategy2_csv_path: str,
    output_csv_path: str
) -> pd.DataFrame:
    """
    Construiește tabelul comparativ final între Strategia 1 și Strategia 2.

    Se compară:
    - total_cost
    - longest_route_cost
    - runtime_seconds
    - validitatea soluției

    Se calculează și diferențe absolute și procentuale.
    """
    s1_df = pd.read_csv(strategy1_csv_path)
    s2_df = pd.read_csv(strategy2_csv_path)

    s1_df = s1_df.rename(columns={
        "is_valid": "s1_is_valid",
        "total_cost": "s1_total_cost",
        "longest_route_cost": "s1_longest_route_cost",
        "runtime_seconds": "s1_runtime_seconds",
        "error_message": "s1_error_message",
    })

    s2_df = s2_df.rename(columns={
        "is_valid": "s2_is_valid",
        "total_cost": "s2_total_cost",
        "longest_route_cost": "s2_longest_route_cost",
        "runtime_seconds": "s2_runtime_seconds",
        "error_message": "s2_error_message",
    })

    # Păstrăm coloanele de identificare doar dintr-un singur tabel
    comparison_df = pd.merge(
        s1_df,
        s2_df,
        on=["instance_name", "num_nodes", "num_salesmen"],
        how="inner"
    )

    # Diferențe absolute
    comparison_df["delta_total_cost"] = comparison_df["s2_total_cost"] - comparison_df["s1_total_cost"]
    comparison_df["delta_longest_route_cost"] = comparison_df["s2_longest_route_cost"] - comparison_df["s1_longest_route_cost"]
    comparison_df["delta_runtime_seconds"] = comparison_df["s2_runtime_seconds"] - comparison_df["s1_runtime_seconds"]

    # Diferențe procentuale
    comparison_df["delta_total_cost_percent"] = (
        comparison_df["delta_total_cost"] / comparison_df["s1_total_cost"] * 100.0
    )
    comparison_df["delta_longest_route_percent"] = (
        comparison_df["delta_longest_route_cost"] / comparison_df["s1_longest_route_cost"] * 100.0
    )

    # Cine este mai bun la cost total
    comparison_df["winner_total_cost"] = comparison_df.apply(
        lambda row: "Strategy1"
        if row["s1_total_cost"] < row["s2_total_cost"]
        else ("Strategy2" if row["s2_total_cost"] < row["s1_total_cost"] else "Tie"),
        axis=1
    )

    # Cine este mai bun la longest route
    comparison_df["winner_longest_route"] = comparison_df.apply(
        lambda row: "Strategy1"
        if row["s1_longest_route_cost"] < row["s2_longest_route_cost"]
        else ("Strategy2" if row["s2_longest_route_cost"] < row["s1_longest_route_cost"] else "Tie"),
        axis=1
    )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    return comparison_df


def build_final_summary(
    comparison_df: pd.DataFrame,
    output_csv_path: str
) -> pd.DataFrame:
    """
    Construiește un rezumat agregat al comparației finale.
    """
    summary = pd.DataFrame([{
        "num_instances": len(comparison_df),

        "strategy1_avg_total_cost": comparison_df["s1_total_cost"].mean(),
        "strategy2_avg_total_cost": comparison_df["s2_total_cost"].mean(),

        "strategy1_avg_longest_route_cost": comparison_df["s1_longest_route_cost"].mean(),
        "strategy2_avg_longest_route_cost": comparison_df["s2_longest_route_cost"].mean(),

        "strategy1_avg_runtime_seconds": comparison_df["s1_runtime_seconds"].mean(),
        "strategy2_avg_runtime_seconds": comparison_df["s2_runtime_seconds"].mean(),

        "avg_delta_total_cost": comparison_df["delta_total_cost"].mean(),
        "avg_delta_total_cost_percent": comparison_df["delta_total_cost_percent"].mean(),

        "avg_delta_longest_route_cost": comparison_df["delta_longest_route_cost"].mean(),
        "avg_delta_longest_route_percent": comparison_df["delta_longest_route_percent"].mean(),

        "strategy1_better_total_cost_count": (comparison_df["winner_total_cost"] == "Strategy1").sum(),
        "strategy2_better_total_cost_count": (comparison_df["winner_total_cost"] == "Strategy2").sum(),
        "tie_total_cost_count": (comparison_df["winner_total_cost"] == "Tie").sum(),

        "strategy1_better_longest_route_count": (comparison_df["winner_longest_route"] == "Strategy1").sum(),
        "strategy2_better_longest_route_count": (comparison_df["winner_longest_route"] == "Strategy2").sum(),
        "tie_longest_route_count": (comparison_df["winner_longest_route"] == "Tie").sum(),

        "all_strategy1_valid": bool(comparison_df["s1_is_valid"].all()),
        "all_strategy2_valid": bool(comparison_df["s2_is_valid"].all()),
    }])

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    return summary