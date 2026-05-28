from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_total_cost_comparison(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """
    Grafic comparativ pentru costul total.
    """
    output_path = _ensure_output_dir(output_dir)

    labels = comparison_df["instance_name"].tolist()
    s1_values = comparison_df["s1_total_cost"].tolist()
    s2_values = comparison_df["s2_total_cost"].tolist()

    x = range(len(labels))
    width = 0.4

    plt.figure(figsize=(16, 6))
    plt.bar([i - width / 2 for i in x], s1_values, width=width, label="Strategy 1")
    plt.bar([i + width / 2 for i in x], s2_values, width=width, label="Strategy 2 v3")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Total Cost")
    plt.title("Total Cost Comparison: Strategy 1 vs Strategy 2 v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "total_cost_comparison.png")
    plt.close()


def plot_longest_route_comparison(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """
    Grafic comparativ pentru longest route cost.
    """
    output_path = _ensure_output_dir(output_dir)

    labels = comparison_df["instance_name"].tolist()
    s1_values = comparison_df["s1_longest_route_cost"].tolist()
    s2_values = comparison_df["s2_longest_route_cost"].tolist()

    x = range(len(labels))
    width = 0.4

    plt.figure(figsize=(16, 6))
    plt.bar([i - width / 2 for i in x], s1_values, width=width, label="Strategy 1")
    plt.bar([i + width / 2 for i in x], s2_values, width=width, label="Strategy 2 v3")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Longest Route Cost")
    plt.title("Longest Route Comparison: Strategy 1 vs Strategy 2 v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "longest_route_comparison.png")
    plt.close()


def plot_runtime_comparison(comparison_df: pd.DataFrame, output_dir: str) -> None:
    """
    Grafic comparativ pentru runtime.
    """
    output_path = _ensure_output_dir(output_dir)

    labels = comparison_df["instance_name"].tolist()
    s1_values = comparison_df["s1_runtime_seconds"].tolist()
    s2_values = comparison_df["s2_runtime_seconds"].tolist()

    x = range(len(labels))
    width = 0.4

    plt.figure(figsize=(16, 6))
    plt.bar([i - width / 2 for i in x], s1_values, width=width, label="Strategy 1")
    plt.bar([i + width / 2 for i in x], s2_values, width=width, label="Strategy 2 v3")

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison: Strategy 1 vs Strategy 2 v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "runtime_comparison.png")
    plt.close()