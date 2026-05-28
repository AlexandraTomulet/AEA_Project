from typing import List
import matplotlib.pyplot as plt
import numpy as np
from src.models.mtsp_instance import MTSPInstance


def plot_routes(
    instance: MTSPInstance,
    routes: List[List[int]],
    title: str = "mTSP Solution",
    save_path: str | None = None
) -> None:
    """
    Desenează rutele pentru o instanță mTSP.
    """
    if instance.coordinates is None:
        raise ValueError("Nu există coordonate pentru vizualizare.")

    coords = instance.coordinates
    depot = instance.depot_index

    plt.figure(figsize=(8, 8))

    # toate nodurile
    plt.scatter(coords[:, 0], coords[:, 1], s=40)
    plt.scatter(coords[depot, 0], coords[depot, 1], s=120, marker="s")

    for idx, route in enumerate(routes, start=1):
        route_coords = coords[route]
        plt.plot(route_coords[:, 0], route_coords[:, 1], marker="o", label=f"Salesman {idx}")

    # etichete pentru noduri
    for node_idx, (x, y) in enumerate(coords):
        plt.text(x + 0.8, y + 0.8, str(node_idx), fontsize=9)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()