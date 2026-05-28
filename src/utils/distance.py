import numpy as np


def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    Calculează distanța euclidiană dintre două puncte 2D.
    """
    return float(np.linalg.norm(point_a - point_b))


def build_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Construiește matricea completă de distanțe pentru toate nodurile.
    """
    num_nodes = coordinates.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(coordinates[i], coordinates[j])

    return distance_matrix