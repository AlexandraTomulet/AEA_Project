from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy2Config:
    """
    Configurația principală pentru Strategia 2.
    """
    num_candidate_neighbors: int = 10
    hidden_dim_1: int = 128
    hidden_dim_2: int = 64
    batch_size: int = 256
    num_epochs: int = 30
    learning_rate: float = 1e-3

    # ponderi pentru ghidarea construcției de rută
    alpha_insertion_cost: float = 1.0
    beta_edge_score: float = 1.0

    # raport negativ / pozitiv în dataset
    negative_sampling_ratio: int = 2