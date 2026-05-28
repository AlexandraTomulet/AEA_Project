from src.data.mtsplib_loader import load_single_mtsplib_instance
from src.strategy2.config import Strategy2Config
from src.strategy2.candidate_graph import build_candidate_graph
from src.strategy2.features import extract_edge_features


def main() -> None:
    config = Strategy2Config()
    instance = load_single_mtsplib_instance("eil51", 3)

    candidate_graph = build_candidate_graph(
        instance=instance,
        num_neighbors=config.num_candidate_neighbors
    )

    X, edge_index = extract_edge_features(
        instance=instance,
        candidate_graph=candidate_graph
    )

    print("Smoke test Strategy 2")
    print(f"Instanță: {instance.name}")
    print(f"Număr noduri: {instance.num_nodes}")
    print(f"Număr muchii candidate: {len(candidate_graph.edges)}")
    print(f"Shape features: {X.shape}")
    print(f"Primele 5 muchii: {edge_index[:5]}")
    print(f"Primul vector de features: {X[0].tolist()}")


if __name__ == "__main__":
    main()