from src.data.mtsplib_loader import load_mtsplib_instances
from src.strategy2.config import Strategy2Config
from src.strategy2.dataset_builder import (
    build_edge_dataset_for_instances,
    save_edge_dataset,
)


def main() -> None:
    print("Încarc instanțele mTSPLIB...")
    instances = load_mtsplib_instances()
    print(f"Au fost încărcate {len(instances)} instanțe.\n")

    config = Strategy2Config()

    print("Construiesc dataset-ul pentru Strategia 2...")
    dataset = build_edge_dataset_for_instances(
        instances=instances,
        config=config,
        random_seed=42
    )

    print("Dataset construit cu succes.")
    print(f"Shape X: {dataset.X.shape}")
    print(f"Shape y: {dataset.y.shape}")
    print(f"Număr exemple pozitive: {(dataset.y == 1).sum()}")
    print(f"Număr exemple negative: {(dataset.y == 0).sum()}")

    output_prefix = "data/processed/strategy2/mtsplib_edge_dataset"
    save_edge_dataset(dataset, output_prefix)

    print("\nFișiere salvate:")
    print(f"{output_prefix}_X.npy")
    print(f"{output_prefix}_y.npy")
    print(f"{output_prefix}_meta.csv")


if __name__ == "__main__":
    main()