from src.data.mtsplib_loader import load_single_mtsplib_instance


def main() -> None:
    instance = load_single_mtsplib_instance("eil51", 2)

    print("Instanță încărcată cu succes.")
    print(f"Nume: {instance.name}")
    print(f"Număr noduri: {instance.num_nodes}")
    print(f"Număr salesmeni: {instance.num_salesmen}")
    print(f"Depot index: {instance.depot_index}")
    print(f"Shape coordonate: {instance.coordinates.shape if instance.coordinates is not None else None}")
    print(f"Shape matrice distanțe: {instance.distance_matrix.shape}")


if __name__ == "__main__":
    main()