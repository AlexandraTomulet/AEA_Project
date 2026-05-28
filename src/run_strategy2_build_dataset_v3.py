from pathlib import Path

from src.strategy2.advanced_data import (
    SyntheticDatasetConfig,
    build_synthetic_instance_split,
)
from src.strategy2.config import Strategy2Config
from src.strategy2.dataset_builder_v3 import (
    build_edge_node_dataset_v3,
    save_edge_node_dataset_v3,
)
from src.strategy2.teacher_solver import MultiRestartTeacherSolver, TeacherConfig


def main() -> None:
    output_dir = Path("data/processed/strategy2_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_config = SyntheticDatasetConfig(
        train_instances_per_setting=8,
        val_instances_per_setting=3,
        test_instances_per_setting=3,
        customer_sizes=(30, 50, 75, 100),
        num_salesmen_values=(2, 3, 5),
    )

    strategy2_config = Strategy2Config(
        num_candidate_neighbors=15,
        num_epochs=50,
        batch_size=512,
        negative_sampling_ratio=2,
    )

    teacher_solver = MultiRestartTeacherSolver(
        config=TeacherConfig(num_restarts=3)
    )

    print("Generez split-ul de instanțe...")
    train_instances, val_instances, test_instances = build_synthetic_instance_split(
        config=synthetic_config,
        random_seed=42
    )

    print(f"Train instances: {len(train_instances)}")
    print(f"Val instances: {len(val_instances)}")
    print(f"Test instances: {len(test_instances)}\n")

    print("Construiesc dataset-ul TRAIN v3...")
    train_dataset = build_edge_node_dataset_v3(
        instances=train_instances,
        strategy2_config=strategy2_config,
        teacher_solver=teacher_solver,
        random_seed=42
    )
    save_edge_node_dataset_v3(train_dataset, str(output_dir / "train_dataset_v3"))

    print("\nConstruiesc dataset-ul VAL v3...")
    val_dataset = build_edge_node_dataset_v3(
        instances=val_instances,
        strategy2_config=strategy2_config,
        teacher_solver=teacher_solver,
        random_seed=43
    )
    save_edge_node_dataset_v3(val_dataset, str(output_dir / "val_dataset_v3"))

    print("\nConstruiesc dataset-ul TEST v3...")
    test_dataset = build_edge_node_dataset_v3(
        instances=test_instances,
        strategy2_config=strategy2_config,
        teacher_solver=teacher_solver,
        random_seed=44
    )
    save_edge_node_dataset_v3(test_dataset, str(output_dir / "test_dataset_v3"))

    print("\nDataset-urile v3 au fost salvate cu succes.")
    print(
        f"Train edges: {train_dataset.X_edges.shape}, "
        f"positives={(train_dataset.y_edges == 1).sum()}, "
        f"negatives={(train_dataset.y_edges == 0).sum()}, "
        f"node_targets={len(train_dataset.node_targets)}"
    )
    print(
        f"Val edges:   {val_dataset.X_edges.shape}, "
        f"positives={(val_dataset.y_edges == 1).sum()}, "
        f"negatives={(val_dataset.y_edges == 0).sum()}, "
        f"node_targets={len(val_dataset.node_targets)}"
    )
    print(
        f"Test edges:  {test_dataset.X_edges.shape}, "
        f"positives={(test_dataset.y_edges == 1).sum()}, "
        f"negatives={(test_dataset.y_edges == 0).sum()}, "
        f"node_targets={len(test_dataset.node_targets)}"
    )


if __name__ == "__main__":
    main()