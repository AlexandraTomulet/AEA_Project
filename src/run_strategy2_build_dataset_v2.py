from pathlib import Path

from src.strategy2.advanced_data import (
    SyntheticDatasetConfig,
    build_synthetic_instance_split,
)
from src.strategy2.config import Strategy2Config
from src.strategy2.dataset_builder_v2 import (
    build_edge_dataset_for_instances_v2,
    save_edge_dataset,
)
from src.strategy2.teacher_solver import MultiRestartTeacherSolver, TeacherConfig


def main() -> None:
    output_dir = Path("data/processed/strategy2_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_config = SyntheticDatasetConfig(
        train_instances_per_setting=40,
        val_instances_per_setting=10,
        test_instances_per_setting=10,
    )

    strategy2_config = Strategy2Config(
        num_candidate_neighbors=15,
        num_epochs=50,
        batch_size=512,
        negative_sampling_ratio=3,
    )

    teacher_solver = MultiRestartTeacherSolver(
        config=TeacherConfig(num_restarts=8)
    )

    print("Generez split-ul de instanțe...")
    train_instances, val_instances, test_instances = build_synthetic_instance_split(
        config=synthetic_config,
        random_seed=42
    )

    print(f"Train instances: {len(train_instances)}")
    print(f"Val instances: {len(val_instances)}")
    print(f"Test instances: {len(test_instances)}\n")

    print("Construiesc dataset-ul TRAIN...")
    train_dataset = build_edge_dataset_for_instances_v2(
        instances=train_instances,
        strategy2_config=strategy2_config,
        teacher_solver=teacher_solver,
        random_seed=42
    )
    save_edge_dataset(train_dataset, str(output_dir / "train_edge_dataset"))

    print("\nConstruiesc dataset-ul VAL...")
    val_dataset = build_edge_dataset_for_instances_v2(
        instances=val_instances,
        strategy2_config=strategy2_config,
        teacher_solver=teacher_solver,
        random_seed=43
    )
    save_edge_dataset(val_dataset, str(output_dir / "val_edge_dataset"))

    print("\nConstruiesc dataset-ul TEST...")
    test_dataset = build_edge_dataset_for_instances_v2(
        instances=test_instances,
        strategy2_config=strategy2_config,
        teacher_solver=teacher_solver,
        random_seed=44
    )
    save_edge_dataset(test_dataset, str(output_dir / "test_edge_dataset"))

    print("\nDataset-urile au fost salvate cu succes.")
    print(f"Train X: {train_dataset.X.shape}, positives={(train_dataset.y == 1).sum()}, negatives={(train_dataset.y == 0).sum()}")
    print(f"Val   X: {val_dataset.X.shape}, positives={(val_dataset.y == 1).sum()}, negatives={(val_dataset.y == 0).sum()}")
    print(f"Test  X: {test_dataset.X.shape}, positives={(test_dataset.y == 1).sum()}, negatives={(test_dataset.y == 0).sum()}")


if __name__ == "__main__":
    main()