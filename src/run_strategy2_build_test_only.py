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
    _, _, test_instances = build_synthetic_instance_split(
        config=synthetic_config,
        random_seed=42
    )

    print(f"Test instances: {len(test_instances)}\n")

    print("Construiesc dataset-ul TEST...")
    test_dataset = build_edge_dataset_for_instances_v2(
        instances=test_instances,
        strategy2_config=strategy2_config,
        teacher_solver=teacher_solver,
        random_seed=44
    )
    save_edge_dataset(test_dataset, str(output_dir / "test_edge_dataset"))

    print("\nDataset-ul TEST a fost salvat cu succes.")
    print(f"Test X: {test_dataset.X.shape}, positives={(test_dataset.y == 1).sum()}, negatives={(test_dataset.y == 0).sum()}")


if __name__ == "__main__":
    main()