import json
from pathlib import Path

from src.strategy2.data_loader_v3 import load_instance_examples_v3
from src.strategy2.train_v3 import (
    TrainingConfigV3,
    evaluate_model_v3,
    save_training_artifacts_v3,
    train_model_v3,
)


def main() -> None:
    base_dir = Path("data/processed/strategy2_v3")

    print("Încarc exemplele train / val / test pentru v3...")

    train_examples = load_instance_examples_v3(str(base_dir / "train_dataset_v3"))
    val_examples = load_instance_examples_v3(str(base_dir / "val_dataset_v3"))
    test_examples = load_instance_examples_v3(str(base_dir / "test_dataset_v3"))

    print(f"Train instances: {len(train_examples)}")
    print(f"Val instances: {len(val_examples)}")
    print(f"Test instances: {len(test_examples)}\n")

    config = TrainingConfigV3(
        hidden_dim=128,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=80,
        early_stopping_patience=10,
        edge_loss_weight=1.0,
        node_loss_weight=0.5,
        device="cpu"
    )

    print("Încep antrenarea modelului v3...")
    artifacts = train_model_v3(
        train_examples=train_examples,
        val_examples=val_examples,
        config=config
    )

    print("\nEvaluez modelul v3 pe test...")
    test_metrics = evaluate_model_v3(
        model=artifacts.model,
        test_examples=test_examples,
        edge_feature_mean=artifacts.edge_feature_mean,
        edge_feature_std=artifacts.edge_feature_std,
        config=config
    )

    output_dir = base_dir / "model_artifacts_v3"
    save_training_artifacts_v3(artifacts, str(output_dir))

    with (output_dir / "test_metrics_v3.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nModelul v3 a fost antrenat și salvat.")
    print(f"Best epoch: {artifacts.best_epoch}")
    print(f"Best val total loss: {artifacts.best_val_loss:.4f}")
    print(f"Test total loss: {test_metrics['test_total_loss']:.4f}")
    print(f"Test edge loss: {test_metrics['test_edge_loss']:.4f}")
    print(f"Test node loss: {test_metrics['test_node_loss']:.4f}")
    print(f"Artefacte salvate în: {output_dir}")


if __name__ == "__main__":
    main()