import json
from pathlib import Path

import numpy as np

from src.strategy2.train_v2 import (
    TrainingConfigV2,
    evaluate_edge_model_v2,
    save_training_artifacts_v2,
    train_edge_model_v2,
)


def main() -> None:
    base_dir = Path("data/processed/strategy2_v2")

    print("Încarc dataset-urile train / val / test...")

    X_train = np.load(base_dir / "train_edge_dataset_X.npy")
    y_train = np.load(base_dir / "train_edge_dataset_y.npy")

    X_val = np.load(base_dir / "val_edge_dataset_X.npy")
    y_val = np.load(base_dir / "val_edge_dataset_y.npy")

    X_test = np.load(base_dir / "test_edge_dataset_X.npy")
    y_test = np.load(base_dir / "test_edge_dataset_y.npy")

    print(f"Train: X={X_train.shape}, y={y_train.shape}, positives={(y_train == 1).sum()}, negatives={(y_train == 0).sum()}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}, positives={(y_val == 1).sum()}, negatives={(y_val == 0).sum()}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}, positives={(y_test == 1).sum()}, negatives={(y_test == 0).sum()}")
    print()

    config = TrainingConfigV2(
        batch_size=1024,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=100,
        early_stopping_patience=12,
        hidden_dims=(256, 256, 128, 64),
        dropout=0.2,
        device="cpu",
    )

    print("Încep antrenarea modelului v2...")
    artifacts = train_edge_model_v2(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config
    )

    print("\nEvaluez pe test set...")
    test_metrics = evaluate_edge_model_v2(
        model=artifacts.model,
        X_test=X_test,
        y_test=y_test,
        feature_mean=artifacts.feature_mean,
        feature_std=artifacts.feature_std,
        device=config.device
    )

    output_dir = base_dir / "model_artifacts_v2"
    save_training_artifacts_v2(artifacts, str(output_dir))

    with (output_dir / "test_metrics_v2.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("\nModelul v2 a fost antrenat și salvat.")
    print(f"Best epoch: {artifacts.best_epoch}")
    print(f"Best val loss: {artifacts.best_val_loss:.4f}")
    print(f"Test loss: {test_metrics['test_loss']:.4f}")
    print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Test F1: {test_metrics['test_f1']:.4f}")
    print(f"Artefacte salvate în: {output_dir}")


if __name__ == "__main__":
    main()