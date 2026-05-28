import numpy as np

from src.strategy2.config import Strategy2Config
from src.strategy2.train import train_edge_model, save_training_artifacts


def main() -> None:
    print("Încarc dataset-ul Strategiei 2...")
    X = np.load("data/processed/strategy2/mtsplib_edge_dataset_X.npy")
    y = np.load("data/processed/strategy2/mtsplib_edge_dataset_y.npy")

    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    print(f"Pozitive: {(y == 1).sum()}")
    print(f"Negative: {(y == 0).sum()}\n")

    config = Strategy2Config()

    print("Încep antrenarea modelului...")
    artifacts = train_edge_model(
        X=X,
        y=y,
        config=config,
        validation_ratio=0.2,
        random_seed=42
    )

    output_dir = "data/processed/strategy2/model_artifacts"
    save_training_artifacts(artifacts, output_dir)

    print("\nModel antrenat și salvat cu succes.")
    print(f"Model: {output_dir}/edge_model.pt")
    print(f"Feature mean: {output_dir}/feature_mean.npy")
    print(f"Feature std: {output_dir}/feature_std.npy")
    print(f"History: {output_dir}/history.json")

    print("\nUltimele metrici:")
    print(f"Train loss final: {artifacts.history['train_loss'][-1]:.4f}")
    print(f"Val loss final: {artifacts.history['val_loss'][-1]:.4f}")
    print(f"Val accuracy final: {artifacts.history['val_accuracy'][-1]:.4f}")
    print(f"Val F1 final: {artifacts.history['val_f1'][-1]:.4f}")


if __name__ == "__main__":
    main()