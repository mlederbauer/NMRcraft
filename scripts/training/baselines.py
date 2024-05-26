import argparse
import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Import your data loading and model configuration utilities
from nmrcraft.data.dataloader import DataLoader


def load_data(target, dataset_size):
    feature_columns = [
        "M_sigma11_ppm",
        "M_sigma22_ppm",
        "M_sigma33_ppm",
        "E_sigma11_ppm",
        "E_sigma22_ppm",
        "E_sigma33_ppm",
    ]
    target_columns = target
    complex_geometry = "oct"
    test_size = 0.3
    random_state = 42
    dataset_size = 0.1
    include_structural_features = False
    testing = False
    dataloader = DataLoader(
        feature_columns=feature_columns,
        target_columns=target_columns,
        complex_geometry=complex_geometry,
        test_size=test_size,
        random_state=random_state,
        dataset_size=dataset_size,
        include_structural_features=include_structural_features,
        testing=testing,
    )
    return dataloader.load_data()


def evaluate_model(y_test, y_pred, y_labels):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")

    metrics = {
        "Accuracy": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
    }

    return metrics, cm


def main():
    parser = argparse.ArgumentParser(
        description="Simplified model training script."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="metal_E",
        help="The Target for the predictions.",
    )
    parser.add_argument(
        "--dataset_size",
        type=float,
        default=1.0,
        help="Size of the dataset to load.",
    )
    parser.add_argument(
        "--random_baseline",
        action="store_true",
        help="Use a random baseline model.",
    )
    args = parser.parse_args()

    # Set up logging
    log.basicConfig(
        level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load data
    X_train, X_test, y_train, y_test, y_labels = load_data(
        target=args.target, dataset_size=args.dataset_size
    )

    if args.random_baseline:
        # Implement random choice baseline
        predictions = np.random.choice(np.unique(y_train), size=len(y_test))
    else:
        # Implement most common choice baseline
        most_common = pd.Series(y_train).mode()[0]
        predictions = np.full(shape=y_test.shape, fill_value=most_common)

    # Evaluate the model
    metrics, confusion_mtx = evaluate_model(y_test, predictions, y_labels)
    log.info("Evaluation Metrics: %s", metrics)

    # Optionally save the results and any plots


if __name__ == "__main__":
    main()
