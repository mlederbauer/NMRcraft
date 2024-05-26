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

# Import your data loading utilities
from nmrcraft.data.dataloader import DataLoader


def evaluate_model(y_test, y_pred, y_labels):
    metrics = {}
    cm_list = []
    target_index = 0
    for target_name, labels in y_labels.items():
        cm = confusion_matrix(y_test[:, target_index], y_pred[:, target_index])
        accuracy = accuracy_score(
            y_test[:, target_index], y_pred[:, target_index]
        )
        f1 = f1_score(
            y_test[:, target_index], y_pred[:, target_index], average="macro"
        )
        precision = precision_score(
            y_test[:, target_index],
            y_pred[:, target_index],
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            y_test[:, target_index], y_pred[:, target_index], average="macro"
        )
        # roc_auc = roc_auc_score(y_test[:, target_index], y_pred[:, target_index])
        metrics[target_name] = {
            "Accuracy": accuracy,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            # "ROC-AUC": roc_auc
        }
        labels = labels
        cm_list.append((target_name, cm))
        target_index += 1
    return metrics, cm_list


def main():
    parser = argparse.ArgumentParser(
        description="Simplified model training script."
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=["metal", "E_ligand"],
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
        type=bool,
        default=False,
        help="Use a random baseline model.",
    )
    args = parser.parse_args()

    # Set up logging
    log.basicConfig(
        level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load data
    dataloader = DataLoader(
        target_columns=args.targets,
        dataset_size=args.dataset_size,
        feature_columns=[
            "M_sigma11_ppm",
            "M_sigma22_ppm",
            "M_sigma33_ppm",
            "E_sigma11_ppm",
            "E_sigma22_ppm",
            "E_sigma33_ppm",
        ],
        complex_geometry="oct",
        test_size=0.3,
        random_state=42,
        include_structural_features=False,
        testing=False,
    )
    # Load data
    X_train, X_test, y_train, y_test, y_labels = dataloader.load_data()

    # Predictions for each target
    predictions = np.zeros_like(y_test)
    for i, target_name in enumerate(y_labels.keys()):
        if args.random_baseline:
            unique_vals = np.unique(y_train[:, i])
            predictions[:, i] = np.random.choice(unique_vals, size=len(y_test))
        else:
            most_common = pd.Series(y_train[:, i]).mode()[0]
            predictions[:, i] = np.full(
                shape=y_test[:, i].shape, fill_value=most_common
            )
        target_name = target_name

    # Evaluate the model
    metrics, confusion_matrices = evaluate_model(y_test, predictions, y_labels)
    log.info("Evaluation Metrics: %s", metrics)


if __name__ == "__main__":
    main()
