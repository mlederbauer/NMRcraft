import argparse
import logging as log

import numpy as np
import pandas as pd

from nmrcraft.data.dataloader import DataLoader
from nmrcraft.evaluation.evaluation import evaluate_model


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
    X_train, X_test, y_train, y_test, y_labels = dataloader.load_data()

    predictions = np.zeros_like(y_test)

    for i in range(len(args.targets)):  # Loop through each target column
        if args.random_baseline:
            unique_vals = np.unique(y_train[:, i])
            predictions[:, i] = np.random.choice(unique_vals, size=len(y_test))
        else:
            most_common = pd.Series(y_train[:, i]).mode()[0]
            predictions[:, i] = np.full(
                shape=y_test[:, i].shape, fill_value=most_common
            )

    # Evaluate the model
    metrics, confusion_matrices = evaluate_model(y_test, predictions, y_labels)
    log.info("Evaluation Metrics: %s", metrics)


if __name__ == "__main__":
    main()
