"""Script to train and evaluate models with multiple targets."""

import argparse
import logging as log
import os

import mlflow
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier

from nmrcraft.analysis import plot_confusion_matrix
from nmrcraft.data import DataLoader
from nmrcraft.evaluation import (
    evaluate_bootstrap,
    evaluate_model,
    metrics_statistics,
)
from nmrcraft.models import load_model, model_configs
from nmrcraft.training import HyperparameterTuner
from nmrcraft.utils import add_rows_metrics, str2bool

# Setup MLflow
mlflow.set_experiment("Final_results")

# Setup parser
parser = argparse.ArgumentParser(
    description="Train a model with MLflow tracking."
)

parser.add_argument(
    "--max_evals",
    type=int,
    default=2,
    help="The max evaluations for the hyperparameter tuning with hyperopt",
)
parser.add_argument(
    "--target",
    type=str,
    default="metal E_ligand X3_ligand",
    help="The Target for the predictions. Choose from: 'metal', 'X1_ligand', 'X2_ligand', 'X3_ligand', 'X4_ligand', 'L_ligand', 'E_ligand'",
)
parser.add_argument(
    "--include_structural",
    type=str2bool,
    default="False",
    help="Handles if structural features will be included or only nmr tensors are used.",
)
parser.add_argument(
    "--plot_folder",
    type=str,
    default="plots/models/",
    help="The Folder where the plots are saved",
)


def main(args) -> pd.DataFrame:

    # Check if folder path exists, if not create it
    if not os.path.exists(args.plot_folder):
        os.makedirs(args.plot_folder)

    # Setup logging
    log.basicConfig(
        format="%(asctime)s %(message)s",
        level=log.INFO,
        force=True,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.getLogger().setLevel(log.INFO)

    dataset_sizes = [
        1.0,
    ]
    models = [
        "random_forest",
        "extra_trees",
    ]

    # Initialize df to store all the info for later plotting
    unified_metrics_columns = [
        "target",
        "model_targets",
        "model",
        "nmr_only",
        "dataset_fraction",
        "max_evals",
        "accuracy_mean",
        "accuracy_lb",
        "accuracy_hb",
        "f1_mean",
        "f1_lb",
        "f1_hb",
    ]
    unified_metrics = pd.DataFrame(columns=unified_metrics_columns)

    with mlflow.start_run():

        for model_name in models:
            config = model_configs[model_name]
            tuner = HyperparameterTuner(
                model_name, config, max_evals=args.max_evals
            )

            for dataset_size in dataset_sizes:
                data_loader = DataLoader(
                    target_columns=args.target,
                    dataset_size=dataset_size,
                )
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_labels,
                ) = data_loader.load_data()

                best_params, _ = tuner.tune(X_train, y_train[:, 1])
                model_func = lambda model_name=model_name, config=config, **params: load_model(
                    model_name, **{**params, **config["model_params"]}
                )
                best_model = model_func(**best_params)
                multioutput_model = MultiOutputClassifier(
                    best_model, n_jobs=-1
                )
                multioutput_model.fit(X_train, y_train)
                y_pred = multioutput_model.predict(X_test)

                metrics, cm_list = evaluate_model(y_test, y_pred, args.target)

                plot_confusion_matrix(
                    cm_list,
                    y_labels,
                    model_name,
                    dataset_size,
                    args.plot_folder,
                )

                bootstrap_metrics = evaluate_bootstrap(
                    X_test, y_test, multioutput_model, args.target
                )

                statistical_metrics = metrics_statistics(bootstrap_metrics)

                unified_metrics = add_rows_metrics(
                    unified_metrics,
                    statistical_metrics,
                    dataset_size,
                    args.include_structural,
                    model_name,
                    args.max_evals,
                )
    return unified_metrics


if __name__ == "__main__":

    # Add arguments
    args = parser.parse_args()
    args.target = args.target.split()

    unified_metrics = main(args)

    # save all the results
    if not os.path.isdir("metrics"):
        os.mkdir("metrics")

    results_path = "metrics/results_multi_target.csv"
    if os.path.exists(results_path):
        existing_data = pd.read_csv(results_path)
        unified_metrics = pd.concat([existing_data, unified_metrics])
    unified_metrics.to_csv(results_path, index=False)
