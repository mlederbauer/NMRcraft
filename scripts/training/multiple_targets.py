import argparse
import logging as log
import os

import mlflow
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier

from nmrcraft.analysis import plotting
from nmrcraft.data.dataloader import DataLoader
from nmrcraft.evaluation import evaluation
from nmrcraft.models.model_configs import model_configs
from nmrcraft.models.models import load_model
from nmrcraft.training.hyperparameter_tune import HyperparameterTuner
from nmrcraft.utils.general import add_rows_metrics

# Setup MLflow
mlflow.set_experiment("Test_final_results")

# Setup parser
parser = argparse.ArgumentParser(
    description="Train a model with MLflow tracking."
)

parser.add_argument(
    "--max_evals",
    type=int,
    default=3,
    help="The max evaluations for the hyperparameter tuning with hyperopt",
)
parser.add_argument(
    "--target",
    type=str,
    default=["metal", "X3_ligand", "X4_ligand"],
    help="The Target for the predictions. Choose from: 'metal', 'X1_ligand', 'X2_ligand', 'X3_ligand', 'X4_ligand', 'L_ligand', 'E_ligand'",
)
parser.add_argument(
    "--include_structural",
    type=bool,
    default=False,
    help="Handles if structural features will be included or only nmr tensors are used.",
)
parser.add_argument(
    "--structural_features",
    type=bool,
    default=False,
    help="Whether to include ligands or not",
)
parser.add_argument(
    "--plot_folder",
    type=str,
    default="plots/",
    help="The Folder where the plots are saved",
)


if __name__ == "__main__":
    # Add arguments
    args = parser.parse_args()

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
        # 0.01,
        0.1,
        0.15,
        # 0.5,
        # 1.0,
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
        model_metrics = []

        for model_name in models:
            data = pd.DataFrame()
            config = model_configs[model_name]
            tuner = HyperparameterTuner(
                model_name, config, max_evals=args.max_evals
            )

            for dataset_size in dataset_sizes:
                data_loader = DataLoader(
                    target_columns=args.target,
                    dataset_size=dataset_size,
                    include_structural_features=args.structural_features,
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

                metrics, cm_list = evaluation.evaluate_model(
                    y_test, y_pred, args.target
                )

                plotting.plot_confusion_matrix(
                    cm_list,
                    y_labels,
                    model_name,
                    dataset_size,
                    args.plot_folder,
                )

                bootstrap_metrics = evaluation.evaluate_bootstrap(
                    X_test, y_test, multioutput_model, args.target
                )

                statistical_metrics = evaluation.metrics_statistics(
                    bootstrap_metrics
                )

                unified_metrics = add_rows_metrics(
                    unified_metrics,
                    statistical_metrics,
                    dataset_size,
                    args.include_structural,
                    model_name,
                    args.max_evals,
                )
                # Add all the newly generated metrics to the unified dataframe
    # save all the results
    if not os.path.isdir("metrics"):
        os.mkdir("metrics")
    unified_metrics.to_csv(f"metrics/metrics_{args.target}.csv")
    # mlflow.log_input(unified_metrics, context="unified metrics")
