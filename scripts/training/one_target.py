import argparse
import logging as log
import os

import mlflow
import numpy as np
import pandas as pd

from nmrcraft.analysis import plotting
from nmrcraft.data.dataloader import DataLoader
from nmrcraft.evaluation import evaluation
from nmrcraft.models.model_configs import model_configs
from nmrcraft.models.models import load_model
from nmrcraft.training.hyperparameter_tune import HyperparameterTuner

# Setup MLflow
mlflow.set_experiment("Test_final_results")

# Setup parser
parser = argparse.ArgumentParser(
    description="Train a model with MLflow tracking."
)

parser.add_argument(
    "--max_evals",
    type=int,
    default=1,
    help="The max evaluations for the hyperparameter tuning with hyperopt",
)
parser.add_argument(
    "--target",
    type=str,
    default=["X3_ligand"],
    help="The Target for the predictions. Choose from: 'metal', 'X1_ligand', 'X2_ligand', 'X3_ligand', 'X4_ligand', 'L_ligand', 'E_ligand'",
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
        # 0.15
        # 0.5,
        # 1.0,
    ]
    models = [
        # "random_forest",
        # "logistic_regression",
        # "gradient_boosting",
        "svc",
    ]

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
                )
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_labels,
                ) = data_loader.load_data()

                best_params, _ = tuner.tune(X_train, np.squeeze(y_train))
                model_func = lambda model_name=model_name, config=config, **params: load_model(
                    model_name, **{**params, **config["model_params"]}
                )
                best_model = model_func(**best_params)
                best_model.fit(X_train, y_train)
                y_pred = np.atleast_2d(best_model.predict(X_test)).T

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
                    X_test, y_test, best_model, args.target
                )

    # TODO: Adapt this code to the new structure
    #         visualizer = Visualizer(
    #             model_name=model_name,
    #             cm=cm,
    #             rates=rates_df,
    #             metrics=metrics,
    #             folder_path=args.plot_folder,
    #             classes=C.y_labels,
    #             dataset_size=str(dataset_size),
    #         )
    #         path_CM = visualizer.plot_confusion_matrix()

    #     data.index = dataset_sizes
    #     model_metrics.append(data)
    #     data.index = dataset_sizes

    # path_AC = visualizer.plot_metric(
    #     data=model_data,
    #     metric="accuracy",
    #     title="Accuracy",
    #     filename="accuracy.png",
    # )
    # path_F1 = visualizer.plot_metric(
    #     data=model_data,
    #     metric="f1_score",
    #     title="F1 Score",
    #     filename="f1_score.png",
    # )

    # for df, model in zip(model_metrics, models):
    #     print(model)
    #     print(df)
