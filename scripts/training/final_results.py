import argparse
import logging as log
import os

import mlflow
import pandas as pd

from nmrcraft.evaluation.visualizer import Visualizer
from nmrcraft.models.classifier import Classifier

# Setup MLflow
mlflow.set_experiment("Test_final_results")

# Setup parser
parser = argparse.ArgumentParser(
    description="Train a model with MLflow tracking."
)

parser.add_argument(
    "--max_evals",
    type=int,
    default=10,
    help="The max evaluatins for the hyperparameter tuning with hyperopt",
)
parser.add_argument(
    "--target",
    type=str,
    default="metal",
    help="The Target for the predictions. Choose from: 'metal', 'X1', 'X2', 'X3', 'X4', 'L', 'E' ",
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
        0.01,
        0.1,
        # 0.5,
        1.0,
    ]
    models = [
        "random_forest",
        "logistic_regression",
        "gradient_boosting",
        "svc",
    ]

    with mlflow.start_run():
        model_data = pd.DataFrame(
            columns=["accuracy", "f1_score", "dataset_size", "model"]
        )
        for model in models:
            data = pd.DataFrame()
            for dataset_size in dataset_sizes:
                # Create a instance of the Classifier_Class
                C = Classifier(
                    model_name=model,
                    max_evals=args.max_evals,
                    target=args.target,
                    dataset_size=dataset_size,
                    random_state=11,
                )
                # mlflow.log_metrics("dataset_size", dataset_size, step=i)
                C.hyperparameter_tune()
                C.train()
                new_data = C.evaluate()
                # data[str(dataset_size)] = new_data
                data = pd.concat(
                    [data, new_data.assign(dataset_size=dataset_size)],
                )
                data_BS = C.train_bootstraped(10)
                model_data = pd.concat([model_data, data_BS])

            data.index = dataset_sizes
            visualizer = Visualizer(
                model_name=model, data=data, folder_path=args.plot_folder
            )
            path_ROC = visualizer.plot_ROC(filename=f"ROC_Plot_{model}.png")
            mlflow.log_artifact(path_ROC, f"ROC_Plot_{model}.png")

        print(model_data)

        path_AC = visualizer.plot_metric(
            data=model_data,
            metric="accuracy",
            title="Accuracy",
            filename="accuracy.png",
        )
        path_F1 = visualizer.plot_metric(
            data=model_data,
            metric="f1_score",
            title="F1 Score",
            filename="f1_score.png",
        )

        # mlflow.log_artifact("F1_Plot", path_F1)
        # mlflow.log_artifact("Accuracy_Plot", path_AC)