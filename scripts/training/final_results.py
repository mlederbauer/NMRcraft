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
    "--model",
    type=str,
    default="random_forest",
    help="The Classifier used for the predictions. Choose from: 'random_forest', 'gradient_boosting', 'logistic_regression', 'svc' ",
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

    dataset_sizes = [0.01, 0.1, 1]

    with mlflow.start_run():
        data = pd.DataFrame()
        for i, dataset_size in enumerate(dataset_sizes):
            # Create a instance of the Classifier_Class
            C = Classifier(
                model_name=args.model,
                max_evals=args.max_evals,
                target=args.target,
                dataset_size=dataset_size,
            )
            # mlflow.log_metrics("dataset_size", dataset_size, step=i)
            print(i)
            C.hyperparameter_tune()
            C.train()
            new_data = C.evaluate()
            # data[str(dataset_size)] = new_data
            data = pd.concat(
                [data, new_data.assign(dataset_size=dataset_size)],
                ignore_index=True,
            )

        visualizer = Visualizer(
            model_name=args.model, data=data, folder_path=args.plot_folder
        )
        path_ROC = visualizer.plot_ROC()
        # path_F1 = visualizer.plot_F1()
        # path_AC = visualizer.plot_Accuracy()

        mlflow.log_artifact(path_ROC, "ROC_Plot")
        # mlflow.log_artifact("F1_Plot", path_F1)
        # mlflow.log_artifact("Accuracy_Plot", path_AC)
