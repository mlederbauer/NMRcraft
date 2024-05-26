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
    default=3,
    help="The max evaluatins for the hyperparameter tuning with hyperopt",
)
parser.add_argument(
    "--target",
    type=str,
    default="X3",
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
        # 0.01,
        0.1,
        0.15,
        # 0.5,
        # 1.0,
    ]
    models = [
        # "random_forest",
        "logistic_regression",
        # "gradient_boosting",
        # "svc",
    ]

    with mlflow.start_run():
        model_data = pd.DataFrame(
            columns=["accuracy", "f1_score", "dataset_size", "model"]
        )
        model_metrics = []
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
                rates_df, metrics, cm = C.evaluate()
                print(rates_df)
                print(metrics)
                print(cm)

                # data[str(dataset_size)] = new_data
                data = pd.concat([data, metrics])
                data_BS = C.train_bootstraped(10)
                model_data = pd.concat([model_data, data_BS])

                visualizer = Visualizer(
                    model_name=model,
                    cm=cm,
                    rates=rates_df,
                    metrics=metrics,
                    folder_path=args.plot_folder,
                    classes=C.classes,
                    dataset_size=str(dataset_size),
                )
                path_CM = visualizer.plot_confusion_matrix()
            # print(data)
            data.index = dataset_sizes
            model_metrics.append(data)
            data.index = dataset_sizes

            # path_ROC = visualizer.plot_ROC(filename=f"ROC_Plot_{model}.png")
            # mlflow.log_artifact(path_ROC, f"ROC_Plot_{model}.png")

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

        for df, model in zip(model_metrics, models):
            print(model)
            print(df)

        # mlflow.log_artifact("F1_Plot", path_F1)
        # mlflow.log_artifact("Accuracy_Plot", path_AC)
