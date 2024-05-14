import argparse

import mlflow

from nmrcraft.analysis.plotting import plot_confusion_matrix, plot_roc_curve
from nmrcraft.data.dataset import DataLoader
from nmrcraft.evaluation.evaluation import (
    get_cm_path,
    get_roc_path,
    model_evaluation,
    model_evaluation_nD,
)
from nmrcraft.models.model_configs import model_configs
from nmrcraft.models.models import load_model
from nmrcraft.training.hyperparameter_tune import HyperparameterTuner
from nmrcraft.utils.set_seed import set_seed

set_seed()


def main(dataset_size, target, model_name):
    # TODO: better experiment naming
    mlflow.set_experiment("Ceci_nest_pas_un_experiment")

    with mlflow.start_run():
        config = model_configs[model_name]

        feature_columns = [
            "M_sigma11_ppm",
            "M_sigma22_ppm",
            "M_sigma33_ppm",
            "E_sigma11_ppm",
            "E_sigma22_ppm",
            "E_sigma33_ppm",
        ]

        data_loader = DataLoader(
            feature_columns=feature_columns,
            target_columns=args.target,
            dataset_size=args.dataset_size,
        )

        # Load and preprocess data
        X_train, X_test, y_train, y_test, y_labels = data_loader.load_data()

        tuner = HyperparameterTuner(model_name, config, max_evals=1)
        best_params, _ = tuner.tune(X_train, y_train, X_test, y_test)

        model_func = lambda **params: load_model(
            model_name, **{**params, **config["model_params"]}
        )
        best_model = model_func(**best_params)
        best_model.fit(X_train, y_train)

        mlflow.log_params(best_params)
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_size": dataset_size,
                "target": target,
            }
        )

        if isinstance(y_test, list):  # if target is 1D
            metrics, cm, fpr, tpr = model_evaluation(
                best_model, X_test, y_test, y_labels, data_loader
            )

            title = r"Confusion matrix, TODO add LaTeX symbols"
            plot_confusion_matrix(
                cm,
                classes=data_loader.confusion_matrix_label_adapter(y_labels),
                title=title,
                path=get_cm_path(),
            )
            # Plot ROC
            title = r"ROC curve, TODO add LaTeX symbols"
            plot_roc_curve(
                fpr, tpr, metrics["roc_auc"], title=title, path=get_roc_path()
            )
            # Logging 1D only data
            mlflow.log_artifact(get_roc_path())

        elif (
            data_loader.more_than_one_target()
        ):  # Multidimensional target Array and Multiple targets
            metrics, cm = model_evaluation_nD(
                best_model, X_test, y_test, y_labels, data_loader
            )

            title = r"Confusion matrix, TODO add LaTeX symbols"
            plot_confusion_matrix(
                cm,
                classes=data_loader.confusion_matrix_label_adapter(y_labels),
                title=title,
                path=get_cm_path(),
                full=False,
                columns_set=data_loader.get_target_columns_separated(),
            )

        else:  # Multidimensional target Array and single target
            metrics, cm = model_evaluation_nD(
                best_model, X_test, y_test, y_labels, data_loader
            )
            title = r"Confusion matrix, TODO add LaTeX symbols"
            plot_confusion_matrix(
                cm,
                classes=data_loader.confusion_matrix_label_adapter(y_labels),
                title=title,
                path=get_cm_path(),
            )

        # Logging common data
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")
        print(f"Accuracy: {metrics['accuracy']}")
        mlflow.log_artifact(get_cm_path())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with MLflow tracking."
    )
    parser.add_argument(
        "--dataset_size",
        type=float,
        default=0.01,
        help="Fraction of dataset to use",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="X1",
        help="Specify the target(s) to select (metal, X1-X4, L, E or combinations of them, e.g., metal_1X_L)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="random_forest",
        help="Model name to load ('random_forest', 'gradient_boosting', 'logistic_regression', 'svc')",
    )
    args = parser.parse_args()

    main(args.dataset_size, args.target, args.model_name)
