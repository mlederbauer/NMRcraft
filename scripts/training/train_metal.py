import argparse
import os

import mlflow
from sklearn.preprocessing import LabelEncoder

from nmrcraft.analysis.plotting import plot_confusion_matrix, plot_roc_curve
from nmrcraft.data.dataset import DataLoader
from nmrcraft.evaluation.evaluation import model_evaluation
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

        if target == "metal":
            # if the target is that, we encode the metals as Mo, W.
            # this is hard coded in the data loader and needs to be changed next
            pass

        # TODO: add categorical feature columns
        # TODO: add target column, here e.g. "metal" with the two possibilities "Mo, W"
        # Create a DataLoader instance

        data_loader = DataLoader(
            feature_columns=feature_columns,
            target_column=args.target,
            dataset_size=args.dataset_size,
        )

        label_encoder = LabelEncoder()
        label_encoder.fit(["Mo", "W"])  # TODO adapt this for other targets!
        # FIXME: ugly hard coded right now to make confusion matrix plotting work

        # Load and preprocess data
        X_train, X_test, y_train, y_test = data_loader.load_data()

        tuner = HyperparameterTuner(model_name, config)
        best_params, _ = tuner.tune(X_train, y_train, X_test, y_test)

        model_func = lambda **params: load_model(
            model_name, **{**params, **config["model_params"]}
        )
        best_model = model_func(**best_params)
        best_model.fit(X_train, y_train)

        metrics, cm, fpr, tpr = model_evaluation(best_model, X_test, y_test)
        mlflow.log_params(best_params)
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_size": dataset_size,
                "target": target,
            }
        )
        mlflow.log_metrics(metrics)

        # TODO: refactor this to a function nmrcraft/analysis/ or nmrcraft/evaluation
        fig_path = "scratch/"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        cm_path = os.path.join(fig_path, "cm.png")
        title = r"Confusion matrix, TODO add LaTeX symbols"
        plot_confusion_matrix(
            cm, classes=label_encoder.classes_, title=title, path=cm_path
        )
        roc_path = os.path.join(fig_path, "roc.png")
        title = r"ROC curve, TODO add LaTeX symbols"
        plot_roc_curve(
            fpr, tpr, metrics["roc_auc"], title=title, path=roc_path
        )

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        mlflow.sklearn.log_model(best_model, "model")

        print(f"Accuracy: {metrics['accuracy']}")


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
        "--target", type=str, default="metal", help="Target variable name"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="random_forest",
        help="Model name to load",
    )
    args = parser.parse_args()

    main(args.dataset_size, args.target, args.model_name)
