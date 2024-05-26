import argparse

import mlflow
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from nmrcraft.data.dataloader import DataLoader

# precision_score,
# recall_score,
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
            target_type="categorical",
        )

        # Load and preprocess data
        X_train, X_test, y_train, y_test, y_labels = data_loader.load_data()

        tuner = HyperparameterTuner(model_name, config, max_evals=1)
        best_params, _ = tuner.tune(X_train, y_train)

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

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        ac = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"Accuracy: {ac}, F1: {f1}, Confusion Matrix:\n{cm}")


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
        default="X3",
        help="Specify the target(s) to select (metal, X1-X4, L, E or combinations of them, e.g., metal_1X_L)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gradient_boosting",
        help="Model name to load ('random_forest', 'logistic_regression', 'svc')",
    )
    args = parser.parse_args()

    main(args.dataset_size, args.target, args.model_name)