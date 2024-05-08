import logging as log

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    # confusion_matrix,
    multilabel_confusion_matrix,
    roc_curve,
)

from nmrcraft.data.dataset import DataLoader
from nmrcraft.models.model_config import model_configs
from nmrcraft.models.models import load_model
from nmrcraft.training.hyperparameter_tune import HyperparameterTuner


class Classifier:
    def __init__(
        self,
        model_name: str,
        max_evals: int,
        target: str,
        dataset_size: float,
        feature_columns=None,
    ):
        if not feature_columns:
            feature_columns = [
                "M_sigma11_ppm",
                "M_sigma22_ppm",
                "M_sigma33_ppm",
                "E_sigma11_ppm",
                "E_sigma22_ppm",
                "E_sigma33_ppm",
            ]

        model_config = model_configs[model_name]

        self.tuner = HyperparameterTuner(model_name, model_config, max_evals)

        self.X_train, self.X_test, self.y_train, self.y_test = DataLoader(
            feature_columns=feature_columns,
            target_columns=target,
            dataset_size=dataset_size,
        ).load_data()

    def hyperparameter_tune(self):
        log.INFO(
            f"Performing Hyperparameter tuning for the Model ({self.model_name})"
        )

        # DATA LEAKAGE!!! MUST be done by CV!!!!
        self.best_params, _ = self.tuner.tune(
            self.X_train, self.y_train, self.X_test, self.y_test
        )

    def train(self):
        model_func = lambda **params: load_model(
            self.model_name, **{**params, **self.model_config}
        )

        self.model = model_func(**self.best_params)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> pd.DataFrame():
        """
        Evaluate the performance of the trained machine learning model.

        Returns:
            Tuple[Dict[str, float], Any, Any, Any]: A tuple containing:
                - A dictionary with evaluation metrics (accuracy, f1_score, roc_auc).
                - The confusion matrix.
                - The false positive rate.
                - The true positive rate.
        """
        results_df = pd.DataFrame(
            index=["accuracy", "f1_score", "roc_auc", "cm", "fpr", "tpr"]
        )
        y_pred = self.model.predict(self.X_test)
        results_df.loc["accuracy"] = accuracy_score(self.y_test, y_pred)
        results_df.loc["f1_score"] = f1_score(
            self.y_test, y_pred, average="weighted"
        )
        results_df.loc["cm"] = multilabel_confusion_matrix(self.y_test, y_pred)
        results_df.loc["fpr"], results_df.loc["tpr"], thresholds = roc_curve(
            self.y_test, self.model.predict_proba(self.X_test)[:, 1]
        )
        results_df.loc["roc_auc"] = auc(
            results_df.loc["fpr"], results_df.loc["tpr"]
        )

        # TODO: Add std for errorbars -> Bootstraping

        return results_df
