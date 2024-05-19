import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    # confusion_matrix,
    multilabel_confusion_matrix,
    roc_curve,
)
from sklearn.utils import resample

from nmrcraft.data.dataset import DataLoader
from nmrcraft.models.model_configs import model_configs
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
        random_state=None,
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
        self.model_name = model_name
        self.model_config = model_configs[model_name]
        self.max_evals = max_evals
        self.random_state = random_state
        self.dataset_size = dataset_size

        self.tuner = HyperparameterTuner(
            model_name=self.model_name,
            model_config=self.model_config,
            max_evals=self.max_evals,
        )  # algo is set to default value, TODO: change this in declaration of Classifier is necessary

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_labels,
        ) = DataLoader(
            feature_columns=feature_columns,
            target_columns=target,
            dataset_size=dataset_size,
        ).load_data()

    def hyperparameter_tune(self):
        log.info(
            f"Performing Hyperparameter tuning for the Model ({self.model_name})"
        )

        self.best_params, _ = self.tuner.tune(self.X_train, self.y_train)

    def train(self):
        """
        Train the machine learning model using the best hyperparameters.

        Returns:
            None
        """
        all_params = {**self.model_config["model_params"], **self.best_params}
        self.model = load_model(self.model_name, **all_params)
        self.model.fit(self.X_train, self.y_train)

    def train_bootstraped(self, n_times=10):
        accuracy = []
        f1_score = []
        i = 0
        while i < n_times:
            self.X_train, self.y_train = resample(
                self.X_train,
                self.y_train,
                replace=True,
                random_state=self.random_state,
            )
            self.hyperparameter_tune()
            self.train()
            eval_data = self.evaluate()
            accuracy.append(eval_data["accuracy"])
            f1_score.append(eval_data["f1_score"])
            i += 1
        new_row = {
            "accuracy": np.mean(accuracy),
            "accuracy_std": np.std(accuracy),
            "f1_score": np.mean(f1_score),
            "f1_score_std": np.std(f1_score),
            "dataset_size": self.dataset_size,
            "model": self.model_name,
        }
        return pd.DataFrame([new_row])

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
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="weighted")
        fpr, tpr, _ = roc_curve(
            self.y_test, self.model.predict_proba(self.X_test)[:, 1]
        )
        cm = multilabel_confusion_matrix(self.y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Create DataFrame with consistent structure
        results_df = pd.DataFrame(
            {
                "accuracy": [accuracy],
                "f1_score": [f1],
                "roc_auc": [roc_auc],
                "fpr": [fpr.tolist()],
                "cm": [cm.tolist()],
                "tpr": [tpr.tolist()],
            }
        )

        return results_df
