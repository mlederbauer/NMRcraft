import logging as log

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils import resample

from nmrcraft.data.dataloader import DataLoader
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
        random_state=42,
        include_structural_features=True,
        complex_geometry="oct",
        test_size=0.2,
        testing=False,
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

        data_loader = DataLoader(
            feature_columns=feature_columns,
            target_columns=target,
            dataset_size=dataset_size,
            include_structural_features=include_structural_features,
            complex_geometry=complex_geometry,
            test_size=test_size,
            random_state=random_state,
            testing=testing,
        )
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.y_labels,
        ) = data_loader.load_data()

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

    def train_bootstrapped(self, n_times=10):
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
            # self.hyperparameter_tune()
            self.train()
            rates_df, metrics, cm = self.evaluate()
            accuracy.append(metrics["Accuracy"])
            f1_score.append(metrics["F1"])
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

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate the performance of the trained machine learning model.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation metrics (accuracy, f1_score, roc_auc),
                        the confusion matrix, false positive rates, and true positive rates for each class.
        """
        y_pred = self.model.predict(self.X_test)
        # print(y_pred)
        # accuracy = accuracy_score(self.y_test, y_pred)
        # f1 = f1_score(self.y_test, y_pred, average="weighted")

        # Binarize the output
        # y_test_bin = label_binarize(
        #     self.y_test, classes=np.unique(self.y_test)
        # )

        # Number of classes
        # n_classes = y_test_bin.shape[1]
        cm = confusion_matrix(self.y_test, y_pred)

        def calculate_fpr_fnr(cm):
            FPR = []
            FNR = []
            num_classes = cm.shape[0]
            for i in range(num_classes):
                FP = cm[:, i].sum() - cm[i, i]
                TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                FN = cm[i, :].sum() - cm[i, i]
                TP = cm[i, i]

                FPR.append(FP / (FP + TN))
                FNR.append(FN / (FN + TP))
            return np.array(FPR), np.array(FNR)

        # Calculate FPR and FNR for each class
        FPR, FNR = calculate_fpr_fnr(cm)
        rates_df = pd.DataFrame()
        rates_df["FPR"] = FPR
        rates_df["FNR"] = FNR
        rates_df.index = self.y_labels

        # Calculating macro-averaged F1 Score, Precision, Recall
        Precision = precision_score(self.y_test, y_pred, average="macro")
        Recall = recall_score(self.y_test, y_pred, average="macro")
        F1 = f1_score(self.y_test, y_pred, average="macro")

        # Calculating Accuracy
        Accuracy = accuracy_score(self.y_test, y_pred)

        metrics = pd.DataFrame()
        metrics["Accuracy"] = [Accuracy]
        metrics["Recall"] = [Recall]
        metrics["F1"] = [F1]
        metrics["Precision"] = [Precision]

        cm = pd.DataFrame(cm)
        cm.columns = self.y_labels
        cm.index = self.y_labels

        return rates_df, metrics, cm
