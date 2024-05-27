from typing import Dict, List, Tuple

import numpy as np
import scipy.stats as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils import resample


def evaluate_model(
    y_test: np.ndarray, y_pred: np.ndarray, targets: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Evaluate the performance of a machine learning model by calculating various metrics.

    Args:
        y_test (numpy.ndarray): The true labels of the test data.
        y_pred (numpy.ndarray): The predicted labels of the test data.
        y_labels (dict): A dictionary mapping target names to their corresponding labels.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the evaluation metrics
               for each target, including accuracy, F1 score, precision, and recall. The second dictionary
               contains the confusion matrices for each target.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    cm_list: Dict[str, np.ndarray] = {}
    target_index = 0
    for target_name in targets:
        cm = confusion_matrix(y_test[:, target_index], y_pred[:, target_index])
        accuracy = accuracy_score(
            y_test[:, target_index], y_pred[:, target_index]
        )
        f1 = f1_score(
            y_test[:, target_index],
            y_pred[:, target_index],
            average="weighted",
        )
        precision = precision_score(
            y_test[:, target_index],
            y_pred[:, target_index],
            average="macro",
            zero_division=0,
        )
        recall = recall_score(
            y_test[:, target_index], y_pred[:, target_index], average="macro"
        )

        # roc_auc = roc_auc_score(y_test[:, target_index], y_pred[:, target_index])
        metrics[target_name] = {
            "Accuracy": accuracy,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
            # "ROC-AUC": roc_auc
        }
        cm_list[target_name] = cm
        target_index += 1
    return metrics, cm_list


def evaluate_bootstrap(X_test, y_test, model, targets, n_times=10):
    bootstrap_metrics: Dict = {}
    for _ in range(n_times):
        X_test, y_test = resample(
            X_test, y_test, replace=True, random_state=42
        )
        y_pred = (
            np.atleast_2d(model.predict(X_test)).T
            if len(targets) == 1
            else model.predict(X_test)
        )
        metrics, _ = evaluate_model(y_test, y_pred, targets)
        for target in targets:
            if target not in bootstrap_metrics:
                bootstrap_metrics[target] = {
                    "Accuracy": [],
                    "F1": [],
                }
            bootstrap_metrics[target]["Accuracy"].append(
                metrics[target]["Accuracy"]
            )
            bootstrap_metrics[target]["F1"].append(metrics[target]["F1"])
    return bootstrap_metrics


def metrics_statistics(bootstrapped_metrics):
    """
    Do statistics with the bootsrapped metrics

    Args:
        dict: bootstrapped_metrics

    Returns:
        dict: Mean and 95% ci for the bootstrapped values for each target
    """
    metrics_stats = {}
    for key, value in bootstrapped_metrics.items():
        metrics_stats[key] = {
            "Accuracy_mean": None,
            "Accuracy_ci": None,
            "F1_mean": None,
            "F1_ci": None,
        }

        print(key)
        print(value["Accuracy"])

        # calc mean and 95% confidence interval for Accuracy
        metrics_stats[key]["Accuracy_mean"] = np.mean(value["Accuracy"])
        metrics_stats[key]["Accuracy_ci"] = st.t.interval(
            confidence=0.95,
            df=len(value["Accuracy"]) - 1,
            loc=np.mean(value["Accuracy"]),
            scale=st.sem(value["Accuracy"]),
        )

        # calc mean and 95% confidence interval for F1 score
        metrics_stats[key]["F1_mean"] = np.mean(value["F1"])
        metrics_stats[key]["F1_ci"] = st.t.interval(
            confidence=0.95,
            df=len(value["F1"]) - 1,
            loc=np.mean(value["F1"]),
            scale=st.sem(value["F1"]),
        )

    return metrics_stats
