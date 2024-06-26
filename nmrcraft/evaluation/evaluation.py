"""Functions to evaluate and bootstrap a model."""

from typing import Dict, List, Tuple, Union

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


def evaluate_bootstrap(
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: object,
    targets: List[str],
    n_times: int = 10,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Perform bootstrap evaluation of a model on test data.

    This function repeatedly samples with replacement from the test dataset and evaluates
    the model on these samples. It aggregates the performance metrics across all bootstrap
    samples to give a robust estimate of the model's generalizability.

    Args:
        X_test (np.ndarray): The input features of the test data.
        y_test (np.ndarray): The true labels of the test data.
        model (object): The model that is being evaluated.
        targets (List[str]): A list of target variable names.
        n_times (int, optional): The number of bootstrap samples to generate.

    Returns:
        Dict[str, Dict[str, List[float]]]: A dictionary containing the computed metrics
        for each target. Each target's value is another dictionary containing lists
        of performance scores ('Accuracy' and 'F1') across the bootstrap samples.
    """
    bootstrap_metrics: Dict[str, Dict[str, List[float]]] = {}
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


def metrics_statistics(
    bootstrapped_metrics: Dict[str, Dict[str, List[float]]],
) -> List[Union[List[str], List[float], List[Tuple[float, float]]]]:
    """Calculate the statistical summary of bootstrapped evaluation metrics with F1 score and Accuracy.

    Args:
        bootstrapped_metrics (dict): A dictionary containing the name of each target with another dictionary
        as value, which includes values of the F1 scores and Accuracies of the bootstrapped models.

    Returns:
        list: A list containing five elements:
            - [0]: List of target names for which metrics are calculated.
            - [1]: List of mean accuracies for each target.
            - [2]: List of tuples where each tuple consists of the lower and upper bounds of the 95% confidence interval for accuracy for each target.
            - [3]: List of mean F1 scores for each target.
            - [4]: List of tuples where each tuple consists of the lower and upper bounds of the 95% confidence interval for F1 score for each target.

        Each element in the list corresponds to a specific set of statistical values related to the performance metrics (accuracy and F1 score) of the bootstrapped models for each target.
    """
    Targets: List[str] = []
    Accuracy_mean: List[float] = []
    Accuracy_ci: List[Tuple[float, float]] = []
    F1_mean: List[float] = []
    F1_ci: List[Tuple[float, float]] = []

    for target, value in bootstrapped_metrics.items():
        Targets.append(target)

        # Calculate mean and 95% confidence interval for Accuracy
        Accuracy_mean.append(np.mean(value["Accuracy"]))
        Accuracy_ci.append(
            st.t.interval(
                confidence=0.95,
                df=len(value["Accuracy"]) - 1,
                loc=np.mean(value["Accuracy"]),
                scale=st.sem(value["Accuracy"]),
            )
        )

        # Calculate mean and 95% confidence interval for F1 score
        F1_mean.append(np.mean(value["F1"]))
        F1_ci.append(
            st.t.interval(
                confidence=0.95,
                df=len(value["F1"]) - 1,
                loc=np.mean(value["F1"]),
                scale=st.sem(value["F1"]),
            )
        )
    metrics_stats: List[
        Union[List[str], List[float], List[Tuple[float, float]]]
    ] = [Targets, Accuracy_mean, Accuracy_ci, F1_mean, F1_ci]

    return metrics_stats
