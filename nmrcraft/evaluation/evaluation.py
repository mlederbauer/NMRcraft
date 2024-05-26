from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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
