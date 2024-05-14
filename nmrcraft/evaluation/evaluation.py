import os
from typing import Any, Dict, Tuple

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)

from nmrcraft.data import dataset


def model_evaluation(
    model: BaseEstimator,
    X_test: Any,
    y_test: Any,
    y_labels: Any,
    dataloader: dataset.DataLoader,
) -> Tuple[Dict[str, float], Any, Any, Any]:
    """
    Evaluate the performance of the trained machine learning model for 1D targets.

    Args:
        model (BaseEstimator): The trained machine learning model.
        X_test (Any): The input features for testing.
        y_test (Any): The true labels for testing.
        y_labels (Any): Label for the columns of the target.
        dataloader (DataLoader): Dataloader to decode the target arrays.

    Returns:
        Tuple[Dict[str, float], Any, Any, Any]: A tuple containing:
            - A dictionary with evaluation metrics (accuracy, f1_score, roc_auc).
            - The confusion matrix.
            - The false positive rate.
            - The true positive rate.
    """
    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    y_test_cm = dataloader.confusion_matrix_data_adapter(y_test)
    y_pred_cm = dataloader.confusion_matrix_data_adapter(y_pred)
    y_labels_cm = dataloader.confusion_matrix_label_adapter(y_labels)
    cm = confusion_matrix(
        y_pred=y_pred_cm, y_true=y_test_cm, labels=y_labels_cm
    )
    return (
        {
            "accuracy": score,
            "f1_score": f1,
            "roc_auc": roc_auc,
        },
        cm,
        fpr,
        tpr,
    )


def model_evaluation_nD(
    model: BaseEstimator,
    X_test: Any,
    y_test: Any,
    y_labels: Any,
    dataloader: dataset.DataLoader,
) -> Tuple[Dict[str, float], Any, Any, Any]:
    """
    Evaluate the performance of the trained machine learning model for 2D+ Targets.

    Args:
        model (BaseEstimator): The trained machine learning model.
        X_test (Any): The input features for testing.
        y_test (Any): The true labels for testing.
        y_labels (Any): Label for the columns of the target.
        dataloader (DataLoader): Dataloader to decode the target arrays.

    Returns:
        Tuple[Dict[str, float], Any]: A tuple containing:
            - A dictionary with evaluation metrics (accuracy, f1_score).
            - The confusion matrix.
    """
    y_pred = model.predict(X_test)

    y_test_cm = dataloader.confusion_matrix_data_adapter(y_test)
    y_pred_cm = dataloader.confusion_matrix_data_adapter(y_pred)
    y_labels_cm = dataloader.confusion_matrix_label_adapter(y_labels)
    score = accuracy_score(y_test_cm, y_pred_cm)
    f1 = f1_score(y_test_cm, y_pred_cm, average="weighted")
    cm = confusion_matrix(
        y_pred=y_pred_cm, y_true=y_test_cm, labels=y_labels_cm
    )
    return (
        {
            "accuracy": score,
            "f1_score": f1,
        },
        cm,
    )


def get_cm_path():
    fig_path = "scratch/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    return os.path.join(fig_path, "cm.png")


def get_roc_path():
    fig_path = "scratch/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    return os.path.join(fig_path, "roc.png")
