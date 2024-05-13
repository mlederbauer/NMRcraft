from typing import Any, Dict, Tuple

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
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
    Evaluate the performance of the trained machine learning model.

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
    y_test_cm = dataloader.confusion_matrix_data_adapter(y_test)
    y_pred_cm = dataloader.confusion_matrix_data_adapter(y_pred)
    y_labels_cm = dataloader.confusion_matrix_label_adapter(y_labels)
    print(y_labels_cm)
    print(y_pred_cm)
    print(y_labels)
    print(y_pred)
    cm = confusion_matrix(
        y_pred=y_pred_cm, y_true=y_test_cm, labels=y_labels_cm
    )
    # fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    fpr = tpr = roc_auc = False
    # roc_auc = auc(fpr, tpr)

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
