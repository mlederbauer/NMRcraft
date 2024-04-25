from typing import Any, Dict, Tuple

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_curve,
)


def model_evaluation(
    model: BaseEstimator, X_test: Any, y_test: Any
) -> Tuple[Dict[str, float], Any, Any, Any]:
    """
    Evaluate the performance of the trained machine learning model.

    Args:
        model (BaseEstimator): The trained machine learning model.
        X_test (Any): The input features for testing.
        y_test (Any): The true labels for testing.

    Returns:
        Tuple[Dict[str, float], Any, Any, Any]: A tuple containing:
            - A dictionary with evaluation metrics (accuracy, f1_score, roc_auc).
            - The confusion matrix.
            - The false positive rate.
            - The true positive rate.
    """
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

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
