from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def load_model(model_name, **kwargs):
    """
    Load a model dynamically based on the model_name argument.

    Args:
    - model_name (str): The name of the model to load.
    - kwargs (dict): Additional keyword arguments for model initialization.

    Returns:
    - An instance of the specified model.
    """
    models = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "svc": SVC,
    }

    model_class = models.get(model_name.lower())
    # TODO: add exceptions and more models
    return model_class(**kwargs)
