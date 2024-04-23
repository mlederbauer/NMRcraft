import inspect
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class InvalidModelNameError(ValueError):
    def __init__(self, model_name, models):
        super().__init__(f"Model {model_name} not found. Available models are {list(models.keys())}")


class InvalidArgumentError(ValueError):
    def __init__(self, kwarg, model_name):
        super().__init__(f"Invalid argument {kwarg} for model {model_name}")


def validate_model_availability(model_name, models):
    if model_name.lower() not in models:
        raise InvalidModelNameError(model_name, models)


def validate_kwargs(kwargs, model_class, model_name):
    args = inspect.signature(model_class.__init__).parameters.keys()
    for kwarg in kwargs:
        if kwarg not in args:
            raise InvalidArgumentError(kwarg, model_name)


def load_model(model_name: str, **kwargs: Any):
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

    try:
        model_class = models.get(model_name.lower())
        validate_model_availability(model_name, models)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    if model_name == "random_forest":
        kwargs.setdefault("n_jobs", -1)  # Set max number of jobs

    # Get the arguments of the model constructor and check if they are valid
    try:
        model = model_class(**kwargs)
        validate_kwargs(kwargs, model_class, model_name)
    except Exception as e:
        print(f"Error with the passed arguments!: {e}")
        return None
    return model
