"""Functions to load a classifier."""

import inspect
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class InvalidModelNameError(ValueError):
    """Exception raised when the specified model name is not found."""

    def __init__(self, model_name, models):
        super().__init__(
            f"Model {model_name} not found. Available models are {list(models.keys())}"
        )


class InvalidArgumentError(ValueError):
    """Exception raised when an invalid argument is passed to the model constructor."""

    def __init__(self, kwarg, model_name):
        super().__init__(f"Invalid argument {kwarg} for model {model_name}")


def validate_model_availability(model_name: str, models: dict) -> None:
    """Ensure the model name exists in the provided models dictionary."""
    if model_name.lower() not in models:
        raise InvalidModelNameError(model_name, models)


def validate_kwargs(kwargs: dict, model_class: type, model_name: str) -> None:
    """Check that all kwargs are valid for the model class constructor."""
    args = inspect.signature(model_class.__init__).parameters.keys()
    for kwarg in kwargs:
        if kwarg not in args:
            raise InvalidArgumentError(kwarg, model_name)


def load_model(model_name: str, **kwargs: Any) -> Any:
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
        "extra_trees": ExtraTreesClassifier,
    }
    # TODO: put model config here

    # First, check if the model exists
    validate_model_availability(model_name, models)
    model_class = models[model_name.lower()]

    # Second, validate all provided kwargs before creating the model instance
    validate_kwargs(kwargs, model_class, model_name)

    # Third, adjust default parameters if necessary
    if model_name == "random_forest":
        kwargs.setdefault("n_jobs", -1)  # Set max number of jobs

    if model_name == "svc":
        kwargs["probability"] = True

    if model_name == "gpc":
        kwargs["multi_class"] = "one_vs_one"
        kwargs.setdefault("n_jobs", -1)  # Set max number of jobs

    if model_name == "extra_trees":
        kwargs.setdefault("n_jobs", -1)  # Set max number of jobs

    if model_name == "logistic_regression":
        kwargs.setdefault("n_jobs", -1)  # Set max number of jobs

    # Forth, validate all provided kwargs before creating the model instance
    validate_kwargs(kwargs, model_class, model_name)

    # Instantiate and return the model
    return model_class(**kwargs)
