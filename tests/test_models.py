import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from nmrcraft.models.models import load_model


def test_load_model():
    models = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression,
        "svc": SVC,
    }

    for model_name, model_class in models.items():
        model = load_model(model_name)
        assert isinstance(
            model, model_class
        ), f"Expected {model_class}, got {type(model)}"


def test_load_model_unsupported_model():
    with pytest.raises(ValueError):
        load_model("unsupported_model")


def test_load_model_unsupported_kwargs():
    with pytest.raises(TypeError):
        load_model("random_forest", "rainbows")
