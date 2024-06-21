"""Setting up and training a (bayesian) hyperparameter tuner with CV."""

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import cross_val_score

from ..models import load_model


class HyperparameterTuner:
    def __init__(
        self,
        model_name: str,
        model_config: dict,
        algo=tpe.suggest,
        max_evals=10,
    ):
        """
        Initialize the HyperparameterTuner class.

        Args:
            model_name (str): The name of the model.
            model_config (dict): The configuration of the model.
            algo (object, optional): The algorithm for hyperparameter tuning. Defaults to tpe.suggest.
            max_evals (int, optional): The maximum number of evaluations. Defaults to 10.
        """
        self.model_name = model_name
        self.model_config = model_config
        self.trials = Trials()
        self.max_evals = max_evals
        self.algo = algo

    def _objective(self, params: dict, X_train, y_train) -> dict:
        """
        Objective function for hyperparameter tuning.

        Args:
            params (dict): The hyperparameters to be tuned.
            X_train: The training data.
            y_train: The training labels.
            X_test: The testing data.
            y_test: The testing labels.

        Returns:
            dict: The loss and status of the objective function.
        """
        model = load_model(
            self.model_name, **{**params, **self.model_config["model_params"]}
        )
        model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # score = accuracy_score(y_test, y_pred)
        score = cross_val_score(model, X_train, y_train, cv=5).mean()
        return {"loss": -score, "status": STATUS_OK}

    def tune(self, X_train, y_train) -> tuple:
        """
        Perform hyperparameter tuning with hyperopt.

        Args:
            X_train: The training data.
            y_train: The training labels.
            X_test: The testing data.
            y_test: The testing labels.

        Returns:
            tuple: The best parameters and the tuning trials.
        """

        best = fmin(
            fn=lambda params: self._objective(params, X_train, y_train),
            space=self.model_config["hyperparameters"],
            algo=self.algo,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(42),
        )
        best_params = space_eval(self.model_config["hyperparameters"], best)
        return best_params, self.trials
