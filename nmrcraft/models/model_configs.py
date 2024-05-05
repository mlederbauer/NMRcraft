from hyperopt import hp

model_configs = {
    "random_forest": {
        "model_params": {"random_state": 42},
        "hyperparameters": {
            "n_estimators": hp.choice("n_estimators", range(10, 1000, 10)), 
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            # "max_depth": hp.choice("max_depth", range(10, 1200, 10)),
            "min_samples_split": hp.uniform("min_samples_split", 0.01, 1.0),
            "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.5),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
        },
    },

    "gradient_boosting": {
        "model_params": {"random_state": 42},
        "hyperparameters": {
            "loss": hp.choice("loss", ["log_loss", "exponential"]),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
            "n_estimators": hp.choice("n_estimators", range(10, 1000, 10)),
            # "subsample": hp.uniform("subsample", 0.01, 1.0), 
            "criterion": hp.choice("criterion", ["friedman_mse", "squared_error"]),
            # "max_depth": hp.choice("max_depth", range(10, 1200, 10)),
            "min_samples_split": hp.uniform("min_samples_split", 0.01, 1.0),
            "min_samples_leaf": hp.uniform("min_samples_leaf", 0.01, 0.5),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
        },
    },
}
