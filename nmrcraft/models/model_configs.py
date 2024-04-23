from hyperopt import hp

model_configs = {
    "random_forest": {
        "model_params": {"random_state": 42},
        "hyperparameters": {
            "n_estimators": hp.choice("n_estimators", range(10, 1000, 10)),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
            "max_depth": hp.choice("max_depth", range(10, 1200, 10)),
            "min_samples_split": hp.uniform("min_samples_split", 0.1, 1),
            "min_samples_leaf": hp.uniform("min_samples_leaf", 0.1, 0.5),
            "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
        },
    },
}
