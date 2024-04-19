from sklearn.model_selection import GridSearchCV


def hyperparameter_tune(model, X_train, y_train):
    """Simple hyperparameter tuning using GridSearchCV."""
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
    # TODO: make more model agnostic
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
