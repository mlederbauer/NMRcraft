import argparse
import os

import mlflow
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nmrcraft.analysis.plotting import plot_confusion_matrix, plot_roc_curve

# Assuming these are your custom modules
from nmrcraft.data.dataset import load_data
from nmrcraft.models.hyperparameter_tune import hyperparameter_tune
from nmrcraft.models.models import load_model
from nmrcraft.utils.set_seed import set_seed

set_seed()


def main(dataset_size, target, model_name):
    """Train a model with MLflow tracking."""
    mlflow.set_experiment("Model_Classification_Experiment")

    with mlflow.start_run():
        # Load and preprocess data
        dataset = load_data()
        dataset = dataset.sample(frac=dataset_size)

        feature_columns = [
            "M_sigma11_ppm",
            "M_sigma22_ppm",
            "M_sigma33_ppm",
            "E_sigma11_ppm",
            "E_sigma22_ppm",
            "E_sigma33_ppm",
        ]
        # TODO: add feature columns if we want to add other features

        X = dataset[feature_columns].to_numpy()
        y_label = dataset[target].to_numpy()

        label_encoder = LabelEncoder()
        label_encoder.fit(["Mo", "W"])  # TODO: change if we have other targets
        y = label_encoder.transform(y_label)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = load_model(model_name, n_estimators=100, random_state=42)
        model = hyperparameter_tune(model, X_train, y_train)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        mlflow.log_params({"model_name": model_name, "dataset_size": dataset_size, "target": target})
        mlflow.log_metrics({"accuracy": score, "f1_score": f1, "roc_auc": roc_auc})

        mlflow.sklearn.log_model(model, "model")

        fig_path = "scratch/"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        cm_path = os.path.join(fig_path, "cm.png")
        title = r"Confusion matrix, TODO add LaTeX symbols"
        plot_confusion_matrix(cm, classes=label_encoder.classes_, title=title, path=cm_path)
        roc_path = os.path.join(fig_path, "roc.png")
        title = r"ROC curve, TODO add LaTeX symbols"
        plot_roc_curve(fpr, tpr, roc_auc, title=title, path=roc_path)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)

        # os.remove(cm_path)
        # os.remove(roc_path)

        print(f"Accuracy: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with MLflow tracking.")
    parser.add_argument("--dataset_size", type=float, default=0.01, help="Fraction of dataset to use")
    parser.add_argument("--target", type=str, default="metal", help="Target variable name")
    parser.add_argument("--model_name", type=str, default="random_forest", help="Model name to load")
    args = parser.parse_args()

    main(args.dataset_size, args.target, args.model_name)
