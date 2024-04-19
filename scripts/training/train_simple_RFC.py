import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming these are your custom modules
from nmrcraft.data.dataset import load_data
from nmrcraft.utils.set_seed import set_seed

set_seed()


def main():
    """Train a simple random forest model with MLflow tracking."""
    # Start an MLflow experiment
    mlflow.set_experiment("Random_Forest_Classification")

    with mlflow.start_run():
        # Load and preprocess data
        dataset = load_data()
        dataset = dataset.sample(frac=0.01)  # Only take 1% of the dataset

        # Extract features and target variables
        feature_columns = [
            "M_sigma11_ppm",
            "M_sigma22_ppm",
            "M_sigma33_ppm",
            "E_sigma11_ppm",
            "E_sigma22_ppm",
            "E_sigma33_ppm",
        ]
        X = dataset[feature_columns].to_numpy()
        y_label = dataset["metal"].to_numpy()

        # Transform the target variable using the label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(["Mo", "W"])
        y = label_encoder.transform(y_label)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        # Log parameters, metrics, and model
        mlflow.log_params({"n_estimators": 100, "test_size": 0.3, "random_state": 42, "sample_fraction": 0.01})
        mlflow.log_metric("accuracy", score)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {score}")


if __name__ == "__main__":
    main()
