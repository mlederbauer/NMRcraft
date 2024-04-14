from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# from nmrcraft.analysis.plotting import plot_predicted_vs_ground_truth
from nmrcraft.data.dataset import load_data, split_data
from nmrcraft.utils.set_seed import set_seed

set_seed()


def main():
    """Train a simple random forest model."""
    # Load the data
    dataset = load_data()

    # Only take 1% of the dataset
    dataset = dataset.sample(frac=0.01)

    # Extract features and target variables
    X = dataset[['M_sigma11_ppm', 'M_sigma22_ppm', 'M_sigma33_ppm', 'E_sigma11_ppm', 'E_sigma22_ppm', 'E_sigma33_ppm']].to_numpy()
    y_label = dataset['metal'].to_numpy()
    # TODO automatize feature selection with hydra for runs

    # Transform the target variable using the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(["Mo", "W"])
    y = label_encoder.transform(y_label)
    # TODO create pipeline

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create a random forest classifier
    model = RandomForestClassifier()
    # TODO optimize hyperparameters

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on the testing data
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    score = accuracy_score(y_test, y_pred)

    # Print the accuracy score
    print(f"Accuracy: {score}")
    # TODO uncertainty prediction?

    # Plot the classification results
    # title = r"M Classification from $\sigma_{11,E}$, $\sigma_{22,E}$, $\sigma_{33,E}$, $\sigma_{11,M}$, $\sigma_{22,M}$, and $\sigma_{33,M}$"
    # TODO implement plots for classification results


if __name__ == "__main__":
    main()
