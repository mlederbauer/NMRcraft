from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nmrcraft.analysis import plot_predicted_vs_ground_truth
from nmrcraft.data import load_data, split_data
from nmrcraft.utils import set_seed

set_seed()


def main():
    """Train a simple random forest model."""
    # Load the data
    dataset = load_data()
    # only take 1% of the dataset
    dataset = dataset.sample(frac=0.01)

    # X is the column "E_sigmaiso_ppm" and "M_sigmaiso_ppm", y is the column "bond_length_ME"
    X = dataset[["E_sigmaiso_ppm", "M_sigmaiso_ppm"]].to_numpy()
    y = dataset["bond_length_M_E"].to_numpy()

    X_train, X_test, y_train, y_test = split_data(X, y)

    pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)

    print(f"R^2 score: {score}")

    # Plot the random numbers
    plot_predicted_vs_ground_truth(y_test, pipe.predict(X_test))


if __name__ == "__main__":
    main()
