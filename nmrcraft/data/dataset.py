"""Load and preprocess data."""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from nmrcraft.utils.set_seed import set_seed

set_seed()


def filename_to_ligands(dataset: pd.DataFrame):
    """
    Extract ligands from the filename and add as columns to the dataset.
    Assumes that filenames are structured in a specific way that can be parsed into ligands.
    """
    filename_parts = dataset["file_name"].str.split("_", expand=True)
    dataset["metal"] = filename_parts.get(0)
    dataset["geometry"] = filename_parts.get(1)
    dataset["E_ligand"] = filename_parts.get(2)
    dataset["X1_ligand"] = filename_parts.get(3)
    dataset["X2_ligand"] = filename_parts.get(4)
    dataset["X3_ligand"] = filename_parts.get(5)
    dataset["X4_ligand"] = filename_parts.get(6)
    dataset["L_ligand"] = filename_parts.get(7).fillna(
        "none"
    )  # Fill missing L_ligand with 'none'
    return dataset


def load_dataset_from_hf(
    dataset_name: str = "NMRcraft/nmrcraft", data_files: str = "all_no_nan.csv"
):
    """
    Load the dataset and preprocess it with ligands extraction.
    """
    dataset = load_dataset(dataset_name, data_files=data_files)[
        "train"
    ].to_pandas()
    return dataset


class DataLoader:
    def __init__(
        self,
        dataset_name="NMRcraft/nmrcraft",
        data_files="all_no_nan.csv",
        feature_columns=None,
        target_column="metal",
        test_size=0.3,
        random_state=42,
        dataset_size=0.01,
    ):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.dataset = load_dataset_from_hf()

    def load_data(self):
        self.dataset = filename_to_ligands(
            self.dataset
        )  # Assuming filename_to_ligands is defined elsewhere
        self.dataset = self.dataset.sample(frac=self.dataset_size)
        return self.split_and_preprocess()

    def preprocess_features(self, X):
        """
        Apply standard normalization to the feature set.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    def split_and_preprocess(self):
        """
        Split data into training and test sets, then apply normalization.
        Ensures that the test data does not leak into training data preprocessing.
        """
        X = self.dataset[self.feature_columns].to_numpy()
        y_label = self.dataset[self.target_column].to_numpy()

        label_encoder = LabelEncoder()
        label_encoder.fit(["Mo", "W"])  # TODO adapt this for other targets!
        y = label_encoder.transform(y_label)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Normalize features with no leakage from test set
        X_train_scaled, scaler = self.preprocess_features(X_train)
        X_test_scaled = scaler.transform(
            X_test
        )  # Apply the same transformation to test set

        return X_train_scaled, X_test_scaled, y_train, y_test
