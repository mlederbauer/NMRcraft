"""Load and preprocess data."""

import itertools

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from nmrcraft.utils.set_seed import set_seed

set_seed()


class InvalidTargetError(ValueError):
    """Exception raised when the specified model name is not found."""

    def __init__(self, t):
        super().__init__(f"Invalid target '{t}'")


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
    """Load the dataset.

    This function loads the dataset using the specified dataset name and data files.
    It assumes that you have logged into the Hugging Face CLI prior to calling this function.

    Args:
        dataset_name (str, optional): The name of the dataset. Defaults to "NMRcraft/nmrcraft".
        data_files (str, optional): The name of the data file. Defaults to 'all_no_nan.csv'.

    Returns:
        pandas.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    dataset = load_dataset(dataset_name, data_files=data_files)[
        "train"
    ].to_pandas()
    return dataset


def get_target_columns(target_columns: str):
    TARGET_TYPES = ["metal", "X1", "X2", "X3", "X4", "L", "E"]

    # Split the target string into individual targets
    targets = [t.strip() for t in target_columns.split("_")]

    # Check if the targets are valid
    for t in targets:
        if t not in TARGET_TYPES:
            raise InvalidTargetError(t)

    # Translate them into Dataframe Column names
    target_map = {
        "metal": "metal",
        "X1": "X1_ligand",
        "X2": "X2_ligand",
        "X3": "X3_ligand",
        "X4": "X4_ligand",
        "L": "L_ligand",
        "E": "E_ligand",
    }
    targets_transformed = [target_map[t] for t in targets]

    return targets_transformed


def get_target_labels(target_columns: str, dataset: pd.DataFrame):
    # Get unique values for each column
    unique_values = [list(set(dataset[col])) for col in target_columns]
    # Convert the list of sets to a list of lists
    return unique_values


# def get_target_labels(target_columns: str, dataset: pd.DataFrame):
#     # Get unique values for each column
#     unique_values = [set(dataset[col]) for col in target_columns]
#     # Convert the list of sets to a list of lists
#     result = [[i for i in s] for s in unique_values]
#     return result


def one_hot_encoding(y):
    if isinstance((y[0]), int):
        print("int")
    if isinstance((y[0]), tuple):
        max_values = (max(row) for row in zip(*y))
        print(max_values)


class DataLoader:
    def __init__(
        self,
        dataset_name="NMRcraft/nmrcraft",
        data_files="all_no_nan.csv",
        feature_columns=None,
        target_columns="metal",
        test_size=0.3,
        random_state=42,
        dataset_size=0.01,
    ):
        self.feature_columns = feature_columns
        self.target_columns = get_target_columns(target_columns=target_columns)
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
        target_unique_labels = get_target_labels(
            target_columns=self.target_columns, dataset=self.dataset
        )

        # Get the targets, rotate, apply encoding, rotate back
        y_labels_rotated = self.dataset[self.target_columns].to_numpy()
        y_labels = [
            list(x) if i == 0 else x
            for i, x in enumerate(map(list, zip(*y_labels_rotated)))
        ]
        self.target_unique_labels = target_unique_labels
        ys = []
        for i in range(len(target_unique_labels)):
            tmp_encoder = LabelEncoder()
            tmp_encoder.fit(target_unique_labels[i])
            ys.append(tmp_encoder.transform(y_labels[i]))
        y = list(zip(*ys))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # Make targets 1D if only one is targeted
        if len(y[0]) == 1:
            y_train = list(itertools.chain(*y_train))
            y_test = list(itertools.chain(*y_test))

        # Normalize features with no leakage from test set
        X_train_scaled, scaler = self.preprocess_features(X_train)
        X_test_scaled = scaler.transform(
            X_test
        )  # Apply the same transformation to test set

        return X_train_scaled, X_test_scaled, y_train, y_test
