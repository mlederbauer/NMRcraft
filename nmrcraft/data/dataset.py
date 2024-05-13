"""Load and preprocess data."""

import itertools
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)

from nmrcraft.utils.set_seed import set_seed

set_seed()


class DatasetLoadError(FileNotFoundError):
    """Exeption raised when the Dataloader could not find data/dataset.csv,
    even after trying to generate it from huggingface"""

    def __init__(self, t):
        super().__init__(f"Could not load raw Dataset '{t}'")


class InvalidTargetError(ValueError):
    """Exception raised when the specified model name is not found."""

    def __init__(self, t):
        super().__init__(f"Invalid target '{t}'")


class InvalidTargetTypeError(ValueError):
    """Exception raised when the specified target type is not valid."""

    def __init__(self, t):
        super().__init__(f"Invalid target Type '{t}'")


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


def load_dummy_dataset_locally(datset_path: str = "tests/data.csv"):
    dataset = pd.read_csv(datset_path)
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
    # Create data dir if needed
    if not os.path.isdir("data"):
        os.mkdir("data")
    # Check if hf dataset is already downloaded, else download it and then load it
    if not os.path.isfile("data/dataset.csv"):
        dataset = load_dataset(dataset_name, data_files=data_files)[
            "train"
        ].to_pandas()
        dataset.to_csv("data/dataset.csv")
    if os.path.isfile("data/dataset.csv"):
        dataset = pd.read_csv("data/dataset.csv")
    elif not os.path.isfile("data/dataset.csv"):
        raise DatasetLoadError(FileNotFoundError)
    return dataset


def get_target_columns(target_columns: str):
    """
    Function takes target columns in underline format f.e 'metal_X1_X4_X2_L' and
    transforms into a list of the column names present in the dataset.
    """
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


def get_structural_feature_columns(target_columns: list):
    """
    Function gets the feature columns given the target columns. The feature columns are those that will be in the X set.
    """
    TARGET_TYPES = [
        "metal",
        "X1_ligand",
        "X2_ligand",
        "X3_ligand",
        "X4_ligand",
        "L_ligand",
        "E_ligand",
    ]

    # Get the features as the not targets
    features = [x for x in TARGET_TYPES if x not in target_columns]

    return features


def get_target_labels(target_columns: str, dataset: pd.DataFrame):
    # Get unique values for each column
    unique_values = [list(set(dataset[col])) for col in target_columns]
    # Convert the list of sets to a list of lists
    return unique_values


def target_label_readabilitizer(readable_labels):
    """
    function takes in the classes from the binarzier and turns them into human readable list of same length of the target.
    """
    # Trun that class_ into list
    human_readable_label_list = list(itertools.chain(*readable_labels))
    # Handle Binarized metal stuff and make the two columns become a single one
    for i in enumerate(human_readable_label_list):
        if (
            human_readable_label_list[i[0]] == "Mo"
            and human_readable_label_list[i[0] + 1] == "W"
        ) or (
            human_readable_label_list[i[0]] == "W"
            and human_readable_label_list[i[0] + 1] == "Mo"
        ):
            human_readable_label_list[i[0]] = "Mo W"
            human_readable_label_list.pop(i[0] + 1)

    return human_readable_label_list


def target_label_readabilitizer_categorical(target_labels):
    good_labels = []
    for label_array in target_labels:
        good_labels.append(list(label_array))
    return good_labels


class DataLoader:
    def __init__(
        self,
        dataset_name="NMRcraft/nmrcraft",
        data_files="all_no_nan.csv",
        feature_columns=None,
        target_columns="metal",
        target_type="one-hot",  # can be "categorical" or "one-hot"
        test_size=0.3,
        random_state=42,
        dataset_size=0.01,
        testing=False,
    ):
        self.feature_columns = feature_columns
        self.target_columns = get_target_columns(target_columns=target_columns)
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.target_type = target_type
        if not testing:
            self.dataset = load_dataset_from_hf()
        elif testing:
            self.dataset = load_dummy_dataset_locally()

    def load_data(self):
        self.dataset = filename_to_ligands(self.dataset)
        self.dataset = self.dataset.sample(frac=self.dataset_size)
        if self.target_type == "categorical":
            return self.split_and_preprocess_categorical()
        elif (
            self.target_type == "one-hot"
        ):  # Target is binarized and Features are one hot
            return self.split_and_preprocess_one_hot()
        else:
            raise InvalidTargetTypeError(ValueError)

    def preprocess_features(self, X):
        """
        Apply standard normalization to the feature set.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler

    def split_and_preprocess_categorical(self):
        """
        Split data into training and test sets, then apply normalization.
        Ensures that the test data does not leak into training data preprocessing.
        X and y are categorical, so each column has a integer that defines which one of the ligands is in the column.
        """
        # Get NMR and structural Features and combine
        X_NMR = self.dataset[self.feature_columns].to_numpy()
        X_Structural_Features_Columns = get_structural_feature_columns(
            target_columns=self.target_columns
        )
        X_Structural_Features = self.dataset[
            X_Structural_Features_Columns
        ].to_numpy()
        X_Structural_Features = [
            list(x) if i == 0 else x
            for i, x in enumerate(map(list, zip(*X_Structural_Features)))
        ]
        self.feature_unique_labels = get_target_labels(
            dataset=self.dataset, target_columns=X_Structural_Features_Columns
        )
        xs = []
        for i in range(len(self.feature_unique_labels)):
            tmp_encoder = LabelEncoder()
            tmp_encoder.fit(self.feature_unique_labels[i])
            xs.append(tmp_encoder.transform(X_Structural_Features[i]))
        X_Structural_Features = list(zip(*xs))

        # Get the targets, rotate, apply encoding, rotate back
        target_unique_labels = get_target_labels(
            target_columns=self.target_columns, dataset=self.dataset
        )
        y_labels_rotated = self.dataset[self.target_columns].to_numpy()
        y_labels = [
            list(x) if i == 0 else x
            for i, x in enumerate(map(list, zip(*y_labels_rotated)))
        ]
        self.target_unique_labels = target_unique_labels
        ys = []
        readable_labels = []
        for i in range(len(target_unique_labels)):
            tmp_encoder = LabelEncoder()
            tmp_encoder.fit(target_unique_labels[i])
            ys.append(tmp_encoder.transform(y_labels[i]))
            readable_labels.append(tmp_encoder.classes_)
        y = list(zip(*ys))
        (
            X_NMR_train,
            X_NMR_test,
            X_train_structural,
            X_test_structural,
            y_train,
            y_test,
        ) = train_test_split(
            X_NMR,
            X_Structural_Features,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # Make targets 1D if only one is targeted
        if len(y[0]) == 1:
            y_train = list(itertools.chain(*y_train))
            y_test = list(itertools.chain(*y_test))

        # Normalize features with no leakage from test set
        X_train_NMR_scaled, scaler = self.preprocess_features(X_NMR_train)
        X_test_NMR_scaled = scaler.transform(
            X_NMR_test
        )  # Apply the same transformation to test set
        X_train_scaled = np.concatenate(
            [X_train_NMR_scaled, X_train_structural], axis=1
        )
        X_test_scaled = np.concatenate(
            [X_test_NMR_scaled, X_test_structural], axis=1
        )

        # Get the target labels going
        y_label = target_label_readabilitizer_categorical(readable_labels)
        return X_train_scaled, X_test_scaled, y_train, y_test, y_label

    def split_and_preprocess_one_hot(self):
        """
        Split data into training and test sets, then apply normalization.
        Ensures that the test data does not leak into training data preprocessing. Returned X is one-hot encoded and y binarized using the sklearn functions.
        """
        target_unique_labels = get_target_labels(
            target_columns=self.target_columns, dataset=self.dataset
        )

        # Get the Targets, rotate, apply binarization, funze into a single array
        y_labels_rotated = self.dataset[self.target_columns].to_numpy()
        y_labels = [
            list(x) if i == 0 else x
            for i, x in enumerate(map(list, zip(*y_labels_rotated)))
        ]
        self.target_unique_labels = target_unique_labels
        ys = []
        readable_labels = []
        for i in range(len(target_unique_labels)):
            LBiner = LabelBinarizer()
            ys.append(LBiner.fit_transform(y_labels[i]))
            readable_labels.append(LBiner.classes_)
        y = np.concatenate(list(ys), axis=1)

        # Get NMR and structural Features, one-hot-encode and combine
        X_NMR = self.dataset[self.feature_columns].to_numpy()
        X_Structural_Features_Columns = get_structural_feature_columns(
            self.target_columns
        )
        X_Structural_Features = self.dataset[
            X_Structural_Features_Columns
        ].to_numpy()
        one_hot = OneHotEncoder().fit(X_Structural_Features)
        X_Structural_Features_enc = one_hot.transform(
            X_Structural_Features
        ).toarray()
        # X = [X_NMR, X_Structural_Features_enc]
        # print(X)

        # Split the datasets
        (
            X_train_NMR,
            X_test_NMR,
            X_train_structural,
            X_test_structural,
            y_train,
            y_test,
        ) = train_test_split(
            X_NMR,
            X_Structural_Features_enc,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        # Make targets 1D if only one is targeted
        if len(y[0]) == 1:
            y_train = list(itertools.chain(*y_train))
            y_test = list(itertools.chain(*y_test))

        # Normalize features with no leakage from test set
        X_train_NMR_scaled, scaler = self.preprocess_features(X_train_NMR)
        X_test_NMR_scaled = scaler.transform(
            X_test_NMR
        )  # Apply the same transformation to test set
        # Combine scaled NMR features with structural features
        X_train_scaled = np.concatenate(
            [X_train_NMR_scaled, X_train_structural], axis=1
        )
        X_test_scaled = np.concatenate(
            [X_test_NMR_scaled, X_test_structural], axis=1
        )

        # Creates the labels that can be used to identify the targets in the binaized y-array
        good_target_labels = target_label_readabilitizer(readable_labels)

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            good_target_labels,
        )
