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


def transpose(array: any):
    """rotate/transpose array to the right"""
    ar = array[:]  # make copy just to be sure
    ar = [  # rotate the array to the right
        list(x) if i == 0 else x for i, x in enumerate(map(list, zip(*ar)))
    ]
    return ar


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


def target_label_readabilitizer(readable_labels):
    """
    function takes in the classes from the binarzier and turns them into human readable list of same length of the target.
    """
    # Trun that class_ into list
    human_readable_label_list = list(itertools.chain(*readable_labels))
    # Handle Binarized metal stuff and make the two columns become a single one because the metals get turned into a single column by the binarizer
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


def column_length_to_indices(column_lengths):
    indices = []
    start_index = 0
    for length in column_lengths:
        if length == 1:
            indices.append([start_index])
        else:
            indices.append(list(range(start_index, start_index + length)))
        start_index += length
    return indices


class DataLoader:
    def __init__(
        self,
        dataset_name="NMRcraft/nmrcraft",
        data_files="all_no_nan.csv",
        feature_columns=None,
        target_columns="metal",
        target_type="one-hot",  # can be "categorical" or "one-hot",
        complex_geometry="all",
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
        self.complex_geometry = complex_geometry

        if not testing:
            self.dataset = load_dataset_from_hf()
        elif testing:
            self.dataset = load_dummy_dataset_locally()

    def load_data(self):
        self.dataset = filename_to_ligands(self.dataset)
        self.dataset = self.dataset.sample(frac=self.dataset_size)
        self.choose_geometry()
        if self.target_type == "categorical":
            return self.split_and_preprocess_categorical()
        elif (
            self.target_type == "one-hot"
        ):  # Target is binarized and Features are one hot
            return self.split_and_preprocess_one_hot()
        else:
            raise InvalidTargetTypeError(ValueError)

    def choose_geometry(self):
        """
        Reduce the dataset down to a certain geometry if a valid
        one was passed, else just leave it as is.
        """
        if self.complex_geometry == "oct":
            self.dataset = self.dataset[
                self.dataset["geometry"] == "oct"
            ]  # only load octahedral complexes
        elif self.complex_geometry == "spy":
            self.dataset = self.dataset[
                self.dataset["geometry"] == "spy"
            ]  # only load square pyramidal complexes
        elif self.complex_geometry == "tbp":
            self.dataset = self.dataset[
                self.dataset["geometry"] == "tbp"
            ]  # only load trigonal bipyramidal complexes

    def scale(self, X):
        """
        Apply standard normalization to the feature set.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def get_target_columns_separated(self):
        """Returns the column indicies of the target array nicely sorted.
        For example: metal_X1: [[0, 1], [1, 2, 3, 4]]"""
        if (
            "metal" in self.target_columns
        ):  # If targets have metal, do weird stuff
            metal_index = self.target_columns.index("metal")
            y_column_indices = column_length_to_indices(
                self.target_column_numbers
            )
            for i in range(len(y_column_indices)):
                if i == metal_index:
                    y_column_indices[i].append(y_column_indices[i][0] + 1)
                if i > metal_index:
                    y_column_indices[i] = [x + 1 for x in y_column_indices[i]]

        elif "metal" not in self.target_columns:
            y_column_indices = column_length_to_indices(
                self.target_column_numbers
            )
        return y_column_indices

    def more_than_one_target(self):
        """Function returns true if more than one target is specified"""
        return len(self.target_columns) > 1

    def categorical_target_decoder(self, y):
        """
        function takes in the  target (y) array and transforms it back to decoded form.
        For this function to be run the split_and_preprocess_categorical already has to have been run beforehand.
        """
        ys = y[:]  # copy y so it's not modified
        target_encoders = self.target_label_encoders
        ys_decoded = []
        ys = transpose(ys)

        # Decode columnwise
        for i, target_column in enumerate(ys):
            ys_decoded.append(
                target_encoders[i].inverse_transform(target_column)
            )

        # Rotate back so each row corresponds to a complex and not the target like metal or X4
        ys_decoded_properly_rotated = [
            list(x) if i == 0 else x
            for i, x in enumerate(map(list, zip(*ys_decoded)))
        ]

        return np.array(ys_decoded_properly_rotated)

    def binarized_target_decoder(self, y):
        """
        function takes in the  target (y) array and transforms it back to decoded form.
        For this function to be run the one-hot-preprocesser already has to have been run beforehand.
        """
        y_column_indices = column_length_to_indices(self.target_column_numbers)
        ys = []
        ys_decoded = []
        # Split up compressed array into the categories
        for i in range(len(y_column_indices)):
            ys.append(y[:, y_column_indices[i]])

        # Decode the binarized categries using the original binarizers
        for i in range(len(ys)):
            ys_decoded.append(self.encoders[i].inverse_transform(ys[i]))

        # Rotate the array
        ys_decoded_properly_rotated = [
            list(x) if i == 0 else x
            for i, x in enumerate(map(list, zip(*ys_decoded)))
        ]
        return ys_decoded_properly_rotated

    def confusion_matrix_data_adapter_categorical(self, y):
        """
        Takes in binary encoded target array and returns decoded flat list.
        Especially designed to work with confusion matrix.
        """
        y_decoded = self.categorical_target_decoder(y)
        flat_y_decoded = [y for ys in y_decoded for y in ys]
        return flat_y_decoded

    def confusion_matrix_data_adapter_one_hot(self, y):
        """
        Takes in binary encoded target array and returns decoded flat list.
        Especially designed to work with confusion matrix.
        """
        y_decoded = self.binarized_target_decoder(y)
        flat_y_decoded = [y for ys in y_decoded for y in ys]
        return flat_y_decoded

    def confusion_matrix_label_adapter(self, y_labels):
        y_labels_copy = y_labels[:]
        for i in range(len(y_labels)):
            if y_labels_copy[i] == "Mo W":
                y_labels_copy[i] = "Mo"
                y_labels_copy.insert(i, "W")
        return y_labels_copy

    def categorical_endocode_X(self):
        # Get NMR Featrues (passed ones) and structural Features
        X_Structural_Features_Columns = get_structural_feature_columns(
            target_columns=self.target_columns
        )
        X_Structural_Features = self.dataset[
            X_Structural_Features_Columns
        ].to_numpy()

        # Transpose the array
        X_Structural_Features = transpose(X_Structural_Features)

        # Target-wise encoding with Label encoder and save encoders for later decoding
        xs = []
        for i in range(len(X_Structural_Features)):
            tmp_encoder = LabelEncoder()
            tmp_encoder.fit(X_Structural_Features[i])
            xs.append(tmp_encoder.transform(X_Structural_Features[i]))
        X_Structural_Features = list(zip(*xs))  # Kind of backtransposing

        return X_Structural_Features

    def categorical_endocode_y(self):
        # Get the targets
        y_labels_rotated = self.dataset[self.target_columns].to_numpy()

        # rotate the list of list (array-like)
        y_labels = transpose(y_labels_rotated)

        # Do targetwise encoding using the label encoder and save the label encoders for later decoding
        ys = []
        self.target_label_encoders = []
        readable_labels = []
        for i in range(len(y_labels)):
            tmp_encoder = LabelEncoder()
            tmp_encoder.fit(y_labels[i])
            ys.append(tmp_encoder.transform(y_labels[i]))
            self.target_label_encoders.append(tmp_encoder)
            readable_labels.append(tmp_encoder.classes_)
        # Combine y
        y = np.array(list(zip(*ys)))
        # Return y fuzed into a single array and y_labels
        return y, readable_labels

    def one_hot_endocode_X(self):
        """
        Method that does the one-hot encoding of the DataLoader's features
        based on the selected targets
        """
        # Get Columns corresponding to the features that are selected
        X_Structural_Features_Columns = get_structural_feature_columns(
            self.target_columns
        )

        # Get the features based on the selected columns
        X_Structural_Features = self.dataset[
            X_Structural_Features_Columns
        ].to_numpy()

        # One hot encode X structural
        X_Structural_Features_enc = (
            OneHotEncoder().fit_transform(X_Structural_Features).toarray()
        )

        return X_Structural_Features_enc

    def label_binarize_endocode_y(self):

        # Get the Targets and transpose
        y_labels_rotated = self.dataset[self.target_columns].to_numpy()
        y_labels = transpose(y_labels_rotated)

        ys = []
        readable_labels = []
        self.encoders = []
        self.target_column_numbers = []

        # Binarize targetwise and save encoders and labels
        for i in range(len(y_labels)):
            # Encode
            label_binerizer = LabelBinarizer()
            ys.append(label_binerizer.fit_transform(y_labels[i]))

            # Save stuff for later decoding
            readable_labels.append(label_binerizer.classes_)
            self.encoders.append(
                label_binerizer
            )  # save encoder for later decoding
            self.target_column_numbers.append(
                len(ys[i][0])
            )  # save column numbers for later decoding

        # Return y fuzed into a single array and labels
        y = np.concatenate(list(ys), axis=1)
        return y, readable_labels

    def split_and_preprocess_categorical(self):
        """
        Split data into training and test sets, then apply normalization.
        Ensures that the test data does not leak into training data preprocessing.
        X and y are categorical, so each column has a integer that defines which one of the ligands is in the column.
        """

        # Get NMR features
        X_NMR = self.dataset[self.feature_columns].to_numpy()

        # Encode X in a categorical fashion with the label encoder columnwise
        X_Structural_Features = self.categorical_endocode_X()

        # Encode y in a categorical fashion with the label encoder columnwise
        y, readable_labels = self.categorical_endocode_y()

        # Train Test splitting
        (
            X_train_NMR,
            X_test_NMR,
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

        # Normalize features with no leakage from test set
        X_train_NMR_scaled = self.scale(X_train_NMR)
        X_test_NMR_scaled = self.scale(X_test_NMR)

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
        # Get NMR features
        X_NMR = self.dataset[self.feature_columns].to_numpy()

        # Get structural features one-hot encoded
        X_Structural_Features_enc = self.one_hot_endocode_X()

        # Get structural targets, binarized
        y, readable_labels = self.label_binarize_endocode_y()

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

        # Normalize features with no leakage from test set
        X_train_NMR_scaled = self.scale(X_train_NMR)
        X_test_NMR_scaled = self.scale(X_test_NMR)

        # Combine scaled NMR features with structural features to get final X
        X_train_scaled = np.concatenate(
            [X_train_NMR_scaled, X_train_structural], axis=1
        )
        X_test_scaled = np.concatenate(
            [X_test_NMR_scaled, X_test_structural], axis=1
        )

        # Creates the labels that can be used to identify the targets in the binaized y-array
        # (basicall handle special metal behaviour)
        good_target_labels = target_label_readabilitizer(readable_labels)

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            good_target_labels,
        )
