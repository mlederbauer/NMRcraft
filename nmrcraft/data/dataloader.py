"""Load and preprocess data."""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from ..utils import set_seed
from .data_utils import (
    filename_to_ligands,
    load_dataset_from_hf,
    load_dummy_dataset_locally,
)

set_seed()

TARGET_TYPES = [
    "metal",
    "X1_ligand",
    "X2_ligand",
    "X3_ligand",
    "X4_ligand",
    "L_ligand",
    "E_ligand",
]


class DataLoader:
    """
    DataLoader is responsible for loading and preparing data for machine learning models
    in the `nmrcraft` project.

    It supports configuration of various dataset parameters including feature selection,
    target column specification, dataset size manipulation, and can return split datasets
    tuned for training and testing phases.

    Parameters:
    feature_columns (list of str): Names of columns to be used as features.
    target_columns (str): Name(s) of the column(s) used as targets.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Seed used by random number generator for reproducibility.
    dataset_size (float): Proportion of the full dataset to use.
    complex_geometry (str): Specifies the type of complex geometries to include ('oct', 'spy', 'tbp', or 'all').
    include_structural_features (bool): Indicates whether structural features should be included in the dataset.

    Returns:
    dataloader (DataLoader): dataloader object that is used to load and preprocess the dataset.
    Example:
    >>> data_loader = DataLoader(
        feature_columns=["M_sigma11_ppm", "M_sigma22_ppm"],
        target_columns="metal X4_ligand E_ligand",
        test_size=0.2,
        random_state=42,
        dataset_size=0.1,
        complex_geometry="all",
        include_structural_features=True
    )
    """

    def __init__(
        self,
        target_columns: str,
        dataset_size: float,
        include_structural_features: bool = False,
        complex_geometry: str = "oct",
        test_size: float = 0.2,
        random_state: int = 42,
        testing: bool = False,
        feature_columns=None,
    ):
        if feature_columns is None:
            feature_columns = [
                "M_sigma11_ppm",
                "M_sigma22_ppm",
                "M_sigma33_ppm",
                "E_sigma11_ppm",
                "E_sigma22_ppm",
                "E_sigma33_ppm",
            ]
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.random_state = random_state
        self.dataset_size = dataset_size
        self.target_columns = target_columns
        self.complex_geometry = complex_geometry
        self.include_structural_features = include_structural_features

        if not testing:
            self.dataset = load_dataset_from_hf()
        elif testing:
            self.dataset = load_dummy_dataset_locally()

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset, preprocesses it, and returns the preprocessed data.

        Returns:
            Preprocessed data (pandas.DataFrame): The preprocessed dataset.
        """
        self.dataset = filename_to_ligands(self.dataset)
        self.choose_geometry()
        return self.split_and_preprocess()

    def choose_geometry(self) -> None:
        """
        Filters the dataset based on the complex geometry.

        This method filters the dataset based on the complex geometry specified by the `complex_geometry` attribute.
        It checks if the specified geometry is valid and then updates the dataset accordingly. If the geometry is not
        valid, a `ValueError` is raised.

        Raises:
            ValueError: If the specified geometry is not valid.

        """
        valid_geometries = {"oct", "spy", "tbp"}
        if self.complex_geometry in valid_geometries:
            self.dataset = self.dataset[
                self.dataset["geometry"] == self.complex_geometry
            ]
        # else:
        #     raise ValueError("Invalid geometry'.") FIXME

    def encode_categorical_features(self) -> np.ndarray:
        """
        Encodes the categorical features in the dataset using LabelEncoder.

        Returns:
            np.ndarray: The encoded features in numpy array format.
        """
        # Select and extract the structural features from the dataset
        structural_features = (
            self.dataset[
                [col for col in TARGET_TYPES if col not in self.target_columns]
            ]
            .to_numpy()
            .T
        )  # Transpose immediately after conversion to numpy

        # Encode features using LabelEncoder and store encoders for potential inverse transform
        encoded_features = []
        self.encoders = []  # To store encoders for each feature
        for features in structural_features:
            encoder = LabelEncoder()
            encoder.fit(features)
            encoded_features.append(encoder.transform(features))
            self.encoders.append(encoder)

        # Convert the list of encoded features back to the original data structure
        return np.array(
            encoded_features
        ).T  # Transpose back to original orientation

    def encode_targets(self) -> Tuple[np.ndarray, dict]:
        """
        Encodes the target variables in the dataset using LabelEncoder.

        Returns:
            Tuple[np.ndarray, dict]: The encoded targets and a dictionary mapping target names to labels.
        """
        # Initialize lists to store encoded targets and corresponding encoders
        encoded_targets = []
        self.target_encoders = []
        y_labels_dict = {}

        # Encode each target column using LabelEncoder
        for target_name in self.target_columns:
            target = self.dataset[target_name].to_numpy()
            encoder = LabelEncoder()
            encoder.fit(target)
            encoded_targets.append(encoder.transform(target))
            self.target_encoders.append(encoder)
            y_labels_dict[
                target_name
            ] = (
                encoder.classes_.tolist()
            )  # Dictionary of labels for each target

        y_encoded = np.array(
            encoded_targets
        ).T  # Transpose to match original data structure
        return y_encoded, y_labels_dict

    def split_and_preprocess(
        self,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[str]]
    ]:
        """
        Split the dataset into training and testing sets, preprocess the data, and return the preprocessed data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[str]]]: A tuple containing the preprocessed training and testing data, encoded target variables, and readable labels.
        """
        # Extract and encode categorical features
        X_NMR = self.dataset[self.feature_columns].to_numpy()
        X_Structural = self.encode_categorical_features()

        # Encode target variables and store readable labels
        (
            y_encoded,
            y_labels,
        ) = self.encode_targets()

        # Split data into training and testing sets
        (
            X_train_NMR,
            X_test_NMR,
            X_train_Structural,
            X_test_Structural,
            y_train,
            y_test,
        ) = train_test_split(
            X_NMR,
            X_Structural,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # Further sample the training data to reduce its size
        train_size = int(len(y_train) * self.dataset_size)
        if train_size < 1:
            train_size = 1  # Ensure at least one sample

        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        indices = indices[:train_size]

        X_train_NMR = X_train_NMR[indices]
        X_train_Structural = X_train_Structural[indices]
        y_train = y_train[indices]

        # Scale numerical features (the NMR tensor)
        scaler = StandardScaler()
        X_train_NMR_scaled = scaler.fit_transform(X_train_NMR)
        X_test_NMR_scaled = scaler.transform(X_test_NMR)

        # Combine features if structural features are included
        if self.include_structural_features:
            X_train = np.concatenate(
                [X_train_NMR_scaled, X_train_Structural], axis=1
            )
            X_test = np.concatenate(
                [X_test_NMR_scaled, X_test_Structural], axis=1
            )
        else:
            X_train = X_train_NMR_scaled
            X_test = X_test_NMR_scaled

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            y_labels,
        )
