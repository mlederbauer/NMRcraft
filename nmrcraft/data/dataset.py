""" Load the data."""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# TODO split filename into ligand names and add them as columns to the dataset
def filename_to_ligands(dataset: pd.DataFrame):
    """
    Extract ligands from the filename and add as columns to the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing a 'file_name' column.

    Returns:
    - pd.DataFrame: The updated dataset with new columns for metal, geometry, E_ligand, and X_ligands.
    """
    # Split the 'file_name' column and expand into separate columns
    filename_parts = dataset["file_name"].str.split("_", expand=True)

    # Assign new columns based on the split parts
    # Using `.get()` with default as None to handle rows with less than 8 parts
    dataset["metal"] = filename_parts.get(0)
    dataset["geometry"] = filename_parts.get(1)
    dataset["E_ligand"] = filename_parts.get(2)
    dataset["X1_ligand"] = filename_parts.get(3)
    dataset["X2_ligand"] = filename_parts.get(4)
    dataset["X3_ligand"] = filename_parts.get(5)
    dataset["X4_ligand"] = filename_parts.get(6)
    dataset["L_ligand"] = filename_parts.get(7).fillna("none")  # Fill missing L_ligand with 'none'

    return dataset


def load_data(dataset_name: str = "NMRcraft/nmrcraft", data_files: str = "all_no_nan.csv"):
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
    ].to_pandas()  # Hack for now to get complete dataset as pandas df

    dataset = filename_to_ligands(dataset)

    return dataset


def split_data(X: np.array, y: np.array, test_size: float = 0.2):
    """
    Split the dataset into train and test sets.

    Parameters:
    - X (np.array): The input features.
    - y (np.array): The target variable.
    - test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    - np.array: The training features.
    - np.array: The test features.
    - np.array: The training target variable.
    - np.array: The test target variable.
    """
    # TODO potentially add more sophisticated splitting / bootstrapping methods
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test
