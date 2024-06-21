"""Load and preprocess data utils."""

import os

import pandas as pd
from datasets import load_dataset


class DatasetLoadError(FileNotFoundError):
    """Exeption raised when the Dataloader could not find data/dataset.csv,
    even after trying to generate it from huggingface"""

    def __init__(self, t):
        super().__init__(f"Could not load raw Dataset '{t}'")


class InvalidTargetError(ValueError):
    """Exception raised when the specified model name is not found."""

    def __init__(self, t):
        super().__init__(f"Invalid target '{t}'")


def filename_to_ligands(dataset: pd.DataFrame) -> pd.DataFrame:
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


def load_dummy_dataset_locally(
    dataset_path: str = "tests/data.csv",
) -> pd.DataFrame:
    """
    Load a dummy dataset from a local CSV file for testing purposes.

    Args:
        dataset_path (str, optional): The path to the CSV file containing the dataset. Defaults to "tests/data.csv".
    Returns:
        pandas.DataFrame: The dataset loaded from the CSV file.
    """
    dataset = pd.read_csv(dataset_path)
    return dataset


def load_dataset_from_hf(
    dataset_name: str = "NMRcraft/nmrcraft", data_files: str = "all_no_nan.csv"
) -> pd.DataFrame:
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
    if not os.path.isdir("dataset"):
        os.mkdir("dataset")
    # Check if hf dataset is already downloaded, else download it and then load it
    if not os.path.isfile("dataset/dataset.csv"):
        dataset = load_dataset(dataset_name, data_files=data_files)[
            "train"
        ].to_pandas()
        dataset.to_csv("dataset/dataset.csv")
    if os.path.isfile("dataset/dataset.csv"):
        dataset = pd.read_csv("dataset/dataset.csv")
    elif not os.path.isfile("dataset/dataset.csv"):
        raise DatasetLoadError(FileNotFoundError)
    return dataset
