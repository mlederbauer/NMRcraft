import pandas as pd


def add_rows_metrics(
    unified_metrics: pd.DataFrame,
    statistical_metrics: list,
    dataset_size,
    include_structural: bool,
    model_name: str,
    max_evals: int,
):
    """
    Compiles and adds a series of statistical metrics into a unified DataFrame, one row at a time.

    Args:
        statistical_metrics (list): List of lists containing metrics that match by their respective indices.
        dataset_size (int): Number of samples in the dataset.
        include_structural (bool): Indicates whether structural data was included in the analysis.
        model_name (str):  Identifier for the model that produced the metrics.
        max_evals (int): Number of evaluations conducted.

    Returns:
        unified_metrics (pd.DataFrame): DataFrame to append new statistical metrics.
    """

    for i in range(len(statistical_metrics[0])):
        new_row = [
            statistical_metrics[0][i],
            statistical_metrics[0],
            model_name,
            not include_structural,
            dataset_size,
            max_evals,
            statistical_metrics[1][i],
            statistical_metrics[2][i][0],
            statistical_metrics[2][i][1],
            statistical_metrics[3][i],
            statistical_metrics[4][i][0],
            statistical_metrics[4][i][1],
        ]
        unified_metrics.loc[len(unified_metrics)] = new_row
    return unified_metrics
