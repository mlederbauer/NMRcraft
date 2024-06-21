"""General utils for dataframe and type transformation."""

import pandas as pd


def add_rows_metrics(
    unified_metrics: pd.DataFrame,
    statistical_metrics: list,
    dataset_size,
    include_structural: bool,
    model_name: str,
    max_evals: int = 0,
) -> pd.DataFrame:
    """
    Compiles and adds a series of statistical metrics into a unified DataFrame, one row at a time.

    Args:
        statistical_metrics (list): List of lists containing the mean and confidence intervals of
        accuracy and F1-score.
        dataset_size (int): Number of samples in the dataset.
        include_structural (bool): Indicates whether structural data was included in the analysis.
        model_name (str):  Name of the model that produced the metrics.
        max_evals (int): Number of evaluations conducted in the Hyperparameter tuning.

    Returns:
        unified_metrics (pd.DataFrame): DataFrame with all metrics containing these columns:
            target, model_targets, model, nmr_only, dataset_fraction, max_evals, accuracy_mean,
            accuracy_lb, accuracy_hb, f1_mean, f1_lb, f1_hb

    """
    # Give meaning to indices
    idx_name = 0
    idx_accuracy_mean = 1
    idx_accuracy_ci = 2
    idx_f1score_mean = 3
    idx_f1score_ci = 4
    idx_lb = 0
    idx_hb = 1

    # Combine all data into single row and append to dataframe
    for i in range(len(statistical_metrics[0])):
        new_row = [
            statistical_metrics[idx_name][i],
            statistical_metrics[idx_name],
            model_name,
            not include_structural,
            dataset_size,
            max_evals,
            statistical_metrics[idx_accuracy_mean][i],
            statistical_metrics[idx_accuracy_ci][i][idx_lb],
            statistical_metrics[idx_accuracy_ci][i][idx_hb],
            statistical_metrics[idx_f1score_mean][i],
            statistical_metrics[idx_f1score_ci][i][idx_lb],
            statistical_metrics[idx_f1score_ci][i][idx_hb],
        ]
        unified_metrics.loc[len(unified_metrics)] = new_row
    return unified_metrics


def str2bool(value: str) -> bool:
    """Function converts a string to boolean in a human expected way.

    Args (str):
      <value> as a string for example 'True' or 'true' or 't'
    Returns (bool):
      bool corresponding to if the <value> was true or false
    """
    return value.lower() in ("yes", "true", "t", "1")
