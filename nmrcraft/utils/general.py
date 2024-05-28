import pandas as pd


def add_rows_metrics(
    unified_metrics: pd.DataFrame,
    statistical_metrics: list,
    dataset_size,
    include_structural: bool,
    model_name: str,
):
    # Add all the newly generated metrics to the unified dataframe targetwise
    for i in range(len(statistical_metrics[0])):
        new_row = [
            statistical_metrics[0][i],
            model_name,
            not include_structural,
            dataset_size,
            statistical_metrics[1][i],
            statistical_metrics[2][i][0],
            statistical_metrics[2][i][1],
            statistical_metrics[3][i],
            statistical_metrics[4][i][0],
            statistical_metrics[4][i][1],
        ]
        unified_metrics.loc[len(unified_metrics)] = new_row
        return unified_metrics
