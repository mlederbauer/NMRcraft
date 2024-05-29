import argparse
import ast
import os

import pandas as pd

from nmrcraft.analysis.plotting import plot_bar, plot_metric


def load_results(results_dir: str, baselines_dir: str, max_evals: int):
    import_filename_base = os.path.join(baselines_dir, "results_baselines.csv")
    import_filename_one = os.path.join(results_dir, "results_one_target.csv")
    import_filename_multi = os.path.join(
        results_dir, "results_multi_target.csv"
    )

    df_base = pd.read_csv(import_filename_base)
    df_one = pd.read_csv(import_filename_one)
    df_one = df_one[df_one["max_evals"] == max_evals]
    df_multi = pd.read_csv(import_filename_multi)
    df_multi = df_multi[df_multi["max_evals"] == max_evals]
    return df_base, df_one, df_multi


def plot_exp_1(df_base, df_one):
    """Compare single output models with baselines for accuracy/f1-score as a function of dataset size."""
    # x axis = datase size
    # y axis = accuracy or f1 score
    # legend on the bottom of the plot (below the plot itself)
    # color according to the used model
    # use bar plots for one dataset_size and put them next to each other

    df_combined = pd.concat([df_base, df_one])

    targets = df_combined["target"].unique()
    for target in targets:
        sub_df = df_combined[df_combined["target"] == target]
        print(sub_df)
        plot_metric(
            sub_df,
            title=f"Accuracy of {target} Prediction",
            filename=f"plots/01_accuracy_{target}.png",
            metric="accuracy",
            iterative_column="model",
            xdata="dataset_fraction",
        )
        plot_metric(
            sub_df,
            title=f"F1-Score of {target} Prediction",
            filename=f"plots/01_f1-score_{target}.png",
            metric="f1",
            iterative_column="model",
            xdata="dataset_fraction",
        )
    return


def plot_exp_2(df_one, df_multi):
    """Compare a single-output model to its multi-output model counterpart. For the targets
    x axis = metal, metal E, metal X3, E X3, metal E X3
    y axus = accuracy or f1 score
    use the best one-target/multi-target model for the target to plot
    and take dataset_size = 1.0
    use bar plot
    """
    df_combined = pd.concat([df_one, df_multi])
    full_df = df_combined[df_combined["dataset_fraction"] == 1]

    true_df = full_df[full_df["nmr_only"]]
    models = true_df["model"].unique()
    for model in models:
        sub_df = true_df[true_df["model"] == model]
        print(sub_df)
        plot_bar(
            sub_df,
            title=f"Accuracy for {model} Predictions",
            filename=f"plots/02_accuracy_{model}.png",
            metric="accuracy",
            iterative_column="target",
            xdata="xlabel",
        )
        plot_bar(
            sub_df,
            title=f"F1-Score for {model} Predictions",
            filename=f"plots/02_f1-score_{model}.png",
            metric="f1",
            iterative_column="target",
            xdata="xlabel",
        )

    full_df["model_targets"] = full_df["model_targets"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Add 'lig' to model_targets if nmr_only is True
    full_df["model_targets"] = full_df.apply(
        lambda row: (
            row["model_targets"] + ["lig"]
            if not row["nmr_only"]
            else row["model_targets"]
        ),
        axis=1,
    )
    return


def plot_exp_3(df_one, df_multi):
    """Compare whether nmr-only is set to true or false
    plot for X3 (best one target model) the bar plot with/without ligands
    plot for metal & E & X3 (best multi target model) the bar plot with/withut ligands
    legens also below the plot itself
    """
    df_combined = pd.concat([df_one, df_multi])
    full_df = df_combined[df_combined["dataset_fraction"] == 1]

    models = full_df["model"].unique()
    for model in models:
        sub_df = full_df[full_df["model"] == model]
        print(sub_df)
        plot_bar(
            sub_df,
            title=f"Accuracy for {model} Predictions",
            filename=f"plots/03_accuracy_{model}.png",
            metric="accuracy",
            iterative_column="target",
            xdata="xlabel",
        )
        plot_bar(
            sub_df,
            title=f"F1-Score for {model} Predictions",
            filename=f"plots/03_f1-score_{model}.png",
            metric="f1",
            iterative_column="target",
            xdata="xlabel",
        )
    return


# Setup parser
parser = argparse.ArgumentParser(
    description="Train a model with MLflow tracking."
)

parser.add_argument(
    "--max_evals",
    type=int,
    default=100,
    help="How many max_evals the analysed data has",
)
# Add arguments
args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists("./plots/results/"):
        os.makedirs("./plots/results/")

    df_base, df_one, df_multi = load_results(
        results_dir="metrics/20eval/",
        baselines_dir="metrics/",
        max_evals=args.max_evals,
    )

    plot_exp_1(df_base, df_one)
    plot_exp_2(df_one, df_multi)
    plot_exp_3(df_one, df_multi)
