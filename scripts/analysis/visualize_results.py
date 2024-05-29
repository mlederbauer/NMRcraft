import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmrcraft.analysis.plotting import plot_bar, style_setup


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


def plot_exp_1(
    df_base: pd.DataFrame, df_one: pd.DataFrame, metric: str = "accuracy"
):
    """Plot single output models with baselines for accuracy/f1-score as a function of dataset size.

    Args:
        df_base (pd.DataFrame): Baseline data frame.
        df_one (pd.DataFrame): Single output data frame.
        metric (str): The metric to plot ('accuracy' or 'f1').
    """

    # Initialize the plot style and colors
    cmap, colors, all_colors = style_setup()
    del cmap, all_colors
    df_full = pd.concat([df_base, df_one])

    # Get targets
    targets = df_full["target"].unique()

    for target in targets:
        # Restrict the dataframe to the given target
        df = df_full[df_full["target"] == target]

        models = df["model"].unique()
        dataset_fractions = df["dataset_fraction"].unique()

        # Set the appropriate metric columns based on the input argument
        metric_mean = f"{metric}_mean"
        metric_lb = f"{metric}_lb"
        metric_hb = f"{metric}_hb"

        # Sort the dataset fractions to ensure consistent plotting
        dataset_fractions = np.sort(dataset_fractions)

        # Set up the plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Bar width
        total_width = 0.8
        single_width = total_width / len(models)

        for idx, model in enumerate(models):
            means = []
            error_up = []
            error_down = []
            for fraction in dataset_fractions:
                # Filter data for each model and fraction
                subset = df[
                    (df["model"] == model)
                    & (df["dataset_fraction"] == fraction)
                ]
                display_label = f"{model.replace('_', ' ')}"

                means.append(subset[metric_mean].values[0])
                error_down.append(
                    subset[metric_mean].values[0] - subset[metric_lb].values[0]
                )
                error_up.append(
                    subset[metric_hb].values[0] - subset[metric_mean].values[0]
                )

            # Positioning of each group of bars
            positions = (
                np.arange(len(dataset_fractions))
                - (total_width - single_width) / 2
                + idx * single_width
            )

            # Plotting the bars. 'label' argument is modified to use 'display_label'.
            ax.bar(
                positions,
                means,
                color=colors[idx],
                width=single_width,
                label=display_label,  # This line has been changed to use 'display_label'.
                yerr=[error_down, error_up],
                capsize=5,
            )

        # Adding labels and titles
        ax.set_xticks(np.arange(len(dataset_fractions)))
        ax.set_xticklabels(dataset_fractions)
        plt.xticks(fontsize=25)
        ax.set_xlabel("Dataset Size", fontsize=25)
        plt.yticks(fontsize=25)
        if metric == "f1":
            ax.set_ylabel("F1 Score", fontsize=25)
        else:
            ax.set_ylabel("Accuracy", fontsize=25)
        target_clean = target.replace("_", " ")
        ax.set_title(
            f"Model Performance by Dataset Size for {target_clean}",
            fontsize=35,
        )

        # Adding the legend on the right side
        ax.legend(
            title="Model",
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            borderaxespad=0.0,
            fontsize=20,
        )

        # Adjust the plot layout to accommodate the legend
        fig.subplots_adjust(right=0.75)
        plt.tight_layout()

        # Show plot
        plt.savefig(f"plots/results/01_{target}_{metric}.png")


def plot_exp_1_multi(
    df_base: pd.DataFrame, df_one: pd.DataFrame, metric: str = "accuracy"
):
    """Plot single output models with baselines for accuracy/f1-score as a function of dataset size.

    Args:
        df_base (pd.DataFrame): Baseline data frame.
        df_one (pd.DataFrame): Single output data frame.
        metric (str): The metric to plot ('accuracy' or 'f1').
    """
    # Initialize the plot style and colors
    cmap, colors, all_colors = style_setup()
    del cmap, all_colors
    df_full = pd.concat([df_base, df_one])

    # Create a unique identifier for each model based on 'model' and 'model_targets'
    df_full["model_id"] = df_full.apply(
        lambda row: f"{row['model']}_{row['model_targets']}", axis=1
    )

    # Get targets
    targets = df_full["target"].unique()

    for target in targets:
        df = df_full[df_full["target"] == target]

        # Use the new 'model_id' for unique identification
        model_ids = df["model_id"].unique()
        dataset_fractions = df["dataset_fraction"].unique()
        metric_mean = f"{metric}_mean"
        metric_lb = f"{metric}_lb"
        metric_hb = f"{metric}_hb"
        dataset_fractions = np.sort(dataset_fractions)

        fig, ax = plt.subplots(figsize=(15, 8))
        total_width = 0.8
        single_width = total_width / len(model_ids)

        for idx, model_id in enumerate(model_ids):
            means = []
            error_up = []
            error_down = []
            for fraction in dataset_fractions:
                subset = df[
                    (df["model_id"] == model_id)
                    & (df["dataset_fraction"] == fraction)
                ]
                means.append(subset[metric_mean].values[0])
                error_down.append(
                    subset[metric_mean].values[0] - subset[metric_lb].values[0]
                )
                error_up.append(
                    subset[metric_hb].values[0] - subset[metric_mean].values[0]
                )

            positions = (
                np.arange(len(dataset_fractions))
                - (total_width - single_width) / 2
                + idx * single_width
            )
            model_label = (
                model_id.replace("_", " ")
                .replace(",", ", ")
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .capitalize()
                .replace("metal", "\nmetal")
                .replace("metal", "Metal")
                .replace("x3", "X3")
                .replace(" e ", " E ")
            )
            ax.bar(
                positions,
                means,
                color=colors[idx % len(colors)],
                width=single_width,
                label=model_label,
                yerr=[error_down, error_up],
                capsize=5,
            )

        ax.set_xticks(np.arange(len(dataset_fractions)))
        ax.set_xticklabels(dataset_fractions)
        ax.set_xlabel("Dataset Size", fontsize=25)
        plt.xticks(fontsize=25)
        ax.set_ylabel(
            "F1 Score" if metric == "f1" else "Accuracy", fontsize=25
        )
        plt.yticks(fontsize=25)
        target_clean = target.replace("_", " ").capitalize()
        ax.set_title(
            f"Model Performance by Dataset Size for {target_clean}",
            fontsize=35,
        )

        ax.legend(
            title="Model",
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            borderaxespad=0.0,
            fontsize=20,
        )
        fig.subplots_adjust(right=0.75)
        plt.tight_layout()
        plt.savefig(f"plots/results/01_{target}_{metric}_multioutput.png")


def plot_exp_2(df_one, df_multi):
    """Compare the best single-output model to the best multi-output model for each target category."""
    df_combined = pd.concat([df_one, df_multi])
    df_full = df_combined[
        (df_combined["dataset_fraction"] == 1.0)
        & (df_combined["model"] == "random_forest")
        & (df_combined["nmr_only"])
    ]

    target_combinations = [
        "metal",
        "E_ligand",
        "X3_ligand",
        "metal & E_ligand",
        "metal & X3_ligand",
        "X3_ligand & E_ligand",
        "metal & E_ligand & X3_ligand",
    ]
    _, colors, _ = style_setup()

    abbreviated_target_combination_dict = {
        "metal": "Metal",
        "E_ligand": "E",
        "X3_ligand": "X3",
        "metal & E_ligand": "Metal & E",
        "metal & X3_ligand": "Metal & X3",
        "X3_ligand & E_ligand": "X3 & E",
        "metal & E_ligand & X3_ligand": "Metal & X3 & E",
    }

    abbreviated_target_combinations = [
        abbreviated_target_combination_dict[target]
        for target in target_combinations
    ]

    metrics = ["accuracy", "f1"]
    metric_labels = {"accuracy": "Accuracy", "f1": "F1 Score"}

    targets = df_full["target"].unique()

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        total_width = 0.8
        single_width = total_width / len(targets)

        for idx, target in enumerate(targets):
            # target = f"""['{target.replace(" & ", "', '")}']"""
            sub_df = df_full[df_full["target"] == target]
            mean = sub_df[f"{metric}_mean"].values
            error = sub_df[f"{metric}_hb"].values - mean

            positions = (
                np.arange(len(mean))
                - (total_width - single_width) / 2
                + idx * single_width
            )

            ax.bar(
                positions,
                mean,
                single_width,
                label=target,
                color=colors[idx],
                yerr=error,
                capsize=5,
            )

        ax.set_ylabel(metric_labels[metric])
        ax.set_title(
            f"Comparison of Best Models by Target Combination ({metric})"
        )
        ax.set_xticks(range(len(target_combinations)))
        ax.set_xticklabels(abbreviated_target_combinations)
        # ax.legend(title="Model Targets", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        plt.savefig(f"plots/results/02_{metric}_multioutput.png")


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
    "-me",
    type=int,
    default=100,
    help="How many max_evals the analysed data has",
)

parser.add_argument(
    "--results_dir",
    "-rd",
    type=str,
    default="metrics/",
    help="What directory the results are in",
)
# Add arguments
args = parser.parse_args()

if __name__ == "__main__":

    if not os.path.exists("./plots/results/"):
        os.makedirs("./plots/results/")

    df_base, df_one, df_multi = load_results(
        results_dir=args.results_dir,
        baselines_dir="metrics/",
        max_evals=args.max_evals,
    )
    plot_exp_1(df_base, df_one)
    plot_exp_2(df_one, df_multi)
    # plot_exp_3(df_one, df_multi)
