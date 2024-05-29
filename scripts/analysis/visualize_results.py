import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nmrcraft.analysis.plotting import plot_bar, plot_metric


def load_results(results_dir: str, baselines_dir: str):
    import_filename_base = os.path.join(baselines_dir, "results_baselines.csv")
    import_filename_one = os.path.join(results_dir, "results_one_target.csv")
    import_filename_multi = os.path.join(
        results_dir, "results_multi_target.csv"
    )

    df_base = pd.read_csv(import_filename_base)
    df_one = pd.read_csv(import_filename_one)
    df_multi = pd.read_csv(import_filename_multi)

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
    """Compare the best single-output model to the best multi-output model for each target category.
    Separate plots for accuracy and F1 score, including upper and lower bounds.
    """
    # Combine and filter the dataframes
    df_combined = pd.concat([df_one, df_multi])
    df_full = df_combined[df_combined["dataset_fraction"] == 1.0]

    # Define the target combinations to plot
    target_combinations = [
        "metal",
        "E_ligand",
        "X3_ligand",
        "metal & E_ligand",
        "metal & X3_ligand",
        "E_ligand & X3_ligand",
        "metal & E_ligand & X3_ligand",
    ]

    # Define colors for clarity
    colors = {"metal": "navy", "E_ligand": "darkorange", "X3_ligand": "green"}

    # Prepare to plot for both Accuracy and F1 score
    metrics = ["accuracy", "f1"]
    metric_labels = {"accuracy": "Accuracy", "f1": "F1 Score"}

    # Iterate over each metric to create separate plots
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))
        bar_width = 0.05  # Adjusted for multiple bars

        # Process each target combination
        for idx, target in enumerate(target_combinations):
            sub_df = df_full[
                df_full["model_targets"].apply(
                    lambda x: set(eval(x))  # noqa: S307 PGH001
                    == set(target.split(" & "))  # noqa: B023
                )
            ]

            if not sub_df.empty:
                # Determine sub-positions for bars within the same main position
                targets_to_plot = target.split(" & ")
                sub_positions = np.arange(len(targets_to_plot)) * bar_width + (
                    idx * 2 * bar_width
                )

                for sub_idx, target_to_plot in enumerate(targets_to_plot):
                    best_model = sub_df[
                        sub_df["target"] == target_to_plot
                    ].nlargest(1, f"{metric}_mean")

                    if not best_model.empty:
                        mean = best_model[f"{metric}_mean"].values[0]
                        lower = mean - best_model[f"{metric}_lb"].values[0]
                        upper = best_model[f"{metric}_hb"].values[0] - mean

                        # Plot bars with error bars
                        ax.bar(
                            sub_positions[sub_idx],
                            mean,
                            bar_width,
                            label=f"{target_to_plot} ({metric})"
                            if idx == 0 and sub_idx == 0
                            else "",
                            color=colors.get(target_to_plot, "grey"),
                            yerr=[[lower], [upper]],
                            capsize=5,
                        )

        # Add labels and legend
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(
            f"Comparison of Best Models by Target Combination ({metric_labels[metric]})"
        )
        ax.set_xticks(
            np.arange(len(target_combinations)) * 2 * bar_width + bar_width
        )
        ax.set_xticklabels(target_combinations)
        ax.legend()

        # Show the plot
        plt.tight_layout()
        plt.savefig(f"{metric}_comparison_plot.png")
        plt.savefig("fooo.png")
        pass


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


if __name__ == "__main__":

    if not os.path.exists("./plots/results/"):
        os.makedirs("./plots/results/")

    df_base, df_one, df_multi = load_results(
        results_dir="metrics/20eval/", baselines_dir="metrics/"
    )
    plot_exp_1(df_base, df_one)
    plot_exp_2(df_one, df_multi)
    plot_exp_3(df_one, df_multi)
