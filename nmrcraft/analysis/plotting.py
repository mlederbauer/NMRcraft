"""Functions to plot."""

import ast
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap


def style_setup() -> Tuple[LinearSegmentedColormap, List[str], List[str]]:
    """Function to set up matplotlib parameters."""
    colors = [
        "#C28340",
        "#854F2B",
        "#61371F",
        "#8FCA5C",
        "#70B237",
        "#477A1E",
        "#3B661A",
    ]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    plt.style.use("./style.mplstyle")
    plt.rcParams["text.latex.preamble"] = r"\usepackage{sansmathfonts}"
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    plt.rcParams["font.size"] = 20  # Set default font size
    plt.rcParams["axes.titlesize"] = 20  # Title font size
    plt.rcParams["axes.labelsize"] = 20  # X and Y label font size
    plt.rcParams["xtick.labelsize"] = 14  # X tick label font size
    plt.rcParams["ytick.labelsize"] = 14  # Y tick label font size
    plt.rcParams["legend.fontsize"] = 12  # Legend font size

    all_colors = [
        plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        for i in range(len(colors))
    ]
    plt.rcParams["text.usetex"] = False

    return cmap, colors, all_colors


def plot_confusion_matrix(
    cm_list, y_labels, model_name, dataset_size, folder_path: str = "plots/"
):
    """
    Plots the confusion matrix.
    Parameters:
    - cm (array-like): Confusion matrix data.
    - classes (list): List of classes for the axis labels.
    - title (str): Title of the plot.
    - full (bool): If true plots one big, else many smaller.
    - columns_set (list of lists): contains all relevant indices.
    Returns:
    None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    cmap, _, _ = style_setup()
    for target in y_labels:
        file_path = os.path.join(
            folder_path,
            f"ConfusionMatrix_{model_name}_{dataset_size}_{target}.png",
        )
        cm = cm_list[target]
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        classes = y_labels[target]
        plt.figure(figsize=(10, 8))
        plt.imshow(
            cm_normalized, interpolation="nearest", cmap=cmap, vmin=0, vmax=1
        )
        plt.title(f"{target} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(file_path)
        plt.close()


def plot_metric(
    data,
    title="Accuracy",
    filename="plots/accuracy.png",
    metric="accuracy",
    iterative_column="model",
    xdata="dataset_fraction",
    legend=True,
):
    _, colors, _ = style_setup()
    if iterative_column == "target":

        def convert_to_labels(target_list):
            label_dict = {"metal": "M", "E_ligand": "E", "X3_ligand": "X3"}
            return ", ".join([label_dict[i] for i in target_list])

        # Convert string representations of lists to actual lists
        data["model_targets"] = data["model_targets"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        data["xlabel"] = data["model_targets"].apply(convert_to_labels)
        print(data)

    for i, iterator in enumerate(data[iterative_column].unique()):
        model_data = data[data[iterative_column] == iterator]
        errors = [
            model_data[metric + "_mean"].values
            - model_data[metric + "_lb"].values,
            model_data[metric + "_hb"].values
            - model_data[metric + "_mean"].values,
        ]
        plt.errorbar(
            model_data[xdata],
            model_data[metric + "_mean"],
            yerr=errors,
            fmt="o-",
            label=iterator,
            color=colors[i],
            capsize=5,
        )
    plt.legend()
    plt.title(title)
    plt.grid(True)
    if iterative_column == "model":
        plt.xlim(0, 1)
        plt.ylim(0, 1.2)
    plt.xlabel("Dataset Size")
    plt.ylabel(metric)
    plt.savefig(filename)
    plt.close()


def plot_bar(
    data,
    title="Accuracy",
    filename="plots/accuracy.png",
    metric="accuracy",
    iterative_column="model",
    xdata="dataset_fraction",
):
    _, colors, _ = style_setup()

    def convert_to_labels(target_list):
        label_dict = {
            "metal": "Metal",
            "E_ligand": "E",
            "X3_ligand": "X3",
            "lig": "\n  (ligands input)",
        }
        return " & ".join([label_dict[i] for i in target_list])

    # Convert string representations of lists to actual lists
    data["model_targets"] = data["model_targets"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    data["xlabel"] = data["model_targets"].apply(convert_to_labels)

    # Aggregate the data to handle duplicates
    aggregated_data = (
        data.groupby(["xlabel", "target"])
        .agg({metric + "_mean": "mean"})
        .reset_index()
    )
    aggregated_data_lb = (
        data.groupby(["xlabel", "target"])
        .agg({metric + "_lb": "mean"})
        .reset_index()
    )
    aggregated_data_hb = (
        data.groupby(["xlabel", "target"])
        .agg({metric + "_hb": "mean"})
        .reset_index()
    )

    # Pivot the aggregated data
    new_df = aggregated_data.pivot(
        index="xlabel", columns="target", values=metric + "_mean"
    ).loc[
        [
            "Metal",
            "E",
            "X3",
            "Metal & E",
            "Metal & X3",
            "X3 & E",
            "Metal & E & X3",
        ]
    ]
    print(new_df)
    new_lb = aggregated_data_lb.pivot(
        index="xlabel", columns="target", values=metric + "_lb"
    ).loc[
        [
            "Metal",
            "E",
            "X3",
            "Metal & E",
            "Metal & X3",
            "X3 & E",
            "Metal & E & X3",
        ]
    ]
    new_hb = aggregated_data_hb.pivot(
        index="xlabel", columns="target", values=metric + "_hb"
    ).loc[
        [
            "Metal",
            "E",
            "X3",
            "Metal & E",
            "Metal & X3",
            "X3 & E",
            "Metal & E & X3",
        ]
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.25  # width of the bar
    x = np.arange(len(new_df.index))

    # Plotting each column (target) as a separate group
    for i, column in enumerate(["metal", "E_ligand", "X3_ligand"]):
        ax.bar(
            x + i * width,
            new_df[column],
            width,
            color=colors[i * 2],
            label=column,
            yerr=[
                new_df[column] - new_lb[column],
                new_hb[column] - new_df[column],
            ],
            capsize=5,
        )

    ax.set_ylabel(
        "Accuracy" if metric == "accuracy" else "F1 Score", fontsize=30
    )
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x + width * (len(new_df.columns) - 1) / 2)
    ax.set_xticklabels(new_df.index, rotation=45, fontsize=30)
    ax.set_title(title, fontsize=35, pad=20)
    plt.grid(False)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
