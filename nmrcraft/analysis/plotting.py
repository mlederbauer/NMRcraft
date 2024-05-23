import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde


def style_setup():
    """Function to set up matplotlib parameters."""
    colors = ["#C28340", "#854F2B", "#61371F", "#8FCA5C", "#70B237", "#477A1E"]
    cmap = LinearSegmentedColormap.from_list("custom", colors)

    plt.style.use("./style.mplstyle")
    plt.rcParams["text.latex.preamble"] = r"\usepackage{sansmathfonts}"
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)

    all_colors = [
        plt.rcParams["axes.prop_cycle"].by_key()["color"][i]
        for i in range(len(colors))
    ]
    plt.rcParams["text.usetex"] = False

    return cmap, colors, all_colors


def plot_predicted_vs_ground_truth(
    y_test: np.array, y_pred: np.array, title: str
):
    """
    Plots the predicted values against the ground truth values.
    Parameters:
    - y_test (array-like): The ground truth values.
    - y_pred (array-like): The predicted values.
    Returns:
    None
    """
    _, _, colors = style_setup()
    first_color = colors[0]
    # Creating the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, color=first_color, edgecolor="k", alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()


def plot_predicted_vs_ground_truth_density(
    y_test: np.array, y_pred: np.array, title: str
):
    """
    Plots the predicted values against the ground truth values with a color gradient based on point density.
    Parameters:
    - y_test (array-like): The ground truth values.
    - y_pred (array-like): The predicted values.
    Returns:
    None
    """
    cmap, _, _ = style_setup()
    # Calculate the point densities
    values = np.vstack([y_test, y_pred])
    kernel = gaussian_kde(values)(values)

    # Sort points by density (so densest points are plotted last)
    idx = kernel.argsort()
    y_test, y_pred, kernel = y_test[idx], y_pred[idx], kernel[idx]

    # Normalize the densities for color mapping
    norm = Normalize(vmin=kernel.min(), vmax=kernel.max())
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)

    # Creating the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        y_test, y_pred, c=scalar_map.to_rgba(kernel), edgecolor="k", alpha=0.9
    )
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()


def plot_confusion_matrix(
    cm, classes, title, path, full=True, columns_set=False
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
    _, _, _ = style_setup()
    if full:  # Plot one big cm
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(path)
        plt.close()

    elif not full:  # Plot many small cms of each target
        cms = []
        for columns in columns_set:  # Make list of confusion matrices
            cms.append(
                cm[
                    slice(columns[0], columns[-1] + 1),
                    slice(columns[0], columns[-1] + 1),
                ]
            )
        fig, axs = plt.subplots(nrows=len(cms), figsize=(10, 8 * len(cms)))
        for i, sub_cm in enumerate(cms):
            sub_classes = classes[
                slice(columns_set[i][0], columns_set[i][-1] + 1)
            ]
            axs[i].imshow(sub_cm, interpolation="nearest", cmap=plt.cm.Blues)
            axs[i].set_title(f"Confusion Matrix {i+1}")
            tick_marks = np.arange(len(sub_classes))
            axs[i].set_xticks(tick_marks)
            axs[i].set_xticklabels(sub_classes, rotation=45)
            axs[i].set_yticks(tick_marks)
            axs[i].set_yticklabels(sub_classes)
            plt.tight_layout()
        print(cm)
        plt.savefig(path)
        plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, title, path):
    """
    Plots the ROC curve.
    Parameters:
    - fpr (array-like): False positive rate.
    - tpr (array-like): True positive rate.
    - roc_auc (float): Area under the ROC curve.
    - title (str): Title of the plot.
    Returns:
    None
    """
    _, _, _ = style_setup()
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()


def plot_bar(df):
    categories = df["target"].unique()
    _, _, colors = style_setup()
    first_color = colors[0]
    second_color = colors[1]

    # Extract data

    x_pos = np.arange(len(categories))
    bar_width = 0.35

    # Initialize plot
    fig, ax = plt.subplots()

    # Loop through each category and plot bars
    for i, category in enumerate(categories):
        subset = df[df["target"] == category]

        # Means and error bars
        means = subset["accuracy_mean"].values
        errors = [
            subset["accuracy_mean"].values
            - subset["accuracy_lower_bd"].values,
            subset["accuracy_upper_bd"].values
            - subset["accuracy_mean"].values,
        ]

        # Bar locations for the group
        bar_positions = x_pos[i] + np.array([-bar_width / 2, bar_width / 2])

        # Determine bar colors based on 'nmr_tensor_input_only' field
        bar_colors = [
            first_color if x else second_color
            for x in subset["nmr_tensor_input_only"]
        ]

        # Plotting the bars
        ax.bar(
            bar_positions,
            means,
            yerr=np.array(errors),
            color=bar_colors,
            align="center",
            ecolor="black",
            capsize=5,
            width=bar_width,
        )

    # Labeling and aesthetics
    ax.set_ylabel("Accuracy / %")
    ax.set_xlabel("Target(s)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_title("Accuracy Measurements with Error Bars")

    handles = [
        mpatches.Patch(color=first_color, label="With Ligand Info"),
        mpatches.Patch(color=second_color, label="Without Ligand Info"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/exp3_incorporate_ligand_info.png")
