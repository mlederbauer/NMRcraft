import os

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

    # Use the first color from the custom color cycle
    first_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    plt.rcParams["text.usetex"] = False

    return cmap, colors, first_color


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
    _, _, first_color = style_setup()
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
    # _, _, _ = style_setup()
    for target in y_labels:
        file_path = os.path.join(
            folder_path,
            f"ConfusionMatrix_{model_name}_{dataset_size}_{target}.png",
        )
        cm = cm_list[target]
        classes = y_labels[target]
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
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
