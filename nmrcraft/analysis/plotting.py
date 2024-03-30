import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import gaussian_kde

colors = ["#C28340", "#854F2B", "#61371F", "#8FCA5C", "#70B237", "#477A1E"]
cmap = LinearSegmentedColormap.from_list("custom", colors)

plt.style.use("./style.mplstyle")
plt.rcParams["text.latex.preamble"] = r"\usepackage{sansmathfonts}"
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)


# Use the first color from the custom color cycle
first_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]


def plot_predicted_vs_ground_truth(y_test: np.array, y_pred: np.array, title: str):
    """
    Plots the predicted values against the ground truth values.
    Parameters:
    - y_test (array-like): The ground truth values.
    - y_pred (array-like): The predicted values.
    Returns:
    None
    """

    # Creating the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, color=first_color, edgecolor="k", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()


def plot_predicted_vs_ground_truth_density(y_test: np.array, y_pred: np.array, title: str):
    """
    Plots the predicted values against the ground truth values with a color gradient based on point density.
    Parameters:
    - y_test (array-like): The ground truth values.
    - y_pred (array-like): The predicted values.
    Returns:
    None
    """

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
    plt.scatter(y_test, y_pred, c=scalar_map.to_rgba(kernel), edgecolor="k", alpha=0.9)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()
