import matplotlib.pyplot as plt
import numpy as np

plt.style.use("./style.mplstyle")
plt.rcParams["text.latex.preamble"] = r"\usepackage{sansmathfonts}"


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
    plt.scatter(y_test, y_pred, color="r", edgecolor="k", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()
