import matplotlib.pyplot as plt

plt.style.use("./style.mplstyle")
plt.rcParams["text.latex.preamble"] = r"\usepackage{sansmathfonts}"


def plot_predicted_vs_ground_truth(y_test, y_pred):
    # Creating the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, color="r", edgecolor="k", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Values")
    plt.show()
