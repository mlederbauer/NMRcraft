# Import Libraries
import os

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap


class Visualizer:
    def __init__(
        self,
        model_name: str,
        cm: None,
        rates=None,
        metrics=None,
        folder_path: str = "plots/",
        classes=None,
        dataset_size=None,
    ):
        self.model_name = model_name
        self.cm = cm
        self.rates = (rates,)
        self.metrics = metrics
        self.folder_path = folder_path
        self.classes = classes
        self.dataset_size = dataset_size
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def style_setup():
        """Function to set up matplotlib parameters."""
        colors = [
            "#C28340",
            "#854F2B",
            "#61371F",
            "#8FCA5C",
            "#70B237",
            "#477A1E",
        ]
        cmap = LinearSegmentedColormap.from_list("custom", colors)

        plt.style.use("./style.mplstyle")
        plt.rcParams["text.latex.preamble"] = r"\usepackage{sansmathfonts}"
        plt.rcParams["axes.prop_cycle"] = cycler(color=colors)

        # Use the first color from the custom color cycle
        first_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        plt.rcParams["text.usetex"] = False

        return cmap, colors, first_color

    def plot_confusion_matrix(self, full=True, columns_set=False):
        """
        Plots the confusion matrix.
        Parameters:
        - classes (list): List of classes for the axis labels.
        - title (str): Title of the plot.
        - full (bool): If true plots one big, else many smaller.
        - columns_set (list of lists): contains all relevant indices.
        Returns:
        None
        """

        def normalize_row_0_1(row):
            return (row - np.min(row)) / (np.max(row) - np.min(row))

        file_path = os.path.join(
            self.folder_path,
            f"ConfusionMatrix_{self.model_name}_{self.dataset_size}.png",
        )
        # _, _, _ = self.style_setup()
        if full:  # Plot one big cm
            plt.figure(figsize=(10, 8))
            plt.imshow(
                self.cm.apply(normalize_row_0_1, axis=1),
                interpolation="nearest",
                cmap=plt.cm.Blues,
            )
            plt.title("The Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(self.classes))
            plt.xticks(tick_marks, self.classes, rotation=45)
            plt.yticks(tick_marks, self.classes)
            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.savefig(file_path)
            plt.close()

        elif not full:  # Plot many small cms of each target
            cms = []
            for columns in columns_set:  # Make list of confusion matrices
                cms.append(
                    self.cm[
                        slice(columns[0], columns[-1] + 1),
                        slice(columns[0], columns[-1] + 1),
                    ]
                )
            fig, axs = plt.subplots(nrows=len(cms), figsize=(10, 8 * len(cms)))
            for i, sub_cm in enumerate(cms):
                sub_classes = self.classes[
                    slice(columns_set[i][0], columns_set[i][-1] + 1)
                ]
                axs[i].imshow(
                    sub_cm, interpolation="nearest", cmap=plt.cm.Blues
                )
                axs[i].set_title(f"Confusion Matrix {i+1}")
                tick_marks = np.arange(len(sub_classes))
                axs[i].set_xticks(tick_marks)
                axs[i].set_xticklabels(sub_classes, rotation=45)
                axs[i].set_yticks(tick_marks)
                axs[i].set_yticklabels(sub_classes)
                plt.tight_layout()
            # plt.savefig(path)
            plt.close()
            return file_path

    def plot_metric(
        self,
        data,
        metric,
        title="Title",
        filename="Plot.png",
    ):
        """
        Generates a plot for a specified metric against dataset size for different models.

        The graph includes error bars representing the standard deviation of the metric.

        Args:
            data (pd.DataFrame): DataFrame with columns 'model', 'dataset_size', metric, and its standard deviation.
            metric (str): Name of the metric to be plotted (e.g., 'accuracy', 'f1_score').
            title (str, optional): Plot title. Defaults to "Title".
            filename (str, optional): Filename for saving the plot. Defaults to "Plot.png".

        Returns:
            str: Path where the plot is saved.
        """

        for model in data["model"].unique():
            model_data = data[data["model"] == model]
            std_name = metric + "_std"
            plt.errorbar(
                model_data["dataset_size"],
                model_data[metric],
                yerr=model_data[std_name],
                fmt="o",
                label=model,
            )
        plt.legend()
        plt.grid(True)
        plt.xlabel("Dataset Size")
        plt.ylabel(metric)
        file_path = os.path.join(self.folder_path, filename)
        plt.savefig(file_path)
        plt.close()
        return file_path
