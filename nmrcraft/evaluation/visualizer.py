# Import Libraries
import os

import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, model_name: str, data: None, folder_path: str):
        self.model_name = model_name
        self.data = data
        self.folder_path = folder_path

    def plot_ROC(
        self, title="ROC Curves by Dataset Size", filename="ROC_Curves.png"
    ):
        print(self.data.index)
        plt.figure(figsize=(10, 8))
        colors = [
            "blue",
            "green",
            "red",
            "violet",
            "orange",
            "cyan",
        ]  # Colors for different dataset sizes
        labels = [
            f"Dataset Size: {idx}" for idx in self.data.index
        ]  # Labels for legend

        for (index, row), color, label in zip(
            self.data.iterrows(), colors, labels
        ):
            index = index + 1
            plt.plot(
                row["fpr"],
                row["tpr"],
                label=f'{label} (AUC = {row["roc_auc"]:.2f})',
                color=color,
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=2,
            color="gray",
            label="Chance",
            alpha=0.8,
        )
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        file_path = os.path.join(self.folder_path, filename)
        plt.savefig(file_path)
        plt.close()  # Close the plot to free up memory
        return file_path

    def plot_metric(
        self,
        data,
        types,
        title="Title",
        filename="Plot.png",
    ):
        for model in data["model"].unique():
            model_data = data[data["model"] == model]
            plt.plot(
                model_data["dataset_size"],
                model_data[types],
                marker="o",
                label=model,
            )
        plt.legend()
        plt.grid(True)
        file_path = os.path.join(self.folder_path, filename)
        plt.savefig(file_path)
        plt.close()
        return file_path
