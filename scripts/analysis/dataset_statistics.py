"""Script to show important statist8ics of the dataset, such as ligand classes, ligand class distriutions."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nmrcraft.analysis.plotting import style_setup
from nmrcraft.data.dataset import filename_to_ligands, load_dataset_from_hf


def plot_all_ligands_with_imido_grouped(
    df: pd.DataFrame, output_file: str
) -> None:
    """
    Plot the distribution of data points for all E-ligands, grouping all imido-containing ligands,
    and color the bars based on metal type.
    """
    _, colors, _ = style_setup()
    plt.figure(figsize=(12, 10))
    custom_palette = sns.color_palette(colors)[
        :2
    ]  # Assuming the first two colors are for Mo and W
    df["E_ligand_grouped"] = df["E_ligand"].apply(
        lambda x: "Imido Group" if "imido" in x else x
    )

    ax = sns.countplot(
        x="E_ligand_grouped", hue="metal", data=df, palette=custom_palette
    )
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("E-Ligand")
    plt.ylabel("Count")
    plt.title(
        "Distribution of data points per E-ligand, with imido grouped",
        fontsize=14,
    )
    plt.legend(title="Metal", loc="upper right")
    plt.yticks()
    plt.ylim(0, df["E_ligand_grouped"].value_counts().max() + 800)

    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(
            f"{count}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=12,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
            rotation=60,
        )

    plt.savefig(
        os.path.join(output_file, "ligands_distribution_grouped.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_imido_ligands_only(df: pd.DataFrame, output_file: str) -> None:
    """
    Plot the distribution of data points only for imido-containing E-ligands,
    and color the bars based on metal type.
    """
    _, colors, _ = style_setup()
    plt.figure(figsize=(12, 10))
    df = df[df["E_ligand"].str.contains("imido")]
    custom_palette = sns.color_palette(colors)[
        :2
    ]  # Same color mapping assumption for Mo and W

    ax = sns.countplot(
        x="E_ligand", hue="metal", data=df, palette=custom_palette
    )
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Imido E-Ligand")
    plt.ylabel("Count")
    plt.title("Distribution of imido-containing E-ligands", fontsize=14)
    plt.legend(title="Metal", loc="upper right")
    plt.yticks()
    plt.ylim(0, df["E_ligand"].value_counts().max() + 800)

    for p in ax.patches:
        count = int(p.get_height())
        ax.annotate(
            f"{count}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=12,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
            rotation=60,
        )

    plt.savefig(
        os.path.join(output_file, "imido_ligands_distribution.png"),
        bbox_inches="tight",
    )
    plt.close()


def main():
    # Load data
    df = filename_to_ligands(load_dataset_from_hf())
    df = df[
        [
            "metal",
            "geometry",
            "E_ligand",
            "X1_ligand",
            "X2_ligand",
            "X3_ligand",
            "X4_ligand",
            "L_ligand",
        ]
    ]

    output_path = "plots"
    os.makedirs(output_path, exist_ok=True)

    # Plotting
    plot_all_ligands_with_imido_grouped(df, output_path)
    plot_imido_ligands_only(df, output_path)


if __name__ == "__main__":
    main()
