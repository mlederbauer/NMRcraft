"""Script to show important statist8ics of the dataset, such as ligand classes, ligand class distriutions."""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from nmrcraft.analysis.plotting import style_setup
from nmrcraft.data.dataloader import filename_to_ligands, load_dataset_from_hf


def plot_stacked_bars(
    df, group_col, stack_col, output_file, title, rotation_deg
):
    """
    Generic function to plot stacked bars, with annotations for counts just below the top of each bar.
    """
    _, colors, _ = style_setup()
    plt.figure(figsize=(12, 10))
    categories = df[group_col].unique()
    custom_palette = sns.color_palette(colors)[
        :2
    ]  # Assuming the first two colors are for Mo and W

    # Calculate bottom heights and keep track of top heights for annotation
    bottom_heights = {category: 0 for category in categories}
    top_heights = {category: 0 for category in categories}

    for metal, color in zip(df[stack_col].unique(), custom_palette):
        metal_data = df[df[stack_col] == metal]
        counts = (
            metal_data[group_col]
            .value_counts()
            .reindex(categories, fill_value=0)
        )
        bars = plt.bar(
            categories,
            counts,
            bottom=[bottom_heights[cat] for cat in categories],
            color=color,
            label=metal,
        )

        # Annotate each bar
        for bar, cat in zip(bars, categories):
            height = bar.get_height()
            if height > 0:  # Only annotate if the bar's height is not zero
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    top_heights[cat] + height - 0.05 * height,
                    f"{height}",
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=18,
                    rotation=rotation_deg,
                )
            top_heights[cat] += height

        # Update the bottom heights for the next metal
        for i, cat in enumerate(categories):
            bottom_heights[cat] += counts.iloc[i]

    plt.xticks(rotation=60, ha="right", fontsize=18)
    plt.xlabel(group_col)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(title="Metal")

    plt.savefig(output_file, bbox_inches="tight")
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

    output_path = "plots/dataset_statistics"
    os.makedirs(output_path, exist_ok=True)

    # Modify 'E_ligand' values to group imido-containing ligands
    df["E_ligand_grouped"] = df["E_ligand"].apply(
        lambda x: "Imido Group" if "imido" in x else x
    )

    # Plotting all ligands with imido grouped
    plot_stacked_bars(
        df,
        "E_ligand_grouped",
        "metal",
        os.path.join(output_path, "ligands_distribution_grouped.png"),
        "Distribution of data points per E-ligand, with imido grouped",
        rotation_deg=0,
    )

    # Plotting only imido ligands
    imido_df = df[df["E_ligand"].str.contains("imido")]
    plot_stacked_bars(
        imido_df,
        "E_ligand",
        "metal",
        os.path.join(output_path, "imido_ligands_distribution.png"),
        "Distribution of imido-containing E-ligands",
        rotation_deg=90,
    )


if __name__ == "__main__":
    main()
