"""Script to plot a PCA of the complexes according to their principal components."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nmrcraft.analysis.plotting import style_setup
from nmrcraft.data.dataloader import filename_to_ligands, load_dataset_from_hf


def perform_pca(df, features):
    """Perform PCA on specified features and return principal components."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for plotting
    principal_components = pca.fit_transform(df_scaled)
    principal_df = pd.DataFrame(
        data=principal_components, columns=["PC1", "PC2"]
    )
    return principal_df


def plot_pca(df, pca_df, category, title, filter_condition=None, suffix=""):
    """Generate and save PCA plots colored by categories, with optional filtering."""
    cmap, colors, _ = style_setup()
    fig, ax = plt.subplots()

    if filter_condition is not None:
        filtered_indices = filter_condition(df)
        df_filtered = df[filtered_indices]
        pca_df_filtered = pca_df[filtered_indices]
    else:
        df_filtered = df
        pca_df_filtered = pca_df

    categories = df_filtered[category].unique()
    colors = cmap(np.linspace(0, 1, len(categories)))

    for c, color in zip(categories, colors):
        indices_to_keep = df_filtered[category] == c
        ax.scatter(
            pca_df_filtered.loc[indices_to_keep, "PC1"],
            pca_df_filtered.loc[indices_to_keep, "PC2"],
            s=50,
            label=c,
            color=color,
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_ylim(-5, 10)
    ax.set_xlim(-4, 10)
    ax.set_title(title)
    if category == "E_ligand":
        if suffix == "_without_imido":
            ax.legend(
                title=category,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3,
            )
    elif category == "metal" or category == "geometry":
        ax.legend(
            title=category,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
    plt.savefig(f"plots/pca_{category}{suffix}.png", bbox_inches="tight")
    plt.close()


def main():
    df = filename_to_ligands(load_dataset_from_hf())
    df = df[
        [
            "metal",
            "geometry",
            "E_ligand",
            "M_sigma11_ppm",
            "M_sigma22_ppm",
            "M_sigma33_ppm",
            "M_sigmaiso_ppm",
            "E_sigma11_ppm",
            "E_sigma22_ppm",
            "E_sigma33_ppm",
            "E_sigmaiso_ppm",
        ]
    ]

    features = [
        "M_sigma11_ppm",
        "M_sigma22_ppm",
        "M_sigma33_ppm",
        "M_sigmaiso_ppm",
        "E_sigma11_ppm",
        "E_sigma22_ppm",
        "E_sigma33_ppm",
        "E_sigmaiso_ppm",
    ]

    pca_df = perform_pca(df, features)

    plot_pca(df, pca_df, "metal", "PCA Plot Colored by Metal")
    plot_pca(df, pca_df, "geometry", "PCA Plot Colored by Geometry")

    # Standard plot for E_ligand
    plot_pca(df, pca_df, "E_ligand", "PCA Plot Colored by E-Ligand")

    # Plot without 'imido' entries
    plot_pca(
        df,
        pca_df,
        "E_ligand",
        "PCA Plot Colored by E-Ligand (Without Imido)",
        filter_condition=lambda x: ~x["E_ligand"].str.contains(
            "imido", na=False
        ),
        suffix="_without_imido",
    )

    # Plot only 'imido' entries
    plot_pca(
        df,
        pca_df,
        "E_ligand",
        "PCA Plot Colored by E-Ligand (Imido Only)",
        filter_condition=lambda x: x["E_ligand"].str.contains(
            "imido", na=False
        ),
        suffix="_imido_only",
    )


if __name__ == "__main__":
    main()
