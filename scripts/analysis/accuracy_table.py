"""Script to print the accuracy of the models with and without ligands for each target."""

import pandas as pd


def create_accuracy_table(one_target_path):
    one_target_df = pd.read_csv(one_target_path)
    targets = ["metal", "E_ligand", "X3_ligand"]
    results = []

    for target in targets:
        with_ligands = one_target_df[
            (one_target_df["target"] == target)
            & (one_target_df["model"] == "random_forest")
            & (one_target_df["dataset_fraction"] == 1.0)
            & (one_target_df["nmr_only"] == False)
        ]

        without_ligands = one_target_df[
            (one_target_df["target"] == target)
            & (one_target_df["model"] == "random_forest")
            & (one_target_df["dataset_fraction"] == 1.0)
            & (one_target_df["nmr_only"] == True)
        ]

        if not with_ligands.empty:
            with_ligands_acc = f"{with_ligands['accuracy_mean'].values[0]*100:.1f} ± {((with_ligands['accuracy_hb'].values[0] - with_ligands['accuracy_lb'].values[0])/2)*100:.1f}"
        else:
            with_ligands_acc = "N/A"

        if not without_ligands.empty:
            without_ligands_acc = f"{without_ligands['accuracy_mean'].values[0]*100:.1f} ± {((without_ligands['accuracy_hb'].values[0] - without_ligands['accuracy_lb'].values[0])/2)*100:.1f}"
        else:
            without_ligands_acc = "N/A"

        results.append([target, with_ligands_acc, without_ligands_acc])

    results_df = pd.DataFrame(
        results,
        columns=[
            "Target",
            "With Ligands: Accuracy / %",
            "Without Ligands: Accuracy / %",
        ],
    )
    print(results_df)
    return results_df


if __name__ == "__main__":
    create_accuracy_table("metrics/results_one_target.csv")
