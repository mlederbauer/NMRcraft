import ast
import os

import pandas as pd

from nmrcraft.analysis.plotting import plot_bar, plot_metric

import_filename_base = "metrics/results_baselines.csv"
import_filename_one = "metrics/results_one_target.csv"
import_filename_multi = "metrics/results_multi_target.csv"


df_base = pd.read_csv(import_filename_base)
df_one = pd.read_csv(import_filename_one)
df_multi = pd.read_csv(import_filename_multi)

df = pd.concat([df_base, df_one])

if not os.path.exists("./plots"):
    os.makedirs("./plots")

targets = df["target"].unique()
for target in targets:
    sub_df = df[df["target"] == target]
    print(sub_df)
    plot_metric(
        sub_df,
        title=f"Accuracy of {target} Prediction",
        filename=f"plots/01_accuracy_{target}.png",
        metric="accuracy",
        iterative_column="model",
        xdata="dataset_fraction",
    )
    plot_metric(
        sub_df,
        title=f"F1-Score of {target} Prediction",
        filename=f"plots/01_f1-score_{target}.png",
        metric="f1",
        iterative_column="model",
        xdata="dataset_fraction",
    )


df = pd.concat([df_one, df_multi])
full_df = df[df["dataset_fraction"] == 1]

true_df = full_df[full_df["nmr_only"]]
models = true_df["model"].unique()
for model in models:
    sub_df = true_df[true_df["model"] == model]
    print(sub_df)
    plot_bar(
        sub_df,
        title=f"Accuracy for {model} Predictions",
        filename=f"plots/02_accuracy_{model}.png",
        metric="accuracy",
        iterative_column="target",
        xdata="xlabel",
    )
    plot_bar(
        sub_df,
        title=f"F1-Score for {model} Predictions",
        filename=f"plots/02_f1-score_{model}.png",
        metric="f1",
        iterative_column="target",
        xdata="xlabel",
    )


full_df["model_targets"] = full_df["model_targets"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Add 'lig' to model_targets if nmr_only is True
full_df["model_targets"] = full_df.apply(
    lambda row: row["model_targets"] + ["lig"]
    if not row["nmr_only"]
    else row["model_targets"],
    axis=1,
)

print(full_df)


models = full_df["model"].unique()
for model in models:
    sub_df = full_df[full_df["model"] == model]
    print(sub_df)
    plot_bar(
        sub_df,
        title=f"Accuracy for {model} Predictions",
        filename=f"plots/03_accuracy_{model}.png",
        metric="accuracy",
        iterative_column="target",
        xdata="xlabel",
    )
    plot_bar(
        sub_df,
        title=f"F1-Score for {model} Predictions",
        filename=f"plots/03_f1-score_{model}.png",
        metric="f1",
        iterative_column="target",
        xdata="xlabel",
    )
