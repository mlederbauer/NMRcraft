import pandas as pd

from nmrcraft.analysis.plotting import plot_bar

csv_path = "data/path_to_results.csv"
df = pd.read_csv(csv_path)

# TODO: make more input agnostic

plot_bar(df)
