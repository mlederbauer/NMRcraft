from nmrcraft.data.dataset import DataLoader

feature_columns = [
    "M_sigma11_ppm",
    "M_sigma22_ppm",
    "M_sigma33_ppm",
    "E_sigma11_ppm",
    "E_sigma22_ppm",
    "E_sigma33_ppm",
]

data = DataLoader(
    feature_columns=feature_columns,
    target_columns="metal_X1_X2_X3_L_E",
    dataset_size=0.00021,
    target_type="categorical",
)

x, x_t, y, y_t = data.load_data()
