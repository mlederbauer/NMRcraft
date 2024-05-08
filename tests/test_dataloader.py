import pytest

from nmrcraft.data.dataset import DataLoader


def test_valid_targets():
    """
    This tests checks whether some correctly passed --targets go through as expected.
    """
    feature_columns = [
        "M_sigma11_ppm",
        "M_sigma22_ppm",
        "M_sigma33_ppm",
        "E_sigma11_ppm",
        "E_sigma22_ppm",
        "E_sigma33_ppm",
    ]

    target_columns_set = [
        "metal",
        "metal_X1",
        "metal_X1_X2_X3",
        "metal_X1_X2_X3_X4_L",
        "metal_X1_X2_X3_X4_E",
    ]
    ys = []
    for target_columns in target_columns_set:
        data_loader = DataLoader(
            feature_columns=feature_columns,
            target_columns=target_columns,
            dataset_size=0.01,
        )
        x, x_t, y, y_t, y_cols = data_loader.load_data()
        ys.append(y_t)
        if isinstance(
            y[0], int
        ):  # if the y_t array is 1D, check if the dimensions are the same
            assert True
        elif isinstance(
            y[0], list
        ):  # if the y_t array isn't 1D int array, check if the dimensions are the same on all and if the contents are correct
            assert len(y_cols) == len(y_t[0]) and len(y[0]) == len(y_t[0])
            assert len(x[0]) == len(x_t[0])
            assert isinstance(x[0][0], int) and isinstance(y[0][0], int)
    print(ys)
    # Here we need to assert if the dimension, content etc of the y_targets are correct.


def test_unsupported_targets():  # Check if unsupported targets get recognized
    with pytest.raises(ValueError):
        feature_columns = [
            "M_sigma11_ppm",
            "M_sigma22_ppm",
            "M_sigma33_ppm",
            "E_sigma11_ppm",
            "E_sigma22_ppm",
            "E_sigma33_ppm",
        ]
        data_loader = DataLoader(
            feature_columns=feature_columns,
            target_columns="metal_X1_R-ligand",
            dataset_size=0.01,
        )
        del data_loader


def test_unsupported_target_type():
    with pytest.raises(ValueError):
        feature_columns = [
            "M_sigma11_ppm",
            "M_sigma22_ppm",
            "M_sigma33_ppm",
            "E_sigma11_ppm",
            "E_sigma22_ppm",
            "E_sigma33_ppm",
        ]
        data_loader = DataLoader(
            feature_columns=feature_columns,
            target_columns="metal_X1_X2_X3_L_E",
            dataset_size=0.01,
            target_type="rone-hot-percoding",  # wrong type of target
        )
        a, b, c, d, e = data_loader.load_data()
        del a, b, c, d, e
