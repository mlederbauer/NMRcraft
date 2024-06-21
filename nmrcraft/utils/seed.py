"""Function to set a random seed."""

import random

import numpy as np

# import torch


def set_seed(seed: int = 42) -> None:
    """Set the seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed}.")
