import torch.nn as nn


def get_loss_fn():
    return nn.MSELoss()
