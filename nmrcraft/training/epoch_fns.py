import torch


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        loss_fn (torch.nn.Module): The loss function to use for training.
        device (torch.device): The device to use for training.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch).squeeze(-1)
        loss = loss_fn(out, batch.y.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)


def val_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Validate the model for one epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        loss_fn (torch.nn.Module): The loss function to use for validation.
        device (torch.device): The device to use for validation.

    Returns:
        float: The average loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch.to(device)
            out = model(batch).squeeze(-1)
            loss = loss_fn(out, batch.y.to(device))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(val_loader.dataset)


def inference_epoch(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Perform inference on the model for one epoch.

    Args:
        model (torch.nn.Module): The model to perform inference on.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        device (torch.device): The device to use for inference.

    Returns:
        torch.Tensor: The concatenated predictions.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch.to(device)
            out = model(batch).squeeze(-1)
            predictions.append(out.cpu())
    return torch.cat(predictions)
