from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SequenceTrainingResult:
    model: nn.Module
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]


def set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def build_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensor_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def train_sequence_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> SequenceTrainingResult:
    set_torch_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_loader = build_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    validation_loader = build_loader(
        X_validation,
        y_validation,
        batch_size=batch_size,
        shuffle=False,
    )

    history: list[dict[str, float]] = []
    best_epoch = 1
    best_validation_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for features, targets in validation_loader:
                features = features.to(device)
                targets = targets.to(device)
                loss = loss_fn(model(features), targets)
                validation_losses.append(float(loss.item()))

        epoch_train_loss = float(np.mean(train_losses))
        epoch_validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": epoch_train_loss,
                "validation_loss": epoch_validation_loss,
            }
        )

        if epoch_validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = epoch_validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return SequenceTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        history=history,
    )


def fit_sequence_fixed_epochs(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> nn.Module:
    set_torch_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = build_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()

    return model


def predict_sequence_model(
    model: nn.Module,
    X_values: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    loader = build_loader(
        X_values,
        np.zeros((len(X_values), 1), dtype=np.float32),
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            predictions.append(model(features).cpu().numpy())

    return np.concatenate(predictions, axis=0)
