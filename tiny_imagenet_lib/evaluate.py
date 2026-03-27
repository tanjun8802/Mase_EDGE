"""Validation accuracy (percentage) — same logic as the training notebooks."""

from __future__ import annotations

from typing import Any, Iterable, Tuple, Union

import torch
import torch.nn as nn

TensorDevice = Union[str, torch.device]


def classification_accuracy_percent(correct: int, total: int) -> float:
    """Percentage matching ``return 100 * correct / total`` in the notebooks."""
    return 100.0 * float(correct) / float(total)


def evaluate(
    model: nn.Module,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    device: TensorDevice | None = None,
) -> float:
    """
    Top-1 accuracy in [0, 100].

    Notebooks used a global ``device``; pass ``device=`` explicitly (e.g. ``\"cpu\"``)
    or leave ``None`` to use the model's parameter device.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            _, preds = model(imgs).max(1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return classification_accuracy_percent(correct, total)
