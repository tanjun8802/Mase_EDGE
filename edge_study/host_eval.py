"""Host-side Tiny ImageNet sanity check."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets

from edge_study import settings


def tiny_train_root() -> Path:
    for base in (
        settings.PROJECT_ROOT / "mase" / "tiny-imagenet-200",
        settings.PROJECT_ROOT / "tiny-imagenet-200",
    ):
        t = base / "train"
        if t.is_dir():
            return t
    return settings.PROJECT_ROOT / "mase" / "tiny-imagenet-200" / "train"


def edge_host_val_sanity_check(
    model: nn.Module,
    *,
    train_dir: Optional[Path] = None,
    max_batches: int = 10,
    batch_size: int = 64,
) -> Optional[float]:
    """Mean top-1 over a few Tiny ImageNet *train* ImageFolder batches."""
    root = train_dir or tiny_train_root()
    if not root.is_dir():
        return None
    transform = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        datasets.ImageFolder(root=str(root), transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if i >= max_batches:
                break
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            pred = logits.argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    return (correct / total) if total else None
