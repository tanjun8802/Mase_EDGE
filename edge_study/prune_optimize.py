"""Torch pruning + edge_optimise_model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn
import torch.nn.utils.prune as prune

from edge_study import model_loaders


def apply_torch_l1_prune(model: nn.Module, amount: float) -> None:
    if amount <= 0:
        return
    to_prune = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in to_prune:
        prune.l1_unstructured(m, name="weight", amount=amount)
    for m in to_prune:
        prune.remove(m, "weight")


def edge_optimise_model(
    config: Dict[str, Any],
    enable_qat: bool = False,
    model: Optional[nn.Module] = None,
) -> nn.Module:
    del enable_qat
    if model is None:
        model = model_loaders.load_model_and_train()
    model = model.cpu()
    apply_torch_l1_prune(model, float(config.get("prune_ratio", 0.0)))
    model.eval()
    return model
