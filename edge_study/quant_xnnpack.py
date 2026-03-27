"""XNNPACK PT2E per-layer scheme helpers."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch.nn as nn
import torchvision.models as models
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
)

__all__ = [
    "LAYER_QUANT_SCHEME_EXPORT_MAP",
    "QUANT_LAYER_NAMES",
    "XNNPACK_LAYER_QUANT_CHOICES",
    "list_xnnpack_quantizable_module_names",
    "optuna_param_name_for_module",
    "resolve_layer_quant_scheme_for_export",
    "scheme_to_xnnpack_config",
]


def list_xnnpack_quantizable_module_names(model: nn.Module) -> List[str]:
    """Module FQNs for XNNPACK-supported weight quant (conv / linear in nn.Module tree)."""
    return [
        n
        for n, m in model.named_modules()
        if n and isinstance(m, (nn.Conv2d, nn.Linear))
    ]


_tm = models.mobilenet_v3_large(weights=None)
QUANT_LAYER_NAMES: List[str] = list_xnnpack_quantizable_module_names(_tm)
del _tm


def optuna_param_name_for_module(module_fqn: str) -> str:
    return "q_" + re.sub(r"[^0-9a-zA-Z]+", "_", module_fqn).strip("_")


XNNPACK_LAYER_QUANT_CHOICES: Tuple[str, ...] = (
    "none",
    "int8_pc_static",
    "int8_pt_static",
)


def scheme_to_xnnpack_config(scheme: str):
    if scheme == "none":
        raise ValueError("scheme 'none' must skip set_module_name")
    if scheme == "int8_pc_static":
        return get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
    if scheme == "int8_pt_static":
        return get_symmetric_quantization_config(is_per_channel=False, is_dynamic=False)
    if scheme == "int8_pc_dynamic":
        return get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    if scheme == "int8_pt_dynamic":
        return get_symmetric_quantization_config(is_per_channel=False, is_dynamic=True)
    raise ValueError(f"Unknown scheme {scheme!r}")


LAYER_QUANT_SCHEME_EXPORT_MAP: Dict[str, str] = {
    "int8_pc_dynamic": "int8_pc_static",
    "int8_pt_dynamic": "int8_pt_static",
}


def resolve_layer_quant_scheme_for_export(scheme: str) -> str:
    return LAYER_QUANT_SCHEME_EXPORT_MAP.get(scheme, scheme)
