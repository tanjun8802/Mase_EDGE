"""Optuna search space for EDGE study."""

from __future__ import annotations

from typing import Any, Dict

from edge_study.quant_xnnpack import (
    QUANT_LAYER_NAMES,
    XNNPACK_LAYER_QUANT_CHOICES,
    optuna_param_name_for_module,
)


def edge_optuna_config(trial) -> Dict[str, Any]:
    """Per-layer XNNPACK-valid INT8 (or none) + L1 pruning amount."""
    prune_ratio = trial.suggest_float("prune_ratio", 0.0, 0.7)
    layer_quant: Dict[str, str] = {}
    for mod_name in QUANT_LAYER_NAMES:
        pname = optuna_param_name_for_module(mod_name)
        layer_quant[mod_name] = trial.suggest_categorical(
            pname, list(XNNPACK_LAYER_QUANT_CHOICES)
        )

    return {
        "prune_ratio": prune_ratio,
        "backend": "xnnpack",
        "use_mixed_delegation": False,
        "delegation_plan": None,
        "layer_quant": layer_quant,
    }
