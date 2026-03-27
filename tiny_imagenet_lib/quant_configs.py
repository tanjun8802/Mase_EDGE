"""CHOP / Mase-style quantization dicts (ResNet PTQ + MobileNet mixed-precision builders)."""

from __future__ import annotations

from typing import Any, Dict

# ResNet18.ipynb — quantize_transform_pass(pass_args=...)
RESNET_QUANTIZATION_CONFIG: Dict[str, Any] = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            "data_in_width": 8,
            "data_in_frac_width": 7,
            "weight_width": 8,
            "weight_frac_width": 7,
            "bias_width": 8,
            "bias_frac_width": 0,
        }
    },
}

# MobileNetV3.ipynb — per-layer Optuna categorical
LAYER_PRECISION_CHOICES = ("fp32", "int8", "fp16")


def quant_cfg_fp32() -> Dict[str, Any]:
    return {"name": None}


def quant_cfg_int8() -> Dict[str, Any]:
    return {
        "name": "integer",
        "data_in_width": 8,
        "data_in_frac_width": 7,
        "weight_width": 8,
        "weight_frac_width": 7,
        "bias_width": 8,
        "bias_frac_width": 0,
    }


def quant_cfg_fp16() -> Dict[str, Any]:
    w, exp, eb = 16, 5, 15
    return {
        "name": "minifloat_ieee",
        "data_in_width": w,
        "data_in_exponent_width": exp,
        "data_in_exponent_bias": eb,
        "weight_width": w,
        "weight_exponent_width": exp,
        "weight_exponent_bias": eb,
        "bias_width": w,
        "bias_exponent_width": exp,
        "bias_exponent_bias": eb,
    }
