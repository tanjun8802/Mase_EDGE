"""Checkpoint loading for ResNet18 / MobileNet V3 (Tiny ImageNet)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models

from edge_study import settings


def _load_resnet18_from_pt(pt_path: Path, load_device: str) -> nn.Module:
    if not pt_path.is_file():
        raise FileNotFoundError(
            f"Missing {pt_path!s}. Add resnet18_qat_fp32.pt under checkpoints/ (see ResNet18.ipynb)."
        )
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    n_cls = 200
    if isinstance(ckpt, nn.Module):
        sd = ckpt.state_dict()
        if hasattr(ckpt, "fc") and isinstance(ckpt.fc, nn.Linear):
            n_cls = int(ckpt.fc.out_features)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        inner = ckpt["state_dict"]
        n_cls = int(ckpt.get("num_classes", 200))
        if isinstance(inner, nn.Module):
            sd = inner.state_dict()
            if hasattr(inner, "fc") and isinstance(inner.fc, nn.Linear):
                n_cls = int(inner.fc.out_features)
        else:
            sd = inner
    elif isinstance(ckpt, dict):
        sd = ckpt
        if "fc.bias" in sd:
            n_cls = int(sd["fc.bias"].shape[0])
    else:
        raise TypeError(f"Unsupported checkpoint type {type(ckpt)}.")
    if not isinstance(sd, dict):
        raise TypeError(f"Expected state_dict dict-like, got {type(sd)}.")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_cls)
    model.load_state_dict(sd, strict=True)
    return model.to(load_device)


def _load_mobilenetv3_from_pt(pt_path: Path, load_device: str) -> nn.Module:
    if not pt_path.is_file():
        raise FileNotFoundError(
            f"Missing {pt_path!s}. Add mobilenet checkpoint under checkpoints/ (see MobileNetV3.ipynb)."
        )
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    n_cls = 200

    def _ncls_from_classifier_module(m: nn.Module) -> Optional[int]:
        if hasattr(m, "classifier") and isinstance(m.classifier, nn.Sequential):
            last = m.classifier[-1]
            if isinstance(last, nn.Linear):
                return int(last.out_features)
        return None

    if isinstance(ckpt, nn.Module):
        sd = ckpt.state_dict()
        n = _ncls_from_classifier_module(ckpt)
        if n is not None:
            n_cls = n
        elif hasattr(ckpt, "fc") and isinstance(ckpt.fc, nn.Linear):
            n_cls = int(ckpt.fc.out_features)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        inner = ckpt["state_dict"]
        n_cls = int(ckpt.get("num_classes", n_cls))
        if isinstance(inner, nn.Module):
            sd = inner.state_dict()
            n = _ncls_from_classifier_module(inner)
            if n is not None:
                n_cls = n
            elif hasattr(inner, "fc") and isinstance(inner.fc, nn.Linear):
                n_cls = int(inner.fc.out_features)
        else:
            sd = inner
    elif isinstance(ckpt, dict):
        sd = ckpt
        if "classifier.3.bias" in sd:
            n_cls = int(sd["classifier.3.bias"].shape[0])
        elif "fc.bias" in sd:
            n_cls = int(sd["fc.bias"].shape[0])
    else:
        raise TypeError(f"Unsupported checkpoint type {type(ckpt)}.")
    if not isinstance(sd, dict):
        raise TypeError(f"Expected state_dict dict-like, got {type(sd)}.")
    if "classifier.3.bias" in sd:
        n_cls = int(sd["classifier.3.bias"].shape[0])
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, n_cls)
    model.load_state_dict(sd, strict=True)
    return model.to(load_device)


def _new_module_from_cpu_state_dict(sd: Dict[str, torch.Tensor]) -> nn.Module:
    keys = set(sd.keys())
    if "classifier.3.weight" in keys or any(k.startswith("features.") for k in keys):
        m = models.mobilenet_v3_large(weights=None)
        n_cls = int(sd["classifier.3.bias"].shape[0])
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, n_cls)
        m.load_state_dict(sd, strict=True)
        return m
    if "fc.weight" in keys:
        m = models.resnet18(weights=None)
        n_cls = int(sd["fc.bias"].shape[0])
        m.fc = nn.Linear(m.fc.in_features, n_cls)
        m.load_state_dict(sd, strict=True)
        return m
    raise ValueError(
        "BASE_STATE_DICT is neither MobileNet V3 Large nor ResNet18; "
        f"sample keys: {list(sd.keys())[:10]}"
    )


def load_model_and_train() -> nn.Module:
    """Load base checkpoint once; each call returns a fresh FP32 module (same arch)."""
    if settings.BASE_STATE_DICT is None:
        print(f"LOADING BASE WEIGHTS FROM {settings.MOBILENET_V3_PATH.name}")
        load_device = (
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = _load_mobilenetv3_from_pt(settings.MOBILENET_V3_PATH, load_device)
        model.eval()
        print("Model is prepared.....")
        settings.BASE_STATE_DICT = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    load_device = (
        "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    assert settings.BASE_STATE_DICT is not None
    m = _new_module_from_cpu_state_dict(settings.BASE_STATE_DICT)
    return m.to(load_device)
