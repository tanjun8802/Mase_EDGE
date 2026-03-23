"""
Load torchvision ResNet-18 parameters from an ONNX file.

Handles:

1. **Named initializers** (``fc.weight``, optional ``model.`` / ``._`` variants, ``fc`` transpose).
2. **Fused inference ONNX** with anonymous ``onnx::Conv_*`` tensors: Conv weights are applied in
   **ONNX graph order** to ``Conv2d`` modules in ``named_modules()`` order (matches torchvision
   export). BatchNorm nodes are absent; we **reset every BatchNorm2d to identity** so
   ``BN(Conv(x)) == Conv(x)`` in eval mode.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn

_PREFIXES = (
    "model.",
    "module.",
    "net.",
    "resnet.",
    "resnet18.",
    "network.",
    "backbone.",
)


def _param_name_candidates(param_name: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        x = x.lstrip("/")
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(param_name)
    add(param_name.replace(".", "_"))
    for p in _PREFIXES:
        add(p + param_name)
        add(p + param_name.replace(".", "_"))
    if "::" in param_name:
        add(param_name.split("::")[-1])
    return out


def _match_fc_weight(t: torch.Tensor, expected: torch.Size) -> torch.Tensor | None:
    if t.shape == expected:
        return t
    if t.ndim == 2 and len(expected) == 2 and t.shape == (expected[1], expected[0]):
        return t.t().contiguous()
    return None


def _reset_batchnorm2d_to_identity(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()


def _conv2d_modules_in_order(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    return [(n, mod) for n, mod in model.named_modules() if isinstance(mod, nn.Conv2d)]


def _load_conv_weights_from_onnx_graph(
    model: nn.Module,
    onnx_model: onnx.ModelProto,
    onnx_tensors: dict[str, object],
) -> int:
    init_names = set(onnx_tensors)
    conv_nodes = [n for n in onnx_model.graph.node if n.op_type == "Conv"]
    conv_modules = _conv2d_modules_in_order(model)
    if len(conv_nodes) != len(conv_modules):
        print(
            f"WARN: ONNX Conv count {len(conv_nodes)} != model Conv2d count {len(conv_modules)}; "
            "skip graph-order conv load."
        )
        return 0

    n = 0
    with torch.no_grad():
        for node, (_, mod) in zip(conv_nodes, conv_modules, strict=True):
            init_inputs = [inp for inp in node.input if inp in init_names]
            if not init_inputs:
                print(f"WARN: Conv node {node.name!r} has no weight initializer; skip.")
                continue
            w_name = init_inputs[0]
            arr = onnx_tensors[w_name]
            t = torch.as_tensor(arr.copy())
            if t.dtype != mod.weight.dtype:
                t = t.to(dtype=mod.weight.dtype)
            if t.shape != mod.weight.shape:
                print(
                    f"WARN: conv weight shape mismatch {mod.weight.shape!r} vs ONNX {tuple(t.shape)} "
                    f"for node {node.name!r}"
                )
                continue
            mod.weight.copy_(t)
            n += 1
    return n


def load_resnet18_weights_from_onnx(
    model: nn.Module,
    onnx_path: Path,
    *,
    verbose: bool = True,
) -> int:
    """
    Copy ONNX weights into *model* (torchvision ResNet-18 + custom ``fc``).

    Returns a rough count of tensors applied (named + conv-by-graph).
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX weights not found: {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))
    onnx_tensors: dict[str, object] = {
        init.name.lstrip("/"): numpy_helper.to_array(init) for init in onnx_model.graph.initializer
    }

    sd = model.state_dict()
    used_onnx: set[str] = set()
    to_load: dict[str, torch.Tensor] = {}

    for key, param in sd.items():
        if "num_batches_tracked" in key:
            continue

        chosen: torch.Tensor | None = None
        chosen_onnx_key: str | None = None

        for cand in _param_name_candidates(key):
            if cand not in onnx_tensors or cand in used_onnx:
                continue
            arr = onnx_tensors[cand]
            t = torch.as_tensor(arr.copy())
            if t.dtype != param.dtype:
                t = t.to(dtype=param.dtype)

            if t.shape == param.shape:
                chosen, chosen_onnx_key = t, cand
                break
            if key == "fc.weight" or key.endswith(".fc.weight"):
                fixed = _match_fc_weight(t, param.shape)
                if fixed is not None:
                    chosen, chosen_onnx_key = fixed, cand
                    break

        if chosen is not None and chosen_onnx_key is not None:
            to_load[key] = chosen
            used_onnx.add(chosen_onnx_key)

    if to_load:
        model.load_state_dict(to_load, strict=False)

    generic_conv = any(k.startswith("onnx::Conv_") for k in onnx_tensors)
    n_conv = len(_conv2d_modules_in_order(model))
    graph_n = 0
    if generic_conv or (len(to_load) < 10 and n_conv == 20):
        graph_n = _load_conv_weights_from_onnx_graph(model, onnx_model, onnx_tensors)
        if graph_n > 0:
            _reset_batchnorm2d_to_identity(model)

    if verbose:
        if graph_n > 0:
            print(
                f"ONNX load: {len(to_load)} tensor(s) by name, {graph_n} Conv weights by graph order; "
                f"BatchNorm2d layers set to identity (fused ONNX → unfused ResNet)."
            )
        elif len(to_load) == 0:
            print("WARN: ONNX load: no tensors matched by name.")
            print(f"      Sample initializer names: {sorted(onnx_tensors.keys())[:30]}")
        else:
            n_expect = sum(1 for k in sd if "num_batches_tracked" not in k)
            still = n_expect - len(to_load)
            if still > 0:
                print(
                    f"WARN: ONNX load: {len(to_load)}/{n_expect} params by name only; "
                    f"{still} keys unchanged (no onnx::Conv_* graph fallback?)."
                )
            else:
                print(f"ONNX load: {len(to_load)} params matched by name.")

    return len(to_load) + graph_n
