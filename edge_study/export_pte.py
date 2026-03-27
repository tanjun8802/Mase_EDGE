"""ExecuTorch .pte export (XNNPACK PT2E)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from executorch.exir import to_edge_transform_and_lower
from torch.utils.data import DataLoader
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchvision import datasets

from edge_study.host_eval import tiny_train_root
from edge_study.quant_xnnpack import (
    resolve_layer_quant_scheme_for_export,
    scheme_to_xnnpack_config,
)


def calibration_input_batches(
    max_batches: int = 4,
    batch_size: int = 8,
) -> List[torch.Tensor]:
    root = tiny_train_root()
    if root.is_dir():
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
        out: List[torch.Tensor] = []
        for i, (imgs, _) in enumerate(loader):
            if i >= max_batches:
                break
            out.append(imgs.cpu())
        if out:
            return out
    return [torch.randn(batch_size, 3, 224, 224)]


def edge_export_model(model: nn.Module, trial_id: int, config: Dict[str, Any]) -> str:
    model.eval().cpu()
    sample_inputs = (torch.randn(1, 3, 224, 224),)
    layer_quant: Dict[str, str] = config.get("layer_quant") or {}
    any_quant = any(scheme != "none" for scheme in layer_quant.values())

    if any_quant:
        quantizer = XNNPACKQuantizer()
        for name, scheme in layer_quant.items():
            if scheme == "none":
                continue
            eff = resolve_layer_quant_scheme_for_export(scheme)
            if eff != scheme:
                warnings.warn(
                    f"Layer {name!r}: {scheme!r} -> {eff!r} for export "
                    "(dynamic activation PT2E ops are not lowered by this to_executorch path).",
                    UserWarning,
                    stacklevel=2,
                )
            quantizer.set_module_name(name, scheme_to_xnnpack_config(eff))

        ep = torch.export.export(model, sample_inputs)
        prepared = prepare_pt2e(ep.module(), quantizer)
        with torch.no_grad():
            for batch in calibration_input_batches():
                for i in range(batch.shape[0]):
                    prepared(batch[i : i + 1])
        quantized_model = convert_pt2e(prepared)
        lowered = to_edge_transform_and_lower(
            torch.export.export(quantized_model, sample_inputs),
            partitioner=[XnnpackPartitioner()],
        )
    else:
        lowered = to_edge_transform_and_lower(
            torch.export.export(model, sample_inputs),
            partitioner=[XnnpackPartitioner()],
        )

    exec_prog = lowered.to_executorch()
    pte_dir = Path("pte_files")
    pte_dir.mkdir(exist_ok=True)
    pte_path = pte_dir / f"mobilenetv3_trial_{trial_id}.pte"
    with open(pte_path, "wb") as f:
        exec_prog.write_to_file(f)
    return str(pte_path)
