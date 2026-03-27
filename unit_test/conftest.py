"""
conftest.py

Single pytest configuration that loads both notebooks:

  - EDGE_optuna_study.ipynb  → registered as `notebook`
  - ResNet18.ipynb           → registered as `resnet18_nb`

Tests import real functions directly from these modules:
    from notebook    import _metrics_payload_ready, edge_optuna_config, ...
    from resnet18_nb import evaluate, ...
"""

import sys
import types
import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torchvision.models as tv_models
import nbformat

HERE        = Path(__file__).parent          # unit_test/
PROJECT_ROOT = HERE.parent                   # Mase_EDGE/

#  mase / chop 
_passes_mock = MagicMock()
_chop_mock   = MagicMock()
_chop_mock.passes = _passes_mock

_mock_mg = MagicMock()
_mock_mg.model    = MagicMock()
_mock_mg.fx_graph = MagicMock()
_chop_mock.MaseGraph = MagicMock(return_value=_mock_mg)
_passes_mock.init_metadata_analysis_pass.return_value       = (_mock_mg, None)
_passes_mock.add_common_metadata_analysis_pass.return_value = (_mock_mg, None)
_passes_mock.prune_transform_pass.return_value              = (_mock_mg, None)
_passes_mock.quantize_transform_pass.return_value           = (_mock_mg, None)

sys.modules.setdefault("chop",                 _chop_mock)
sys.modules.setdefault("chop.passes",          _passes_mock)
sys.modules.setdefault("mase",                 MagicMock())
sys.modules.setdefault("mase.src",             MagicMock())
sys.modules.setdefault("mase.src.chop",        _chop_mock)
sys.modules.setdefault("mase.src.chop.passes", _passes_mock)

_et = MagicMock()
for _mod in [
    "executorch", "executorch.exir", "executorch.exir.backend",
    "executorch.exir.backend.backend_api", "executorch.backends",
    "executorch.backends.xnnpack", "executorch.backends.xnnpack.partition",
    "executorch.backends.xnnpack.partition.xnnpack_partitioner",
]:
    sys.modules.setdefault(_mod, _et)

_edge = MagicMock()
_edge.get_adb_path.return_value = "/usr/bin/adb"
sys.modules.setdefault("EDGE_device",                       _edge)
sys.modules.setdefault("EDGE_device.device_specifications", _edge)


_optuna = MagicMock()
sys.modules.setdefault("optuna",               _optuna)
sys.modules.setdefault("optuna.samplers",      _optuna)
sys.modules.setdefault("optuna.visualization", _optuna)


def _load_notebook(notebook_path, module_name, extra_patches=()):
    """
    Read notebook_path, execute every code cell into a fresh module called
    module_name, register it in sys.modules, and return it.

    extra_patches: list of patch() context managers active during execution.
    """
    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Cannot find {notebook_path}. "
            f"Place conftest.py in the same directory as {notebook_path.name}."
        )

    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    sys.modules[module_name] = module

    nb = nbformat.read(str(notebook_path), as_version=4)
    skipped = []

    def _run_cells():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            clean = "\n".join(
                line for line in cell.source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            ).strip()
            if not clean:
                continue
            try:
                exec(compile(clean, f"{notebook_path}:cell{i}", "exec"),
                     module.__dict__)
            except Exception as exc:
                skipped.append((i, type(exc).__name__, str(exc)[:120]))

    if extra_patches:
        with contextlib.ExitStack() as stack:
            for p in extra_patches:
                stack.enter_context(p)
            _run_cells()
    else:
        _run_cells()

    if skipped:
        print(f"\n[conftest:{module_name}] Cells skipped "
              "(hardware / missing files / optional deps):")
        for idx, ename, emsg in skipped:
            print(f"  Cell {idx}: {ename}: {emsg}")
        print()

    captured = [k for k, v in module.__dict__.items()
                if callable(v) and not k.startswith("__")]
    print(f"[conftest:{module_name}] Functions captured: {captured}\n")
    return module

# 1. Load EDGE_optuna_study.ipynb  →  module `notebook`


_load_notebook(
        notebook_path = PROJECT_ROOT / "EDGE_optuna_study.ipynb",
        module_name   = "notebook",
)

# 2. Load ResNet18.ipynb  →  module `resnet18_nb`


# Save the real resnet18 BEFORE patching to avoid infinite recursion
_orig_resnet18 = tv_models.resnet18
def _fake_resnet18(weights=None, **kw):
    return _orig_resnet18(weights=None)   # real model, no download

_fake_img    = torch.randn(2, 3, 224, 224)
_fake_labels = torch.zeros(2, dtype=torch.long)

class _FakeDataset:
    def __len__(self):        return 2
    def __getitem__(self, i): return _fake_img[0], _fake_labels[0]

class _FakeLoader:
    def __init__(self, *a, **kw): pass
    def __iter__(self):  yield _fake_img, _fake_labels
    def __len__(self):   return 1

_load_notebook(
    notebook_path = PROJECT_ROOT / "python notebooks" / "ResNet18.ipynb",
    module_name   = "resnet18_nb",
    extra_patches = [
        patch("torchvision.datasets.ImageFolder",
              side_effect=lambda root, **kw: _FakeDataset()),
        patch("torch.utils.data.DataLoader",      _FakeLoader),
        patch("torchvision.models.resnet18",       _fake_resnet18),
        patch("torchvision.models.ResNet18_Weights", MagicMock(DEFAULT=None)),
        patch("torch.onnx.export"),
    ],
)
