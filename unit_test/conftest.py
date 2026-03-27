"""
Pytest setup: repo root on ``sys.path``, stubs for ExecuTorch / torchao / optuna when missing.

Tests import ``edge_study`` and ``tiny_imagenet_lib`` directly (no notebook execution).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _register_pkg(dotted: str) -> types.ModuleType:
    parts = dotted.split(".")
    mod: types.ModuleType | None = None
    for i in range(len(parts)):
        name = ".".join(parts[: i + 1])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
        mod = sys.modules[name]
    assert mod is not None
    return mod


def _inject_executorch_torchao_stubs() -> None:
    """Stub modules so ``edge_study`` imports without a full ExecuTorch install in CI."""
    xq = _register_pkg("executorch.backends.xnnpack.quantizer.xnnpack_quantizer")

    class _XNNPACKQuantizer:
        def set_global(self, *_a, **_kw):
            return self

        def set_module_name(self, *_a, **_kw):
            return self

    def _get_symmetric_quantization_config(**_kwargs):
        return MagicMock(name="QuantizationConfig")

    xq.XNNPACKQuantizer = _XNNPACKQuantizer
    xq.get_symmetric_quantization_config = _get_symmetric_quantization_config

    part = _register_pkg("executorch.backends.xnnpack.partition.xnnpack_partitioner")
    part.XnnpackPartitioner = MagicMock(return_value=MagicMock())

    exir = _register_pkg("executorch.exir")

    def _lowered():
        ep = MagicMock()

        def _write(_f):
            return None

        ep.to_executorch = MagicMock(return_value=MagicMock(write_to_file=_write))
        return ep

    exir.to_edge_transform_and_lower = MagicMock(side_effect=lambda *_a, **_k: _lowered())

    ta = _register_pkg("torchao.quantization.pt2e.quantize_pt2e")
    ta.prepare_pt2e = MagicMock(side_effect=lambda mod, _q: mod)
    ta.convert_pt2e = MagicMock(side_effect=lambda m: m)


_inject_executorch_torchao_stubs()


def _install_optuna_stub() -> None:
    o = types.ModuleType("optuna")
    o.samplers = types.ModuleType("optuna.samplers")
    o.visualization = types.ModuleType("optuna.visualization")
    o.trial = types.ModuleType("optuna.trial")
    o.samplers.TPESampler = MagicMock(return_value=MagicMock())
    o.create_study = MagicMock(
        return_value=MagicMock(
            best_trial=MagicMock(number=0, user_attrs={}, params={}),
            best_value=0.0,
            optimize=MagicMock(),
        )
    )
    o.trial.FixedTrial = MagicMock
    o.visualization.plot_optimization_history = MagicMock(
        return_value=MagicMock(show=MagicMock())
    )
    sys.modules.setdefault("optuna", o)
    sys.modules.setdefault("optuna.samplers", o.samplers)
    sys.modules.setdefault("optuna.visualization", o.visualization)
    sys.modules.setdefault("optuna.trial", o.trial)


_install_optuna_stub()

_torchinfo = types.ModuleType("torchinfo")


def _noop_summary(*_a, **_k):
    return None


_torchinfo.summary = _noop_summary
sys.modules.setdefault("torchinfo", _torchinfo)
