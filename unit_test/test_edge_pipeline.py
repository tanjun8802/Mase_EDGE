"""
Unit tests for ``edge_study`` (EDGE Optuna + ExecuTorch pipeline).
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models

from edge_study import (
    QUANT_LAYER_NAMES,
    XNNPACK_LAYER_QUANT_CHOICES,
    _adb_cat_stdout_not_metrics_json,
    _metrics_payload_ready,
    edge_host_val_sanity_check,
    edge_optuna_config,
    edge_optimise_model,
)
from edge_study.model_loaders import _load_resnet18_from_pt
from edge_study.objective import compute_objective_score


def _run_objective(acc: float, latency: float, memory: float) -> float:
    return compute_objective_score(acc, latency, memory)


def _tiny_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )


def _make_optuna_trial(
    *,
    prune_ratio: float = 0.3,
    layer_scheme: str = "none",
):
    """MagicMock trial matching ``edge_optuna_config`` (per-layer categoricals)."""
    trial = MagicMock()
    trial.suggest_float.return_value = prune_ratio

    def _cat(_name: str, choices: list | tuple):
        if layer_scheme in choices:
            return layer_scheme
        return choices[0]

    trial.suggest_categorical.side_effect = _cat
    return trial


def _minimal_edge_config(
    prune_ratio: float = 0.0,
    layer_scheme: str = "none",
) -> dict:
    return {
        "prune_ratio": prune_ratio,
        "backend": "xnnpack",
        "use_mixed_delegation": False,
        "delegation_plan": None,
        "layer_quant": {n: layer_scheme for n in QUANT_LAYER_NAMES},
    }


# --- 1. Objective score formula ---


class TestObjectiveScoreFormula:
    def test_perfect_model_score(self):
        score = _run_objective(1.0, 0.0, 0.0)
        assert score == pytest.approx(1.0, rel=1e-3)

    def test_accuracy_dominates(self):
        high = _run_objective(0.90, 50.0, 300.0)
        low = _run_objective(0.40, 50.0, 300.0)
        assert high > low

    def test_latency_penalty_exists(self):
        fast = _run_objective(0.7, 10.0, 0.0)
        slow = _run_objective(0.7, 500.0, 0.0)
        assert fast > slow

    def test_memory_penalty_exists(self):
        small = _run_objective(0.7, 0.0, 100.0)
        large = _run_objective(0.7, 0.0, 5000.0)
        assert small > large

    def test_score_can_be_negative(self):
        assert _run_objective(0.0, 9999.0, 9999.0) < 0.0

    def test_score_ordering_matches_expected(self):
        best = _run_objective(0.85, 20.0, 400.0)
        middle = _run_objective(0.70, 50.0, 800.0)
        worst = _run_objective(0.50, 200.0, 2000.0)
        assert best > middle > worst


# --- 2. Metrics helpers ---


class TestMetricsPayloadReady:
    def test_status_done_is_ready(self):
        assert _metrics_payload_ready({"status": "done"}) is True

    def test_status_error_is_ready(self):
        assert _metrics_payload_ready({"status": "error"}) is True

    def test_both_required_fields_present_is_ready(self):
        assert _metrics_payload_ready({"top1_acc": 0.75, "latency_p95_ms": 42.0}) is True

    def test_missing_latency_not_ready(self):
        assert _metrics_payload_ready({"top1_acc": 0.75}) is False

    def test_missing_accuracy_not_ready(self):
        assert _metrics_payload_ready({"latency_p95_ms": 42.0}) is False

    def test_empty_dict_not_ready(self):
        assert _metrics_payload_ready({}) is False

    def test_pending_status_not_ready(self):
        assert _metrics_payload_ready({"status": "running"}) is False


class TestAdbCatStdoutNotMetricsJson:
    def test_empty_bytes_is_not_metrics(self):
        assert _adb_cat_stdout_not_metrics_json(b"") is True

    def test_whitespace_only_is_not_metrics(self):
        assert _adb_cat_stdout_not_metrics_json(b"   \n\t  ") is True

    def test_cat_error_prefix_is_not_metrics(self):
        assert _adb_cat_stdout_not_metrics_json(b"cat: files/model.pte: No such file") is True

    def test_no_such_file_message_is_not_metrics(self):
        assert _adb_cat_stdout_not_metrics_json(b"No such file or directory") is True

    def test_permission_denied_is_not_metrics(self):
        assert _adb_cat_stdout_not_metrics_json(b"Permission denied") is True

    def test_valid_json_bytes_is_metrics(self):
        payload = json.dumps({"top1_acc": 0.7, "latency_p95_ms": 35.0}).encode()
        assert _adb_cat_stdout_not_metrics_json(payload) is False


# --- 3. Optuna search space (per-layer XNNPACK static INT8 + prune) ---


class TestEdgeOptunaConfig:
    def test_config_has_required_keys(self):
        cfg = edge_optuna_config(_make_optuna_trial())
        assert {"prune_ratio", "backend", "layer_quant", "use_mixed_delegation", "delegation_plan"} <= cfg.keys()

    def test_backend_is_xnnpack(self):
        assert edge_optuna_config(_make_optuna_trial())["backend"] == "xnnpack"

    def test_mixed_delegation_off(self):
        cfg = edge_optuna_config(_make_optuna_trial())
        assert cfg["use_mixed_delegation"] is False
        assert cfg["delegation_plan"] is None

    def test_prune_ratio_passed_through(self):
        cfg = edge_optuna_config(_make_optuna_trial(prune_ratio=0.55))
        assert cfg["prune_ratio"] == pytest.approx(0.55)

    def test_layer_quant_covers_all_quant_layers(self):
        cfg = edge_optuna_config(_make_optuna_trial())
        assert set(cfg["layer_quant"].keys()) == set(QUANT_LAYER_NAMES)

    def test_layer_quant_values_are_valid_choices(self):
        cfg = edge_optuna_config(_make_optuna_trial(layer_scheme="int8_pc_static"))
        for v in cfg["layer_quant"].values():
            assert v in XNNPACK_LAYER_QUANT_CHOICES

    def test_suggest_float_bounds(self):
        trial = _make_optuna_trial()
        edge_optuna_config(trial)
        trial.suggest_float.assert_called_once_with("prune_ratio", 0.0, 0.7)

    def test_suggest_categorical_per_layer(self):
        trial = _make_optuna_trial()
        edge_optuna_config(trial)
        n_calls = trial.suggest_categorical.call_count
        assert n_calls == len(QUANT_LAYER_NAMES)


# --- 4. ResNet-18 checkpoint loader ---


class TestLoadResnet18FromPt:
    @staticmethod
    def _fn():
        return _load_resnet18_from_pt

    @staticmethod
    def _build_resnet18(n_cls=200):
        m = tv_models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
        return m

    def test_load_raw_state_dict(self, tmp_path):
        src = self._build_resnet18()
        pt = tmp_path / "raw.pt"
        torch.save(src.state_dict(), pt)
        assert self._fn()(pt, "cpu").fc.out_features == 200

    def test_load_wrapped_state_dict(self, tmp_path):
        src = self._build_resnet18()
        pt = tmp_path / "wrapped.pt"
        torch.save({"state_dict": src.state_dict(), "num_classes": 200}, pt)
        assert self._fn()(pt, "cpu").fc.out_features == 200

    def test_load_full_nn_module(self, tmp_path):
        src = self._build_resnet18()
        pt = tmp_path / "full.pt"
        torch.save(src, pt)
        assert isinstance(self._fn()(pt, "cpu"), nn.Module)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._fn()(tmp_path / "nonexistent.pt", "cpu")

    def test_forward_pass_output_shape(self, tmp_path):
        src = self._build_resnet18()
        pt = tmp_path / "forward.pt"
        torch.save(src.state_dict(), pt)
        loaded = self._fn()(pt, "cpu")
        loaded.eval()
        with torch.no_grad():
            out = loaded(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 200)


# --- 5. edge_optimise_model (torch prune only; PT2E in export) ---


class TestEdgeOptimiseModel:
    def test_returns_nn_module_on_cpu(self):
        model = _tiny_model()
        out = edge_optimise_model(_minimal_edge_config(0.0), model=deepcopy(model))
        assert isinstance(out, nn.Module)
        assert next(out.parameters()).device.type == "cpu"

    def test_eval_mode(self):
        model = _tiny_model()
        model.train()
        out = edge_optimise_model(_minimal_edge_config(0.0), model=deepcopy(model))
        assert not out.training

    def test_zero_prune_preserves_weights(self):
        m = _tiny_model()
        ref = deepcopy(m.state_dict())
        out = edge_optimise_model(_minimal_edge_config(0.0), model=m)
        for k in ref:
            assert torch.allclose(ref[k], out.state_dict()[k]), k

    def test_nonzero_prune_changes_weights(self):
        m = _tiny_model()
        ref = deepcopy(m.state_dict())
        out = edge_optimise_model(_minimal_edge_config(0.5), model=m)
        changed = any(
            not torch.allclose(ref[k], out.state_dict()[k]) for k in ref if "weight" in k
        )
        assert changed

    @pytest.mark.parametrize("ratio", [0.0, 0.3, 0.7])
    def test_prune_ratios_do_not_raise(self, ratio):
        m = _tiny_model()
        edge_optimise_model(_minimal_edge_config(ratio), model=deepcopy(m))


# --- 6. Host sanity check ---


class TestEdgeHostValSanityCheck:
    def test_returns_none_when_dataset_missing(self, tmp_path):
        r = edge_host_val_sanity_check(
            nn.Linear(10, 10),
            train_dir=tmp_path / "does_not_exist",
        )
        assert r is None

    def test_accuracy_in_range_with_patched_loader(self, tmp_path):
        (tmp_path / "class_a").mkdir()
        model = _tiny_model()
        model.eval()
        batches = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
        with patch("edge_study.host_eval.datasets.ImageFolder"), patch(
            "edge_study.host_eval.DataLoader",
            return_value=batches,
        ):
            result = edge_host_val_sanity_check(model, train_dir=tmp_path)
        assert result is not None
        assert 0.0 <= result <= 1.0


# --- 7. Score roundtrip ---


class TestConfigToScoreRoundtrip:
    @pytest.mark.parametrize(
        "acc,latency,memory",
        [
            (0.65, 50.0, 300.0),
            (0.80, 20.0, 100.0),
            (0.30, 200.0, 2000.0),
            (0.00, 9999.0, 9999.0),
        ],
    )
    def test_score_is_finite(self, acc, latency, memory):
        assert math.isfinite(_run_objective(acc, latency, memory))
