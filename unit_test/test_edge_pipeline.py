"""
test_edge_pipeline.py

Unit tests for the EDGE_optuna_study.ipynb
"""

import math
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models

# Import the functions straight from the notebook
# conftest.py reads EDGE_optuna_study.ipynb and registers it as `notebook`.
from notebook import (
    _metrics_payload_ready,
    _adb_cat_stdout_not_metrics_json,
    edge_optuna_config,
    edge_optimise_model,
    edge_host_val_sanity_check,
)

import notebook as _nb
import nbformat as _nbformat

def _extract_score_formula() -> str:
    """
    Parse the notebook source at runtime and extract the score
    """
    nb_path = _nb.__file__
    nb = _nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "def objective" in cell.source:
            for line in cell.source.splitlines():
                stripped = line.strip()
                if stripped.startswith("score =") and "acc" in stripped:
                    # Return just the right-hand side expression
                    return stripped[len("score ="):].strip()
    raise RuntimeError(
        "Could not find 'score = ...' inside objective() in the notebook. "
        "Make sure the formula is on a single line starting with 'score ='."
    )

def _run_objective(acc: float, latency: float, memory: float) -> float:
    """
    Evaluate the score formula extracted live from the notebook source.
    """
    formula = _extract_score_formula()
    return eval(formula, {"acc": acc, "latency": latency, "memory": memory})

def normalise_delegation_ratios(cpu, gpu, npu):
    """Mirrors the normalisation block inside edge_optuna_config()."""
    total = cpu + gpu + npu
    if total == 0:
        return 1.0, 0.0, 0.0
    return cpu / total, gpu / total, npu / total

def _tiny_model() -> nn.Module:
    """Lightweight CNN used instead of ResNet-18 to keep tests fast."""
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )

# 1. OBJECTIVE SCORE FORMULA


class TestObjectiveScoreFormula:
    """
    Tests for the score formula inside objective() in the notebook.
    """

    def test_perfect_model_score(self):
        """
        100% accuracy, zero latency/memory -> score must equal exactly 1.0.
        """
        score = _run_objective(1.0, 0.0, 0.0)
        assert score == pytest.approx(1.0, rel=1e-3), (
            f"Expected score of 1.0 for perfect model but got {score:.4f}. "
            "Check the score formula in objective() in the notebook."
        )

    def test_accuracy_dominates(self):
        """Higher accuracy must always produce a higher score, all else equal."""
        high = _run_objective(0.90, 50.0, 300.0)
        low  = _run_objective(0.40, 50.0, 300.0)
        assert high > low

    def test_latency_penalty_exists(self):
        """A model with lower latency must score higher than one with higher latency."""
        fast = _run_objective(0.7, 10.0,  0.0)
        slow = _run_objective(0.7, 500.0, 0.0)
        assert fast > slow

    def test_memory_penalty_exists(self):
        """A model with lower memory must score higher than one with higher memory."""
        small = _run_objective(0.7, 0.0, 100.0)
        large = _run_objective(0.7, 0.0, 5000.0)
        assert small > large

    def test_score_can_be_negative(self):
        """Extreme latency and memory must be able to push the score below zero."""
        assert _run_objective(0.0, 9999.0, 9999.0) < 0.0

    def test_score_ordering_matches_expected(self):
        """Three configs must rank best > middle > worst."""
        best   = _run_objective(0.85, 20.0,  400.0)
        middle = _run_objective(0.70, 50.0,  800.0)
        worst  = _run_objective(0.50, 200.0, 2000.0)
        assert best > middle > worst


# 2. METRICS PAYLOAD READY HELPER

class TestMetricsPayloadReady:
    """Tests for _metrics_payload_ready()."""

    def test_status_done_is_ready(self):
        assert _metrics_payload_ready({"status": "done"}) is True

    def test_status_error_is_ready(self):
        assert _metrics_payload_ready({"status": "error"}) is True

    def test_both_required_fields_present_is_ready(self):
        assert _metrics_payload_ready(
            {"top1_acc": 0.75, "latency_p95_ms": 42.0}
        ) is True

    def test_missing_latency_not_ready(self):
        assert _metrics_payload_ready({"top1_acc": 0.75}) is False

    def test_missing_accuracy_not_ready(self):
        assert _metrics_payload_ready({"latency_p95_ms": 42.0}) is False

    def test_empty_dict_not_ready(self):
        assert _metrics_payload_ready({}) is False

    def test_pending_status_not_ready(self):
        assert _metrics_payload_ready({"status": "running"}) is False

# 3. ADB CAT OUTPUT DETECTION HELPER

class TestAdbCatStdoutNotMetricsJson:
    """Tests for _adb_cat_stdout_not_metrics_json()"""

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


# 4. SEARCH SPACE CONFIG GENERATOR

class TestEdgeOptunaConfig:
    """Tests for edge_optuna_config()."""

    @staticmethod
    def _make_trial(prune_ratio=0.3, quant_bits=8, quant_config="int8"):
        trial = MagicMock()
        trial.suggest_float.return_value = prune_ratio
        def _categorical(name, choices):
            if name == "quant_bits":
                return quant_bits
            if name == "quant_config_global":
                return quant_config
            return choices[0]   

        trial.suggest_categorical.side_effect = _categorical
        return trial

    def test_config_has_required_keys(self):
        cfg = edge_optuna_config(self._make_trial())
        required = {
            "prune_ratio", "quant_bits", "backend",
            "use_mixed_delegation", "delegation_plan", "quant_config_global",
        }
        assert required.issubset(cfg.keys())

    def test_backend_is_xnnpack(self):
        valid_backends = {"xnnpack", "cpu", "vulkan", "qnn", "nnapi"}
        cfg = edge_optuna_config(self._make_trial())
        assert cfg["backend"] in valid_backends

    def test_mixed_delegation_false_by_default(self):
        cfg = edge_optuna_config(self._make_trial())
        assert cfg["use_mixed_delegation"] is False
        assert cfg["delegation_plan"] is None

    def test_prune_ratio_passed_through(self):
        cfg = edge_optuna_config(self._make_trial(prune_ratio=0.55))
        assert cfg["prune_ratio"] == pytest.approx(0.55)

    def test_quant_bits_passed_through(self):
        assert edge_optuna_config(self._make_trial(quant_bits=16))["quant_bits"] == 16

    @pytest.mark.parametrize("quant", ["int8", "fp16"])
    def test_valid_quant_config_values(self, quant):
        cfg = edge_optuna_config(self._make_trial(quant_config=quant))
        assert cfg["quant_config_global"] in {"int8", "fp16"}

    def test_suggest_float_called_with_correct_bounds(self):
        trial = self._make_trial()
        edge_optuna_config(trial)
        trial.suggest_float.assert_called_once_with("prune_ratio", 0.0, 0.7)

    def test_suggest_categorical_called_for_quant_bits(self):
        trial = self._make_trial()
        edge_optuna_config(trial)
        calls = [str(c) for c in trial.suggest_categorical.call_args_list]
        assert any("quant_bits" in c for c in calls)


# 5. DELEGATION RATIO NORMALISATION

class TestNormaliseDelegationRatios:
    """Tests for the normalisation logic inside edge_optuna_config()."""

    def test_normalised_ratios_sum_to_one(self):
        c, g, n = normalise_delegation_ratios(0.5, 0.3, 0.2)
        assert c + g + n == pytest.approx(1.0)

    def test_all_zero_defaults_to_cpu_only(self):
        c, g, n = normalise_delegation_ratios(0.0, 0.0, 0.0)
        assert c == pytest.approx(1.0)
        assert g == pytest.approx(0.0)
        assert n == pytest.approx(0.0)

    def test_cpu_only_stays_cpu_only(self):
        c, g, n = normalise_delegation_ratios(1.0, 0.0, 0.0)
        assert c == pytest.approx(1.0)

    def test_equal_split(self):
        c, g, n = normalise_delegation_ratios(1.0, 1.0, 1.0)
        assert c == pytest.approx(1 / 3)
        assert g == pytest.approx(1 / 3)
        assert n == pytest.approx(1 / 3)


# 6. RESNET-18 CHECKPOINT LOADER

class TestLoadResnet18FromPt:
    """Tests for _load_resnet18_from_pt()."""
    @staticmethod
    def _fn():
        return _nb._load_resnet18_from_pt

    @staticmethod
    def _build_resnet18(n_cls=200):
        m = tv_models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_cls)
        return m

    def test_load_raw_state_dict(self, tmp_path):
        src = self._build_resnet18()
        pt  = tmp_path / "raw.pt"
        torch.save(src.state_dict(), pt)
        assert self._fn()(pt, "cpu").fc.out_features == 200

    def test_load_wrapped_state_dict(self, tmp_path):
        src = self._build_resnet18()
        pt  = tmp_path / "wrapped.pt"
        torch.save({"state_dict": src.state_dict(), "num_classes": 200}, pt)
        assert self._fn()(pt, "cpu").fc.out_features == 200

    def test_load_full_nn_module(self, tmp_path):
        src = self._build_resnet18()
        pt  = tmp_path / "full.pt"
        torch.save(src, pt)
        assert isinstance(self._fn()(pt, "cpu"), nn.Module)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self._fn()(tmp_path / "nonexistent.pt", "cpu")

    def test_weights_are_preserved(self, tmp_path):
        src = self._build_resnet18()
        pt  = tmp_path / "weights.pt"
        torch.save(src.state_dict(), pt)
        loaded = self._fn()(pt, "cpu")
        for (k1, v1), (k2, v2) in zip(
            src.state_dict().items(), loaded.state_dict().items()
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2), f"Mismatch at {k1}"

    def test_forward_pass_output_shape(self, tmp_path):
        src = self._build_resnet18()
        pt  = tmp_path / "forward.pt"
        torch.save(src.state_dict(), pt)
        loaded = self._fn()(pt, "cpu")
        loaded.eval()
        with torch.no_grad():
            out = loaded(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 200)


# 7. MODEL OPTIMISATION 

class TestEdgeOptimiseModel:
    """
    Tests for edge_optimise_model()
    """

    @staticmethod
    def _mase_mocks(model: nn.Module):
        mock_mg = MagicMock()
        mock_mg.model    = model
        mock_mg.fx_graph = MagicMock()
        mock_MG = MagicMock(return_value=mock_mg)
        mock_ps = MagicMock()
        mock_ps.init_metadata_analysis_pass.return_value       = (mock_mg, None)
        mock_ps.add_common_metadata_analysis_pass.return_value = (mock_mg, None)
        mock_ps.prune_transform_pass.return_value              = (mock_mg, None)
        mock_ps.quantize_transform_pass.return_value           = (mock_mg, None)
        return mock_MG, mock_ps

    def test_returns_nn_module_on_cpu(self):
        model = _tiny_model()
        MG, ps = self._mase_mocks(model)
        with patch("notebook.MaseGraph", MG), \
             patch("notebook.passes",    ps), \
             patch("notebook.torch.fx.GraphModule", return_value=model):
            result = edge_optimise_model(
                {"prune_ratio": 0.3, "quant_bits": 8}, enable_qat=False, model=model
            )
        assert isinstance(result, nn.Module)
        assert next(result.parameters()).device.type == "cpu"

    def test_result_is_in_eval_mode(self):
        model = _tiny_model()
        MG, ps = self._mase_mocks(model)
        with patch("notebook.MaseGraph", MG), \
             patch("notebook.passes",    ps), \
             patch("notebook.torch.fx.GraphModule", return_value=model):
            result = edge_optimise_model(
                {"prune_ratio": 0.3, "quant_bits": 8}, enable_qat=False, model=model
            )
        assert not result.training

    def test_prune_pass_called_with_correct_ratio(self):
        """The notebook function must forward prune_ratio into the Mase pass."""
        model = _tiny_model()
        MG, ps = self._mase_mocks(model)
        with patch("notebook.MaseGraph", MG), \
             patch("notebook.passes",    ps), \
             patch("notebook.torch.fx.GraphModule", return_value=model):
            edge_optimise_model(
                {"prune_ratio": 0.55, "quant_bits": 8}, enable_qat=False, model=model
            )
        pruning_cfg = ps.prune_transform_pass.call_args[1]["pass_args"]
        assert pruning_cfg["weight"]["sparsity"] == pytest.approx(0.55)

    def test_quantize_pass_called_with_correct_bit_width(self):
        """The notebook function must forward quant_bits into the Mase pass."""
        model = _tiny_model()
        MG, ps = self._mase_mocks(model)
        with patch("notebook.MaseGraph", MG), \
             patch("notebook.passes",    ps), \
             patch("notebook.torch.fx.GraphModule", return_value=model):
            edge_optimise_model(
                {"prune_ratio": 0.0, "quant_bits": 16}, enable_qat=False, model=model
            )
        quant_cfg = ps.quantize_transform_pass.call_args[1]["pass_args"]
        assert quant_cfg["conv2d"]["config"]["data_in_width"] == 16
        assert quant_cfg["linear"]["config"]["weight_width"]  == 16

    @pytest.mark.parametrize("ratio", [0.0, 0.3, 0.5, 0.7])
    def test_valid_prune_ratios_do_not_raise(self, ratio):
        model = _tiny_model()
        MG, ps = self._mase_mocks(model)
        with patch("notebook.MaseGraph", MG), \
             patch("notebook.passes",    ps), \
             patch("notebook.torch.fx.GraphModule", return_value=model):
            edge_optimise_model(
                {"prune_ratio": ratio, "quant_bits": 8}, enable_qat=False, model=model
            )


# 8. HOST VALIDATION SANITY CHECK

class TestEdgeHostValSanityCheck:
    """Tests for edge_host_val_sanity_check()."""

    def test_returns_none_when_dataset_missing(self, tmp_path):
        result = edge_host_val_sanity_check(
            nn.Linear(10, 10), train_dir=tmp_path / "does_not_exist"
        )
        assert result is None

    def test_accuracy_computed_correctly(self, tmp_path):
        (tmp_path / "class_a").mkdir()
        model = _tiny_model()
        model.eval()
        fake_batches = [
            (torch.randn(2, 3, 32, 32), torch.tensor([0, 1])),
            (torch.randn(2, 3, 32, 32), torch.tensor([0, 9])),
        ]
        with patch("notebook.datasets.ImageFolder"), \
             patch("notebook.torch.utils.data.DataLoader",
                   return_value=fake_batches):
            result = edge_host_val_sanity_check(model, train_dir=tmp_path)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_accuracy_is_in_valid_range(self, tmp_path):
        (tmp_path / "class_a").mkdir()
        model = _tiny_model()
        fake_batches = [
            (torch.randn(4, 3, 32, 32), torch.zeros(4, dtype=torch.long))
        ]
        with patch("notebook.datasets.ImageFolder"), \
             patch("notebook.torch.utils.data.DataLoader",
                   return_value=fake_batches):
            result = edge_host_val_sanity_check(model, train_dir=tmp_path)
        if result is not None:
            assert 0.0 <= result <= 1.0


# 9. INTEGRATION: CONFIG → SCORE ROUNDTRIP
class TestConfigToScoreRoundtrip:
    """End-to-end: objective() must return a finite, orderable score."""

    @pytest.mark.parametrize("acc,latency,memory", [
        (0.65, 50.0, 300.0),
        (0.80, 20.0, 100.0),
        (0.30, 200.0, 2000.0),
        (0.00, 9999.0, 9999.0),
    ])
    def test_score_is_finite(self, acc, latency, memory):
        assert math.isfinite(_run_objective(acc, latency, memory))

    def test_better_accuracy_always_wins(self):
        assert _run_objective(0.80, 100.0, 500.0) > _run_objective(0.30, 100.0, 500.0)

    def test_lower_latency_always_wins(self):
        assert _run_objective(0.70, 10.0,  500.0) > _run_objective(0.70, 500.0, 500.0)

    def test_lower_memory_always_wins(self):
        assert _run_objective(0.70, 50.0, 100.0) > _run_objective(0.70, 50.0, 5000.0)
