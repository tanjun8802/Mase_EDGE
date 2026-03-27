"""
Tests for ResNet18 Tiny ImageNet helpers in ``tiny_imagenet_lib``
(mirrors ``python notebooks/ResNet18.ipynb``).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models

from tiny_imagenet_lib import (
    RESNET_QUANTIZATION_CONFIG,
    classification_accuracy_percent,
    evaluate,
)


def _tiny_resnet(n_cls=200) -> nn.Module:
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m


def _fake_loader(batches):
    return batches


class TestEvaluate:
    def test_perfect_accuracy(self):
        model = _tiny_resnet()
        model.eval()

        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0

        labels_all_zero = torch.zeros(4, dtype=torch.long)
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_all_zero)]

        with torch.no_grad():
            result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert result == pytest.approx(100.0, abs=1e-3)

    def test_zero_accuracy(self):
        model = _tiny_resnet()
        model.eval()

        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0

        labels_all_one = torch.ones(4, dtype=torch.long)
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_all_one)]

        result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert result == pytest.approx(0.0, abs=1e-3)

    def test_half_accuracy(self):
        model = _tiny_resnet()
        model.eval()

        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0

        labels_mixed = torch.tensor([0, 0, 1, 1])
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_mixed)]

        result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert result == pytest.approx(50.0, abs=1e-3)

    def test_result_is_percentage(self):
        model = _tiny_resnet()
        model.eval()
        fake_batches = [(torch.randn(2, 3, 224, 224), torch.zeros(2, dtype=torch.long))]

        result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert 0.0 <= result <= 100.0

    def test_result_is_float(self):
        model = _tiny_resnet()
        model.eval()
        fake_batches = [(torch.randn(2, 3, 224, 224), torch.zeros(2, dtype=torch.long))]

        result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert isinstance(result, float)

    def test_accumulates_across_batches(self):
        model = _tiny_resnet()
        model.eval()

        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0

        batch1 = (torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
        batch2 = (torch.randn(4, 3, 224, 224), torch.ones(4, dtype=torch.long))

        result = evaluate(model, _fake_loader([batch1, batch2]), device="cpu")

        assert result == pytest.approx(50.0, abs=1e-3)


class TestAccuracyFormula:
    def test_formula_gives_percentage_not_fraction(self):
        result = classification_accuracy_percent(correct=1, total=1)
        assert result == pytest.approx(100.0)

    def test_formula_zero_correct(self):
        assert classification_accuracy_percent(correct=0, total=10) == pytest.approx(0.0)

    def test_formula_all_correct(self):
        assert classification_accuracy_percent(correct=50, total=50) == pytest.approx(100.0)

    def test_formula_half_correct(self):
        assert classification_accuracy_percent(correct=5, total=10) == pytest.approx(50.0)

    def test_formula_is_finite(self):
        assert math.isfinite(classification_accuracy_percent(correct=37, total=100))


class TestModelArchitecture:
    def test_output_classes_is_200(self):
        m = _tiny_resnet(n_cls=200)
        assert m.fc.out_features == 200

    def test_input_features_unchanged(self):
        m = _tiny_resnet()
        assert m.fc.in_features == 512

    def test_model_forward_shape(self):
        m = _tiny_resnet()
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 200)

    def test_model_is_nn_module(self):
        assert isinstance(_tiny_resnet(), nn.Module)

    def test_model_has_fc_layer(self):
        m = _tiny_resnet()
        assert hasattr(m, "fc")
        assert isinstance(m.fc, nn.Linear)


class TestQuantisationConfig:
    @pytest.fixture(autouse=True)
    def load_config(self):
        self.cfg = RESNET_QUANTIZATION_CONFIG

    def test_quantisation_by_type(self):
        assert self.cfg["by"] == "type"

    def test_default_layers_unquantised(self):
        assert self.cfg["default"]["config"]["name"] is None

    def test_linear_uses_integer_scheme(self):
        assert self.cfg["linear"]["config"]["name"] == "integer"

    def test_linear_data_width_is_valid(self):
        valid_widths = {4, 8, 16, 32}
        width = self.cfg["linear"]["config"]["data_in_width"]
        assert width in valid_widths

    def test_linear_weight_width_is_valid(self):
        valid_widths = {4, 8, 16, 32}
        width = self.cfg["linear"]["config"]["weight_width"]
        assert width in valid_widths

    def test_bias_frac_width_is_zero(self):
        assert self.cfg["linear"]["config"]["bias_frac_width"] == 0

    def test_frac_widths_less_than_total_widths(self):
        linear_cfg = self.cfg["linear"]["config"]
        assert linear_cfg["data_in_frac_width"] < linear_cfg["data_in_width"]
        assert linear_cfg["weight_frac_width"] < linear_cfg["weight_width"]

    def test_bit_widths_are_positive(self):
        linear_cfg = self.cfg["linear"]["config"]
        for field in ("data_in_width", "weight_width", "bias_width"):
            assert linear_cfg[field] > 0


class TestOnnxExportPath:
    def test_output_dir_name(self):
        output_dir = Path("./mase_output")
        assert output_dir.name == "mase_output"

    def test_onnx_filename(self):
        output_dir = Path("./mase_output")
        onnx_path = output_dir / "resnet18_qat_fp32.onnx"
        assert onnx_path.name == "resnet18_qat_fp32.onnx"

    def test_onnx_extension(self):
        output_dir = Path("./mase_output")
        onnx_path = output_dir / "resnet18_qat_fp32.onnx"
        assert onnx_path.suffix == ".onnx"

    def test_output_dir_is_relative(self):
        output_dir = Path("./mase_output")
        assert not output_dir.is_absolute()
