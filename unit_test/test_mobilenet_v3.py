"""
Tests for MobileNet mixed-precision helpers in ``tiny_imagenet_lib``
(mirrors ``python notebooks/MobileNetV3.ipynb``).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models

from tiny_imagenet_lib import (
    LAYER_PRECISION_CHOICES,
    classification_accuracy_percent,
    evaluate,
    quant_cfg_fp16,
    quant_cfg_fp32,
    quant_cfg_int8,
)


def _tiny_mobilenet(n_cls: int = 200) -> nn.Module:
    m = tv_models.mobilenet_v3_large(weights=None)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, n_cls)
    return m


def _fake_loader(batches):
    return batches


class TestEvaluate:
    def test_perfect_accuracy(self):
        model = _tiny_mobilenet()
        model.eval()
        with torch.no_grad():
            model.classifier[3].weight.zero_()
            model.classifier[3].bias.zero_()
            model.classifier[3].bias[0] = 10.0

        labels_all_zero = torch.zeros(4, dtype=torch.long)
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_all_zero)]

        with torch.no_grad():
            result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert result == pytest.approx(100.0, abs=1e-3)

    def test_zero_accuracy(self):
        model = _tiny_mobilenet()
        model.eval()
        with torch.no_grad():
            model.classifier[3].weight.zero_()
            model.classifier[3].bias.zero_()
            model.classifier[3].bias[0] = 10.0

        labels_all_one = torch.ones(4, dtype=torch.long)
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_all_one)]

        result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert result == pytest.approx(0.0, abs=1e-3)

    def test_half_accuracy(self):
        model = _tiny_mobilenet()
        model.eval()
        with torch.no_grad():
            model.classifier[3].weight.zero_()
            model.classifier[3].bias.zero_()
            model.classifier[3].bias[0] = 10.0

        labels_mixed = torch.tensor([0, 0, 1, 1])
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_mixed)]

        result = evaluate(model, _fake_loader(fake_batches), device="cpu")

        assert result == pytest.approx(50.0, abs=1e-3)

    def test_result_is_percentage(self):
        model = _tiny_mobilenet()
        model.eval()
        fake_batches = [(torch.randn(2, 3, 224, 224), torch.zeros(2, dtype=torch.long))]
        result = evaluate(model, _fake_loader(fake_batches), device="cpu")
        assert 0.0 <= result <= 100.0

    def test_result_is_float(self):
        model = _tiny_mobilenet()
        model.eval()
        fake_batches = [(torch.randn(2, 3, 224, 224), torch.zeros(2, dtype=torch.long))]
        result = evaluate(model, _fake_loader(fake_batches), device="cpu")
        assert isinstance(result, float)


class TestAccuracyFormula:
    def test_formula_gives_percentage_not_fraction(self):
        assert classification_accuracy_percent(1, 1) == pytest.approx(100.0)

    def test_formula_zero_correct(self):
        assert classification_accuracy_percent(0, 10) == pytest.approx(0.0)

    def test_formula_all_correct(self):
        assert classification_accuracy_percent(50, 50) == pytest.approx(100.0)

    def test_formula_half_correct(self):
        assert classification_accuracy_percent(5, 10) == pytest.approx(50.0)

    def test_formula_is_finite(self):
        assert math.isfinite(classification_accuracy_percent(37, 100))


class TestMobileNetHelpers:
    def test_layer_precision_choices(self):
        assert LAYER_PRECISION_CHOICES == ("fp32", "int8", "fp16")

    def test_quant_cfg_fp32(self):
        assert quant_cfg_fp32()["name"] is None

    def test_quant_cfg_int8_integer_scheme(self):
        c = quant_cfg_int8()
        assert c["name"] == "integer"
        assert c["data_in_width"] == 8
        assert c["bias_frac_width"] == 0

    def test_quant_cfg_fp16_minifloat(self):
        c = quant_cfg_fp16()
        assert c["name"] == "minifloat_ieee"
        assert c["data_in_width"] == 16


class TestModelArchitecture:
    def test_output_classes_is_200(self):
        m = _tiny_mobilenet(200)
        assert m.classifier[3].out_features == 200

    def test_forward_shape(self):
        m = _tiny_mobilenet()
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 200)
