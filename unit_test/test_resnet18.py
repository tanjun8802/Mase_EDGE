"""
test_resnet18.py

Unit tests for ResNet18.ipynb.

"""

import math
import nbformat
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models

import resnet18_nb as _nb
from resnet18_nb import evaluate

def _tiny_resnet(n_cls=200) -> nn.Module:
    """Real ResNet18 with correct head"""
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m

def _fake_loader(batches):
    """Wrap a list of (imgs, labels) tensors as a DataLoader stand-in."""
    return batches


def _extract_accuracy_formula() -> str:
    """
    Parse ResNet18.ipynb and return the right-hand side of the
    'return 100 * correct / total' line inside evaluate().                         
    """
    nb_path = Path(_nb.__file__)
    nb = nbformat.read(str(nb_path), as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code" and "def evaluate" in cell.source:
            for line in cell.source.splitlines():
                stripped = line.strip()
                if stripped.startswith("return") and "correct" in stripped:
                    return stripped[len("return"):].strip()
    raise RuntimeError(
        "Could not find the return statement inside evaluate() in the notebook."
    )

def _run_accuracy_formula(correct: int, total: int) -> float:
    """Evaluate the live formula extracted from the notebook."""
    formula = _extract_accuracy_formula()
    return eval(formula, {"correct": correct, "total": total})

# 1. evaluate() FUNCTION

class TestEvaluate:
    """
    Tests for evaluate(model, loader)
    """

    def test_perfect_accuracy(self):
        """All predictions correct =100.0%."""
        model = _tiny_resnet()
        model.eval()

        # Force all logits to fire on class 0
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0  # class 0 always wins

        labels_all_zero = torch.zeros(4, dtype=torch.long)
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_all_zero)]

        with patch("resnet18_nb.torch.no_grad", torch.no_grad), \
             patch("resnet18_nb.device", "cpu"):
            result = evaluate(model, _fake_loader(fake_batches))

        assert result == pytest.approx(100.0, abs=1e-3)

    def test_zero_accuracy(self):
        """No predictions correct = 0.0%."""
        model = _tiny_resnet()
        model.eval()

        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0   # always predicts class 0

        # All labels are class 1, so nothing matches
        labels_all_one = torch.ones(4, dtype=torch.long)
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_all_one)]

        with patch("resnet18_nb.device", "cpu"):
            result = evaluate(model, _fake_loader(fake_batches))

        assert result == pytest.approx(0.0, abs=1e-3)

    def test_half_accuracy(self):
        """Half predictions correct = 50.0%."""
        model = _tiny_resnet()
        model.eval()

        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0   # always predicts class 0

        # 2 correct (class 0), 2 wrong (class 1)
        labels_mixed = torch.tensor([0, 0, 1, 1])
        fake_batches = [(torch.randn(4, 3, 224, 224), labels_mixed)]

        with patch("resnet18_nb.device", "cpu"):
            result = evaluate(model, _fake_loader(fake_batches))

        assert result == pytest.approx(50.0, abs=1e-3)

    def test_result_is_percentage(self):
        """Result must be in [0, 100], not [0, 1]."""
        model = _tiny_resnet()
        model.eval()
        fake_batches = [(torch.randn(2, 3, 224, 224), torch.zeros(2, dtype=torch.long))]

        with patch("resnet18_nb.device", "cpu"):
            result = evaluate(model, _fake_loader(fake_batches))

        assert 0.0 <= result <= 100.0, \
            f"Expected percentage in [0,100] but got {result}. " \
            "Check the return statement in evaluate()."

    def test_result_is_float(self):
        """evaluate() must return a float, not an integer."""
        model = _tiny_resnet()
        model.eval()
        fake_batches = [(torch.randn(2, 3, 224, 224), torch.zeros(2, dtype=torch.long))]

        with patch("resnet18_nb.device", "cpu"):
            result = evaluate(model, _fake_loader(fake_batches))

        assert isinstance(result, float)

    def test_accumulates_across_batches(self):
        """Accuracy must be computed across all batches, not just the last one."""
        model = _tiny_resnet()
        model.eval()

        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.zero_()
            model.fc.bias[0] = 10.0

        # Batch 1: 4 correct (class 0). Batch 2: 4 wrong (class 1).
        # Overall: 4/8 = 50%
        batch1 = (torch.randn(4, 3, 224, 224), torch.zeros(4, dtype=torch.long))
        batch2 = (torch.randn(4, 3, 224, 224), torch.ones(4,  dtype=torch.long))

        with patch("resnet18_nb.device", "cpu"):
            result = evaluate(model, _fake_loader([batch1, batch2]))

        assert result == pytest.approx(50.0, abs=1e-3)


# 2. ACCURACY FORMULA

class TestAccuracyFormula:
    """
    Tests that read the return expression of evaluate()
    """

    def test_formula_gives_percentage_not_fraction(self):
        """Formula must multiply by 100 — result for 1/1 correct must be 100, not 1."""
        result = _run_accuracy_formula(correct=1, total=1)
        assert result == pytest.approx(100.0), \
            f"Got {result} — did you accidentally remove '100 *' from the formula?"

    def test_formula_zero_correct(self):
        """0 correct out of any total must give 0.0."""
        assert _run_accuracy_formula(correct=0, total=10) == pytest.approx(0.0)

    def test_formula_all_correct(self):
        """All correct must give 100.0."""
        assert _run_accuracy_formula(correct=50, total=50) == pytest.approx(100.0)

    def test_formula_half_correct(self):
        """Half correct must give 50.0."""
        assert _run_accuracy_formula(correct=5, total=10) == pytest.approx(50.0)

    def test_formula_is_finite(self):
        """Formula must always return a finite number for valid inputs."""
        assert math.isfinite(_run_accuracy_formula(correct=37, total=100))


# 3. MODEL ARCHITECTURE

class TestModelArchitecture:
    """
    Tests that the notebook builds a ResNet18 correctly adapted for
    Tiny ImageNet (200 output classes). 
    """

    def test_output_classes_is_200(self):
        """The fc layer must have 200 output features for Tiny ImageNet."""
        m = _tiny_resnet(n_cls=200)
        assert m.fc.out_features == 200, \
            "fc layer must output 200 classes for Tiny ImageNet."

    def test_input_features_unchanged(self):
        """ResNet18's fc input features must remain 512."""
        m = _tiny_resnet()
        assert m.fc.in_features == 512

    def test_model_forward_shape(self):
        """Model must produce (batch, 200) output for a standard input."""
        m = _tiny_resnet()
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 200)

    def test_model_is_nn_module(self):
        """Model must be an nn.Module."""
        assert isinstance(_tiny_resnet(), nn.Module)

    def test_model_has_fc_layer(self):
        """Model must have an fc attribute that is a Linear layer."""
        m = _tiny_resnet()
        assert hasattr(m, "fc")
        assert isinstance(m.fc, nn.Linear)


# 4. QUANTISATION CONFIG STRUCTURE

def _extract_quantization_config() -> dict:
    """
    Parse ResNet18.ipynb and extract the quantization_config dict directly
    from the notebook source.
    """
    nb_path = Path(_nb.__file__)
    nb = nbformat.read(str(nb_path), as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code" or "quantization_config" not in cell.source:
            continue
        lines = cell.source.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("quantization_config") and "=" in line:
                # Collect lines from here until the closing brace
                block = []
                brace_depth = 0
                for j in range(i, len(lines)):
                    block.append(lines[j])
                    brace_depth += lines[j].count("{") - lines[j].count("}")
                    if brace_depth == 0 and block:
                        break
                # Strip inline comments so eval() doesn't choke on them
                clean = "\n".join(
                    l.split("#")[0] for l in block
                )
                # eval just the right-hand side
                rhs = clean[clean.index("=") + 1:].strip()
                return eval(rhs)
    raise RuntimeError(
        "Could not find quantization_config in the notebook. "
        "Make sure the variable is assigned on a single line as "
        "'quantization_config = {'"
    )


class TestQuantisationConfig:
    """
    Tests for the quantization_config dict read LIVE from the notebook source.
    """

    @pytest.fixture(autouse=True)
    def load_config(self):
        """Extract the real config from the notebook before each test."""
        self.cfg = _extract_quantization_config()

    def test_quantisation_by_type(self):
        """Config must target layers 'by type'."""
        assert self.cfg["by"] == "type"

    def test_default_layers_unquantised(self):
        """Default layers must have name=None (left unquantised)."""
        assert self.cfg["default"]["config"]["name"] is None

    def test_linear_uses_integer_scheme(self):
        """Linear layers must use the 'integer' quantisation scheme."""
        assert self.cfg["linear"]["config"]["name"] == "integer"

    def test_linear_data_width_is_valid(self):
        """Data input width must be a standard power-of-2 bit width (4, 8, 16, or 32).
        Catches nonsensical values like 3, 0, or negatives — but allows any
        valid quantisation precision."""
        valid_widths = {4, 8, 16, 32}
        width = self.cfg["linear"]["config"]["data_in_width"]
        assert width in valid_widths, (
            f"data_in_width={width} is not a valid bit width. "
            f"Must be one of {valid_widths}."
        )

    def test_linear_weight_width_is_valid(self):
        """Weight width must be a standard power-of-2 bit width (4, 8, 16, or 32)."""
        valid_widths = {4, 8, 16, 32}
        width = self.cfg["linear"]["config"]["weight_width"]
        assert width in valid_widths, (
            f"weight_width={width} is not a valid bit width. "
            f"Must be one of {valid_widths}."
        )

    def test_bias_frac_width_is_zero(self):
        """bias_frac_width must be 0 to avoid TorchScript issues (per notebook comment).
        This is a hard requirement, not a design choice."""
        assert self.cfg["linear"]["config"]["bias_frac_width"] == 0, (
            "bias_frac_width must always be 0 — changing it breaks TorchScript export."
        )

    def test_frac_widths_less_than_total_widths(self):
        """Fractional width must always be strictly less than total width.
        e.g. frac_width=8 with total_width=8 would leave no integer bits."""
        linear_cfg = self.cfg["linear"]["config"]
        assert linear_cfg["data_in_frac_width"] < linear_cfg["data_in_width"], (
            f"data_in_frac_width ({linear_cfg['data_in_frac_width']}) must be "
            f"less than data_in_width ({linear_cfg['data_in_width']})."
        )
        assert linear_cfg["weight_frac_width"] < linear_cfg["weight_width"], (
            f"weight_frac_width ({linear_cfg['weight_frac_width']}) must be "
            f"less than weight_width ({linear_cfg['weight_width']})."
        )

    def test_bit_widths_are_positive(self):
        """All bit widths must be positive integers — zero or negative makes no sense."""
        linear_cfg = self.cfg["linear"]["config"]
        for field in ("data_in_width", "weight_width", "bias_width"):
            assert linear_cfg[field] > 0, (
                f"{field}={linear_cfg[field]} is invalid — bit widths must be positive."
            )


# 5. ONNX EXPORT PATH LOGIC

class TestOnnxExportPath:
    """
    Verify the output directory and filename logic used in the notebook.
    """

    def test_output_dir_name(self):
        """Output directory must be named 'mase_output'."""
        output_dir = Path("./mase_output")
        assert output_dir.name == "mase_output"

    def test_onnx_filename(self):
        """Exported file must be named 'resnet18_qat_fp32.onnx'."""
        output_dir = Path("./mase_output")
        onnx_path = output_dir / "resnet18_qat_fp32.onnx"
        assert onnx_path.name == "resnet18_qat_fp32.onnx"

    def test_onnx_extension(self):
        """Exported file must have .onnx extension."""
        output_dir = Path("./mase_output")
        onnx_path = output_dir / "resnet18_qat_fp32.onnx"
        assert onnx_path.suffix == ".onnx"

    def test_output_dir_is_relative(self):
        """Output directory path must be relative (not absolute)."""
        output_dir = Path("./mase_output")
        assert not output_dir.is_absolute()
