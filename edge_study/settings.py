"""Mutable run configuration for the EDGE study (override from notebooks or tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Repo root: …/ADLSProject
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

IMAGE_CLASSIFICATION_APP_ID: str = "com.image_classification_app"

EDGE_EVAL_SPLIT: str = "val"
EDGE_METRICS_POLL_TIMEOUT_FLAG: int = 3600
EDGE_BENCHMARK_MAX_IMAGES_FLAG: Optional[int] = 100

CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
RESNET18_PT_PATH: Path = CHECKPOINTS_DIR / "resnet18_qat_fp32.pt"
MOBILENET_V3_PATH: Path = CHECKPOINTS_DIR / "mobilenet_v3_large_qat_fp32.pt"

EPOCHS: int = 1
LR: float = 1e-4
BATCH_SIZE: int = 64

# Filled by init_default_dataset_path()
EDGE_DEVICE_DATASET_PATH: Optional[str] = None

# CPU weight snapshot after first load_model_and_train()
BASE_STATE_DICT: Optional[dict] = None


def init_default_dataset_path() -> None:
    """Set EDGE_DEVICE_DATASET_PATH from mase/ or repo-root tiny-imagenet-200."""
    global EDGE_DEVICE_DATASET_PATH
    for cand in (
        PROJECT_ROOT / "mase" / "tiny-imagenet-200",
        PROJECT_ROOT / "tiny-imagenet-200",
    ):
        if cand.is_dir():
            EDGE_DEVICE_DATASET_PATH = str(cand)
            return
    EDGE_DEVICE_DATASET_PATH = None


init_default_dataset_path()
