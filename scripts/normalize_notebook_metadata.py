#!/usr/bin/env python3
"""Strip volatile Jupyter/Colab metadata from .ipynb files to reduce git merge noise.

Removes per-cell Colab `id` / `outputId` / `colab` blobs, clears execution counts and outputs,
and drops notebook-level `widgets`, `colab`, and `accelerator` metadata.

Usage:
  python scripts/normalize_notebook_metadata.py [path/to/notebook.ipynb ...]

Default: EDGE_optuna_study.ipynb in repo root.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "EDGE_optuna_study.ipynb"

STABLE_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.0",
    },
}


def normalize(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        cell["metadata"] = {}
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    nb["metadata"] = dict(STABLE_METADATA)
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Normalized {path} ({len(nb['cells'])} cells)")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [DEFAULT_NOTEBOOK]
    for p in paths:
        if not p.is_file():
            print(f"Skip (missing): {p}", file=sys.stderr)
            continue
        normalize(p.resolve())


if __name__ == "__main__":
    main()
