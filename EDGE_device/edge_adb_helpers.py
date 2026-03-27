"""ADB helpers for pushing .pte and datasets into MV3Demo private storage (used by EDGE_optuna notebooks)."""

from __future__ import annotations

import os
import re
import shlex
import subprocess

from EDGE_device.device_specifications import get_adb_path


def move_pte_to_app(pte_path: str, app_name: str) -> None:
    """Push .pte to /data/local/tmp/ then copy into app files/model.pte via run-as."""
    adb = get_adb_path()

    maybe_model_name = re.search(r"([^/\\]+\.pte)$", pte_path)
    model_name = maybe_model_name.group(1) if maybe_model_name else "model.pte"

    subprocess.run(
        [adb, "push", pte_path, "/data/local/tmp/"],
        check=True,
        timeout=30,
    )
    subprocess.run(
        [
            adb,
            "shell",
            "run-as",
            app_name,
            "cp",
            f"/data/local/tmp/{model_name}",
            "files/model.pte",
        ],
        check=True,
        timeout=30,
    )


def move_dataset_to_app(dataset_path: str, app_name: str) -> None:
    """Copy dataset into files/dataset/ so MV3Demo sees files/dataset/train|val|test/ (ImageFolder)."""
    adb = get_adb_path()
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    subprocess.run(
        [adb, "push", dataset_path, "/data/local/tmp/"],
        check=True,
        timeout=600,
    )
    src_dot = shlex.quote(f"/data/local/tmp/{dataset_name}/.")
    inner_script = (
        f"DEST=$(ls -d /data/user/*/{app_name}/files 2>/dev/null | head -n1); "
        f'[ -n "$DEST" ] || DEST=/data/data/{app_name}/files; '
        f'test -d "$DEST" && rm -rf "$DEST/dataset" && mkdir -p "$DEST/dataset" && '
        f'cp -R {src_dot} "$DEST/dataset/"'
    )
    remote_line = f"run-as {shlex.quote(app_name)} sh -c {shlex.quote(inner_script)}"
    r = subprocess.run(
        [adb, "shell", remote_line],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        raise subprocess.CalledProcessError(
            r.returncode, [adb, "shell", remote_line], r.stdout, r.stderr
        )


def clear_edge_metrics_json_on_device(app_name: str) -> None:
    """Remove metrics_trial_*.json under app files/ (avoids stale reads before a study or trial)."""
    adb = get_adb_path()
    inner_script = (
        f"DEST=$(ls -d /data/user/*/{app_name}/files 2>/dev/null | head -n1); "
        f'[ -n "$DEST" ] || DEST=/data/data/{app_name}/files; '
        f'rm -f "$DEST"/metrics_trial_*.json'
    )
    remote_line = f"run-as {shlex.quote(app_name)} sh -c {shlex.quote(inner_script)}"
    subprocess.run(
        [adb, "shell", remote_line],
        capture_output=True,
        text=True,
        timeout=30,
    )
