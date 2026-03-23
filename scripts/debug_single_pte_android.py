#!/usr/bin/env python3
"""
Run one .pte through the same MV3Demo benchmark path as EDGE_optuna_study.ipynb.

Prerequisites:
  - Device USB-connected, adb working (PATH or ANDROID_HOME / ADB_PATH — see EDGE_device.device_specifications).
  - Debuggable build of com.image_classification_app installed.
  - Optional: Tiny ImageNet (or ImageFolder layout) on host to push to files/dataset/.

View prediction vs label logs while the benchmark runs (second terminal):
  adb logcat -c && adb logcat -s MaseOptimise:D MaseOptimise:I

Or run this script with --logcat to spawn logcat in the foreground (blocks until Ctrl+C).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EDGE_device.device_specifications import get_adb_path  # noqa: E402

DEFAULT_APP = "com.image_classification_app"


def move_pte_to_app(adb: str, pte_path: str, app_name: str) -> None:
    m = re.search(r"([^/\\]+\.pte)$", pte_path)
    model_name = m.group(1) if m else "model.pte"
    subprocess.run([adb, "push", pte_path, "/data/local/tmp/"], check=True, timeout=120)
    subprocess.run(
        [adb, "shell", "run-as", app_name, "cp", f"/data/local/tmp/{model_name}", "files/model.pte"],
        check=True,
        timeout=60,
    )


def move_dataset_to_app(adb: str, dataset_host_path: str, app_name: str) -> None:
    dataset_name = os.path.basename(os.path.normpath(dataset_host_path))
    subprocess.run([adb, "push", dataset_host_path, "/data/local/tmp/"], check=True, timeout=600)
    src_dot = shlex.quote(f"/data/local/tmp/{dataset_name}/.")
    inner_script = (
        f"DEST=$(ls -d /data/user/*/{app_name}/files 2>/dev/null | head -n1); "
        f'[ -n "$DEST" ] || DEST=/data/data/{app_name}/files; '
        f'test -d "$DEST" && rm -rf "$DEST/dataset" && mkdir -p "$DEST/dataset" && '
        f"cp -R {src_dot} \"$DEST/dataset/\""
    )
    remote_line = f"run-as {shlex.quote(app_name)} sh -c {shlex.quote(inner_script)}"
    r = subprocess.run([adb, "shell", remote_line], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, remote_line, r.stdout, r.stderr)


def start_benchmark(adb: str, app_name: str, split: str, trial_id: int) -> None:
    action = f"{app_name}.action.BENCHMARK"
    subprocess.run(
        [
            adb,
            "shell",
            "am",
            "start",
            "-n",
            f"{app_name}/.MainActivity",
            "-a",
            action,
            "-e",
            "split",
            split,
            "-e",
            "trial_id",
            str(trial_id),
        ],
        check=True,
        timeout=30,
    )


def poll_metrics(adb: str, app_name: str, trial_id: int, timeout: int = 600) -> dict:
    remote = f"files/metrics_trial_{trial_id}.json"
    t0 = time.time()
    time.sleep(2)
    while time.time() - t0 < timeout:
        proc = subprocess.run(
            [adb, "exec-out", "run-as", app_name, "cat", remote],
            capture_output=True,
            timeout=15,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return json.loads(proc.stdout.decode("utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(2)
    raise TimeoutError(f"No metrics at {remote} after {timeout}s")


def main() -> None:
    p = argparse.ArgumentParser(description="Push one .pte and run MV3Demo BENCHMARK once.")
    p.add_argument("pte", type=Path, help="Path to .pte on host")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Host folder to push as files/dataset/ (e.g. mase/tiny-imagenet-200)",
    )
    p.add_argument("--app", default=DEFAULT_APP)
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--trial-id", type=int, default=999)
    p.add_argument("--skip-dataset", action="store_true", help="Do not push dataset (reuse phone copy)")
    p.add_argument(
        "--logcat",
        action="store_true",
        help="Print adb logcat MaseOptimise lines in this process (foreground)",
    )
    args = p.parse_args()

    pte = args.pte.resolve()
    if not pte.is_file():
        sys.exit(f"Missing .pte: {pte}")

    adb = get_adb_path()
    print("adb:", adb)
    print("pte:", pte)

    if args.logcat:

        def _run_logcat() -> None:
            subprocess.run([adb, "logcat", "-s", "MaseOptimise:D", "MaseOptimise:I"], check=False)

        threading.Thread(target=_run_logcat, daemon=True).start()
        time.sleep(0.5)

    move_pte_to_app(adb, str(pte), args.app)
    print("Pushed model → files/model.pte")

    if not args.skip_dataset:
        ds = args.dataset
        if ds is None:
            guess = REPO_ROOT / "mase" / "tiny-imagenet-200"
            if guess.is_dir():
                ds = guess
        if ds is not None and ds.is_dir():
            print("Pushing dataset (slow)…", ds)
            move_dataset_to_app(adb, str(ds), args.app)
            print("Dataset → files/dataset/")
        elif not args.skip_dataset:
            print("WARN: no --dataset and mase/tiny-imagenet-200 not found; use --skip-dataset if data already on device")

    print("Starting benchmark intent…")
    start_benchmark(adb, args.app, args.split, args.trial_id)
    print("Polling metrics…")
    metrics = poll_metrics(adb, args.app, args.trial_id)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
