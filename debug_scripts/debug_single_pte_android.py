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
from typing import Any, Optional

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EDGE_device.device_specifications import get_adb_path  # noqa: E402

DEFAULT_APP = "com.image_classification_app"

DEFAULT_POLL_TIMEOUT = int(os.environ.get("EDGE_METRICS_POLL_TIMEOUT_SEC", "3600"))

def _metrics_payload_ready(data: dict) -> bool:
    st = data.get("status")
    if st in ("done", "error"):
        return True
    return isinstance(data, dict) and "latency_p95_ms" in data and "top1_acc" in data


def _adb_cat_stdout_not_metrics_json(raw: bytes) -> bool:
    """True when adb/run-as/cat failed or file missing — message often lands on stdout with rc=0."""
    if not (raw or b"").strip():
        return True
    if raw.lstrip().startswith(b"cat:"):
        return True
    if b"No such file or directory" in raw:
        return True
    if b"Permission denied" in raw:
        return True
    return False


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


def clear_edge_metrics_json_on_device(adb: str, app_name: str) -> None:
    """Remove all metrics_trial_*.json so the host never reads a stale file from a prior study/trial."""
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


def start_benchmark(
    adb: str,
    app_name: str,
    split: str,
    trial_id: int,
    *,
    max_images: Optional[int] = None,
) -> None:
    action = f"{app_name}.action.BENCHMARK"
    cmd = [
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
    ]
    if max_images is not None and max_images > 0:
        cmd.extend(["--ei", "max_images", str(max_images)])
    subprocess.run(cmd, check=True, timeout=30)


def poll_metrics(adb: str, app_name: str, trial_id: int, timeout: int = 600) -> dict:
    remote = f"files/metrics_trial_{trial_id}.json"
    t0 = time.time()
    last_info_ts = 0.0
    time.sleep(2)
    while time.time() - t0 < timeout:
        try:
            proc = subprocess.run(
                [adb, "exec-out", "run-as", app_name, "cat", remote],
                capture_output=True,
                timeout=30,
            )
            raw_out = proc.stdout or b""
            stripped = raw_out.strip()
            if proc.returncode != 0:
                err_txt = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
                if err_txt:
                    print(f"  [adb] cat rc={proc.returncode}: {err_txt[:400]}")
                time.sleep(2)
                continue
            if not stripped:
                time.sleep(2)
                continue
            # Missing metrics file while benchmark runs: cat prints to stdout; adb may still return 0.
            if _adb_cat_stdout_not_metrics_json(raw_out):
                now = time.time()
                if now - last_info_ts >= 30:
                    last_info_ts = now
                    elapsed = int(now - t0)
                    print(
                        f"  [poll] metrics file not written yet ({elapsed}s / {timeout}s). "
                        "MaseOptimise only creates JSON after the full eval loop finishes "
                        f"(use --max-images for a quick run). remote={remote!r}"
                    )
                time.sleep(2)
                continue
            try:
                data = json.loads(stripped.decode("utf-8"))
            except json.JSONDecodeError as je:
                print(
                    f"  [poll] JSONDecodeError: {je!s}; rc={proc.returncode}; "
                    f"stdout_preview={repr(raw_out[:200])}"
                )
                time.sleep(2)
                continue
            if not _metrics_payload_ready(data):
                time.sleep(2)
                continue
            err = data.get("error")
            if err and str(err).lower() not in ("null", "none", ""):
                print(f"  [metrics] device error field: {err}")
            st_raw = data.get("status")
            status_str = st_raw if isinstance(st_raw, str) else ""
            out: dict[str, Any] = {
                "top1_acc": float(data.get("top1_acc", 0.0)),
                "latency_p95_ms": float(data.get("latency_p95_ms", 999.0)),
                "memory_peak_mb": float(data.get("memory_peak_mb", 0.0)),
                "num_samples": int(data.get("num_samples", 0)),
                "status": status_str,
                "valid_forward_count": int(data.get("valid_forward_count", 0)),
                "decode_failures": int(data.get("decode_failures", 0)),
                "error": None if err in (None, "null", "") else str(err),
            }
            if data.get("num_samples_total_split") is not None:
                out["num_samples_total_split"] = int(data["num_samples_total_split"])
            return out
        except subprocess.TimeoutExpired:
            print("  [poll] adb cat timed out")
            time.sleep(2)
    raise TimeoutError(f"No complete metrics at {remote} after {timeout}s")


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
            print(
                "WARN: no --dataset and mase/tiny-imagenet-200 not found; "
                "use --skip-dataset if data already on device"
            )

    print("Starting benchmark intent…")
    clear_edge_metrics_json_on_device(adb, args.app)
    start_benchmark(
        adb,
        args.app,
        args.split,
        args.trial_id,
        max_images=args.max_images,
    )
    print("Polling metrics…")
    metrics = poll_metrics(adb, args.app, args.trial_id, timeout=args.poll_timeout)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
