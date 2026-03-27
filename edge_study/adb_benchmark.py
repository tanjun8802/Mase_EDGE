"""ADB: push .pte / dataset, start benchmark intent, poll metrics JSON."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from typing import Any, Optional

from EDGE_device.device_specifications import get_adb_path

from . import metrics_json
from . import settings


def move_pte_to_app(pte_path: str, app_name: str) -> None:
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


def edge_poll_metrics(
    trial_id: int,
    timeout: Optional[int] = None,
    app_package: Optional[str] = None,
) -> dict[str, Any]:
    app_package = app_package or settings.IMAGE_CLASSIFICATION_APP_ID
    timeout = settings.EDGE_METRICS_POLL_TIMEOUT_FLAG if timeout is None else timeout
    remote_rel = f"files/metrics_trial_{trial_id}.json"
    adb = get_adb_path()
    start = time.time()
    last_info_ts = 0.0
    time.sleep(2)
    while time.time() - start < timeout:
        proc: Optional[subprocess.CompletedProcess[bytes]] = None
        raw_out = b""
        try:
            proc = subprocess.run(
                [adb, "exec-out", "run-as", app_package, "cat", remote_rel],
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
            if metrics_json._adb_cat_stdout_not_metrics_json(raw_out):
                now = time.time()
                if now - last_info_ts >= 30:
                    last_info_ts = now
                    elapsed = int(now - start)
                    print(
                        f"  [poll] metrics file not written yet ({elapsed}s / {timeout}s). "
                        f"remote={remote_rel!r}"
                    )
                time.sleep(2)
                continue
            try:
                text = stripped.decode("utf-8")
                data = json.loads(text)
            except json.JSONDecodeError as je:
                print(
                    f"  [poll] JSONDecodeError: {je!s}; rc={proc.returncode}; "
                    f"stdout_preview={repr(raw_out[:200])}"
                )
                time.sleep(2)
                continue

            if not metrics_json._metrics_payload_ready(data):
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
            }
            if data.get("num_samples_total_split") is not None:
                out["num_samples_total_split"] = int(data["num_samples_total_split"])
            out["error"] = None if err in (None, "null", "") else str(err)
            out["trial_id_json"] = data.get("trial_id")
            out["split_json"] = data.get("split")
            return out
        except subprocess.TimeoutExpired:
            print("  [poll] adb cat timed out")
            time.sleep(2)
        except Exception as ex:
            print(f"  [poll] {ex!r}")
            time.sleep(2)

    print(
        f"  [poll] timeout after {timeout}s — no readable {remote_rel}. "
        f"adb shell run-as {app_package} ls files/"
    )
    return {
        "top1_acc": 0.0,
        "latency_p95_ms": 999.0,
        "memory_peak_mb": 9999.0,
        "num_samples": 0,
        "error": "poll_timeout",
        "status": "error",
        "valid_forward_count": 0,
        "decode_failures": 0,
    }


def edge_benchmark_trial(
    trial_id: Optional[int],
    pte_path: Optional[str],
    dataset_path: Optional[str] = None,
    split: Optional[str] = None,
    app_name: Optional[str] = None,
    *,
    poll_timeout_sec: Optional[int] = None,
    max_images: Optional[int] = None,
    eval_only: bool = False,
) -> dict[str, Any]:
    app_name = app_name or settings.IMAGE_CLASSIFICATION_APP_ID
    ds = dataset_path if dataset_path is not None else settings.EDGE_DEVICE_DATASET_PATH
    ev_split = (split or settings.EDGE_EVAL_SPLIT).lower()
    benchmark_action = f"{app_name}.action.BENCHMARK"
    cap = settings.EDGE_BENCHMARK_MAX_IMAGES_FLAG if max_images is None else max_images
    tid = -1 if trial_id is None else int(trial_id)
    try:
        if not eval_only:
            if not pte_path:
                raise ValueError("pte_path is required when eval_only=False")
            move_pte_to_app(pte_path=pte_path, app_name=app_name)
            if tid == 0 and ds:
                move_dataset_to_app(dataset_path=ds, app_name=app_name)

        adb = get_adb_path()
        am_cmd = [
            adb,
            "shell",
            "am",
            "start",
            "-n",
            f"{app_name}/.MainActivity",
            "-a",
            benchmark_action,
            "-e",
            "split",
            ev_split,
            "-e",
            "trial_id",
            str(tid),
        ]
        if cap is not None and cap > 0:
            am_cmd.extend(["--ei", "max_images", str(cap)])
        clear_edge_metrics_json_on_device(app_name)
        subprocess.run(am_cmd, check=True, timeout=30)

        return edge_poll_metrics(
            tid,
            app_package=app_name,
            timeout=poll_timeout_sec
            if poll_timeout_sec is not None
            else settings.EDGE_METRICS_POLL_TIMEOUT_FLAG,
        )
    except Exception as e:
        print(f"Trial {tid} failed: {e}")
        return {
            "top1_acc": 0.0,
            "latency_p95_ms": 999.0,
            "memory_peak_mb": 0.0,
            "num_samples": 0,
            "error": str(e),
            "status": "error",
            "valid_forward_count": 0,
            "decode_failures": 0,
        }
