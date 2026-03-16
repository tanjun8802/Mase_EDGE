import os
import subprocess
import re
import json

# This script gathers device specifications via ADB and saves them to a JSON file.

def get_adb_output(cmd: str) -> str:
    result = subprocess.run(
        ["adb"] + cmd.split(),
        capture_output=True,
        text=True,
        check=False
    )
    print(cmd + " -> " + repr(result.stdout))
    return result.stdout.strip()  # removes \n, \r, spaces


def get_cpu_cores():
    # try /proc/cpuinfo first
    out = get_adb_output("shell cat /proc/cpuinfo")
    if out:
        n = out.count("processor")
        if n > 0:
            return n

    # fallback: nproc
    nproc_out = get_adb_output("shell nproc")
    if nproc_out.strip():
        try:
            return int(nproc_out.strip())
        except ValueError:
            pass

    # last fallback: heuristic
    return 4  # assume at least 4 cores if we truly can't detect


def get_ram_mb():
    out = get_adb_output("shell cat /proc/meminfo | grep MemTotal")
    # MemTotal: 7825476 kB
    match = re.search(r"MemTotal:\s+(\d+)", out)
    if match:
        return int(match.group(1)) // 1024  # MB
    return 0

def get_gpu_info():
    # e.g., via `dumpsys SurfaceFlinger` or simple GL string
    out = get_adb_output("shell dumpsys SurfaceFlinger | grep GLES")
    # or use a small native helper that prints GPU vendor/renderer
    if "Adreno" in out:
        return "Adreno", "high"
    elif "Mali" in out:
        return "Mali", "mid"
    elif "PowerVR" in out:
        return "PowerVR", "mid"
    else:
        return "unknown", "unknown"

def get_npu_hint():
    # Check if NNAPI is enabled and HW accelerators exist
    # or probe a small C helper / JNI
    nnapi = get_adb_output("shell getprop ro.hardware.nnapi")
    if "qnn" in nnapi or "hexagon" in nnapi:
        return True, "Snapdragon NPU"
    # ... other vendor patterns
    return False, "none"

import re

def get_device_info():
    build = {}
    for field in [
        "MANUFACTURER",
        "MODEL",
        "BRAND",
        "SOC_MANUFACTURER",
        "SOC_MODEL",
        "HARDWARE",
        "SUPPORTED_ABIS",
        "SDK_INT",
    ]:
        val = get_adb_output(f"shell getprop ro.product.{field.lower()}")
        if not val:
            val = get_adb_output(f"shell getprop ro.{field.lower()}")
        build[field] = val.strip() if val else ""

    # CPU cores from /proc/cpuinfo
    cpuinf = get_adb_output("shell cat /proc/cpuinfo")
    cpu_cores = cpuinf.count("processor") if cpuinf else 0

    # RAM
    mem = get_adb_output("shell cat /proc/meminfo | grep MemTotal")
    m = re.search(r"MemTotal:\s+(\d+)", mem)
    ram_mb = int(m.group(1)) // 1024 if m else 0

    # GPU
    sf = get_adb_output("shell dumpsys SurfaceFlinger | grep GLES")
    gpu = None
    gpu_level = "unknown"
    if "Adreno" in sf:
        gpu = "Adreno 830"
        gpu_level = "high"
    elif "Mali" in sf:
        gpu = "Mali"
        gpu_level = "mid"
    # etc.

    # NPU (you can extend this later)
    nnapi = get_adb_output("shell getprop ro.hardware.nnapi")
    has_npu = bool(nnapi.strip())

    # Build your Optuna‑oriented JSON
    hw = {
        "manufacturer": build.get("MANUFACTURER"),
        "model": build.get("MODEL"),
        "brand": build.get("BRAND"),
        "soc_manufacturer": build.get("SOC_MANUFACTURER") or "Qualcomm",
        "soc_model": build.get("SOC_MODEL") or "SM8750",
        "hardware": build.get("HARDWARE") or "qcom",
        "cpu_cores": cpu_cores,
        "cpu_freq_max_ghz": 3.2,  # assume Snapdragon‑class; you can refine later
        "ram_gb": ram_mb / 1024.0,
        "has_gpu": bool(gpu),
        "gpu": gpu,
        "gpu_compute_level": gpu_level,
        "has_npu": has_npu,
        "npu_type": "Qualcomm NPU" if has_npu else "none",
        "prefers_npu": has_npu,
        "prefers_gpu": bool(gpu and gpu_level == "high"),
        "prefers_cpu": not (has_npu or (gpu and gpu_level == "high")),
    }
    return hw



if __name__ == "__main__":

    hw = get_device_info()
    key = f"{hw['manufacturer']}:{hw['model']}:{hw['soc_model']}"
    specs = {
        key: hw
    }
    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "device_specifications.json")

    with open(json_path, "w") as f:
        json.dump(specs, f, indent=2)

