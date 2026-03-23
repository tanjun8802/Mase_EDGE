import json
import os
import re
import shutil
import subprocess
import sys

# This script gathers device specifications via ADB and saves them to a JSON file.

PHONE_SPECS_FILE = "device_specifications.json"

_adb_path: str | None = None


def get_adb_path() -> str:
    """
    Resolve the adb binary. Notebooks and GUI apps often inherit a minimal PATH, so
    `adb` may be missing even when Android Studio’s SDK has it. We try PATH, then
    ANDROID_HOME / ANDROID_SDK_ROOT, then the default macOS SDK layout.
    """
    global _adb_path
    if _adb_path is not None:
        return _adb_path

    override = os.environ.get("ADB_PATH")
    if override and os.path.isfile(override):
        _adb_path = override
        return override

    exe = "adb.exe" if sys.platform == "win32" else "adb"
    candidates: list[str] = []
    for w in (shutil.which("adb"), shutil.which("adb.exe")):
        if w and w not in candidates:
            candidates.append(w)
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(var)
        if root:
            p = os.path.join(root, "platform-tools", exe)
            if p not in candidates:
                candidates.append(p)
    if sys.platform == "darwin":
        p = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
        if p not in candidates:
            candidates.append(p)
    elif sys.platform == "win32":
        lad = os.environ.get("LOCALAPPDATA")
        if lad:
            p = os.path.join(lad, "Android", "Sdk", "platform-tools", "adb.exe")
            if p not in candidates:
                candidates.append(p)

    for p in candidates:
        if p and os.path.isfile(p):
            _adb_path = p
            return p

    raise FileNotFoundError(
        "adb not found. Install Android SDK Platform-Tools, then either: "
        "add its folder to PATH (e.g. …/platform-tools), or set ANDROID_HOME or "
        "ANDROID_SDK_ROOT to the SDK root, or set ADB_PATH to the full path to adb. "
        "Default on macOS: ~/Library/Android/sdk/platform-tools/adb"
    )


def get_adb_output(cmd: str) -> str:
    adb = get_adb_path()
    result = subprocess.run([adb] + cmd.split(), capture_output=True, text=True, check=False)
    out = result.stdout
    print(f"{cmd} -> {repr(out[:200])}...")  # Truncate for readability
    return out.strip()

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

    out = get_adb_output("shell dumpsys SurfaceFlinger | grep GLES")

    if "Adreno" in out:
        return "Adreno", "high"
    elif "Mali" in out:
        return "Mali", "mid"
    elif "PowerVR" in out:
        return "PowerVR", "mid"
    else:
        return "unknown", "unknown"

def get_npu_info():
    
    nnapi_prop = get_adb_output("shell getprop ro.hardware.nnapi")
    props = get_adb_output("shell getprop | grep -Ei 'nnapi|neural|hexagon|htp|npu'")
    devs = get_adb_output("shell ls /dev | grep -Ei 'npu|neuron|hexagon|htp'")

    text = "\n".join([nnapi_prop, props, devs]).lower()

    has_npu = False
    npu_type = "none"
    npu_level = "unknown"

    if text:
        has_npu = True
        if "hexagon" in text or "htp" in text or "qnn" in text:
            npu_type = "Qualcomm NPU / HTP"
            npu_level = "high"
        elif "ethosu" in text or "npu" in text:
            npu_type = "generic NPU"
            npu_level = "mid"
        else:
            npu_type = "unknown NPU"
            npu_level = "unknown"

    return has_npu, npu_type, npu_level


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


    has_npu, npu_type, npu_level = get_npu_info()

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
        "npu_type": npu_type,
        "npu_level": npu_level,
        "prefers_npu": bool(has_npu and npu_level == "high"),
        "prefers_gpu": bool(gpu and gpu_level == "high"),
        "prefers_cpu": not (has_npu or (gpu and gpu_level == "high")),
    }
    return hw


def load_phone_specs(path: str = PHONE_SPECS_FILE) -> dict:
    """Load device specs JSON from file; return dict or {}."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_phone_specs(specs: dict, path: str = PHONE_SPECS_FILE):
    """Save device specs JSON to file."""
    with open(path, "w") as f:
        json.dump(specs, f, indent=2)


def get_hardware_for_phone() -> dict:
    """
    Get hardware spec for currently connected device.
    
    Auto detects device, queries via ADB, saves to `device_specifications.json`,
    and always overwrites with fresh data.
    
    Returns:
        dict of hardware specs
    """
    # Always query fresh and overwrite
    hw = get_device_info()
    key = f"{hw['manufacturer']}:{hw['model']}:{hw['soc_model']}"
    
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    json_path = os.path.join(script_dir, PHONE_SPECS_FILE)
    
    specs = load_phone_specs(json_path)
    specs[key] = hw
    save_phone_specs(specs, json_path)
    
    print(f"Updated specs for {key} → {json_path}")
    return hw


if __name__ == "__main__":

    hw = get_device_info()
    key = f"{hw['manufacturer']}:{hw['model']}:{hw['soc_model']}"
    specs = {
        key: hw
    }
    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, PHONE_SPECS_FILE)

    with open(json_path, "w") as f:
        json.dump(specs, f, indent=2)

