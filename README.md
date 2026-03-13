# Mase_EDGE

ADLS Group 5 Project 2026, Imperial College London

End-to-end pipeline: **PyTorch training → MASE quantisation → ONNX/ExecuTorch export → Android deployment**.

---

## Repository structure

```
Mase_EDGE/
├── mase/                    # Git submodule – MASE/chop quantisation framework
├── python notebooks/        # Jupyter notebooks (ResNet18 QAT → ONNX export)
├── app/                     # Android application (ExecuTorch runtime)
├── pyproject.toml           # Python project & dependency specification
├── requirements.txt         # Flat pip requirements for the ML pipeline
└── README.md
```

---

## Dependency overview

| Layer | Technology | Managed by |
|-------|-----------|------------|
| ML training & quantisation | PyTorch 2.6, MASE/chop (submodule) | `pyproject.toml` / `requirements.txt` |
| Model export | ONNX, ExecuTorch Python tools | `pyproject.toml` / `requirements.txt` |
| Android runtime | ExecuTorch Android 0.6.0, Kotlin, OkHttp | `app/build.gradle.kts` |

---

## Setup

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/tanjun8802/Mase_EDGE.git
# or, if already cloned:
git submodule update --init --recursive
```

### 2. Create a Python virtual environment (Python ≥ 3.11)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install Python dependencies

**Option A – via `pyproject.toml` (recommended)**

```bash
pip install -e .            # installs mase-edge project deps
pip install -e ./mase       # installs the MASE/chop submodule in editable mode
```

**Option B – via `requirements.txt`**

```bash
pip install -r requirements.txt
pip install -e ./mase
```

> **GPU / CUDA users**: replace the CPU PyTorch wheels with CUDA-enabled ones:
> ```bash
> pip install torch==2.6.0 torchvision torchaudio \
>     --index-url https://download.pytorch.org/whl/cu124
> ```

### 4. Run the Python notebooks

```bash
cd "python notebooks"
jupyter notebook
```

### 5. Build & run the Android app

Open the project in **Android Studio** (Hedgehog or newer) and run on a device or emulator with API level ≥ 26.

The Gradle dependencies (ExecuTorch Android 0.6.0, Kotlin coroutines, OkHttp, etc.) are declared in `app/build.gradle.kts` and are resolved automatically by Gradle.

---

## Instructions

 1.
