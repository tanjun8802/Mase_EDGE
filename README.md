# Mase_EDGE

ADLS Group 5 Project 2026, Imperial College London

## Layout

| Path | Purpose |
|------|---------|
| `edge_study/` | EDGE Optuna study: pruning, XNNPACK PT2E export, ADB benchmark hooks, objective |
| `tiny_imagenet_lib/` | Shared Tiny ImageNet helpers (`evaluate`, CHOP-style quant configs) used by the training notebooks |
| `EDGE_device/` | Device specs and ADB helpers |
| `android_app/` | ExecuTorch MV3 Android demo |
| `python notebooks/` | ResNet18 and MobileNetV3 training notebooks (import from `tiny_imagenet_lib`) |
| `checkpoints/` | Base `.pt` weights (`resnet18_qat_fp32.pt`, `mobilenet_v3_large_qat_fp32.pt`, `mobilenet_v3_best_ptq.pt`); `edge_study.settings` points here |
| `EDGE_optuna_study.ipynb` | Optuna study driver (imports `edge_study`) |

## Android app and `.pte` on device

1. Open and run the app from `android_app/executorch-examples/mv3/android/MV3Demo` in Android Studio (Gradle). Skip if the app is already installed.

2. Close the app after it launches.

3. Push the `.pte` to device temp storage:

   ```bash
   adb push <path-to-pte>/<name>.pte /data/local/tmp/
   ```

4. Copy into app-private storage:

   ```bash
   adb shell "run-as com.image_classification_app cp /data/local/tmp/<name>.pte files/model.pte"
   ```

5. Relaunch the app, pick an image, run inference.

## Tests

From the repo root, with dependencies installed (`pip install -r unit_test/requirements.txt` plus ExecuTorch if you want real export rather than the test stubs):

```bash
python -m pytest unit_test/ -v
```

Use `-m pytest` so the interpreter loads the installed `pytest` package; `python pytest …` looks for a file named `pytest`.

`unit_test/conftest.py` stubs ExecuTorch/torchao when needed so imports resolve without a full ET install.
