# Mase_EDGE 

 ADLS Group 5 Project 2026, Imperial College London

 ## Instructions

1. Run the app from the `android_app/executorch-examples/mv3/android` directory. You will need Android studio to configure the gradle there. SKIP THIS STEP IF THE APP IS ALREADY INSTALLED,

2. Once the app launches, close it.

3. Copy the `.pte` file to the phone's temporary storage:

   ```bash
   adb push <path-to-pte-file>/<file-name>.pte /data/local/tmp/
   ```
4. Copy the model from temporary storage into the app's private storage:

```bash
adb shell "run-as org.pytorch.executorchexamples.mv3 cp /data/local/tmp/<file-name>.pte files/model.pte"
```
5. Relaunch the app on your phone, select an image, and run inference.