package com.example.adlsproject

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.adlsproject.databinding.ActivityMainBinding
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var module: Module? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        loadModel()
        runInference()
    }

    private fun loadModel() {
        try {
            val modelPath = assetFilePath("model.pte")
            module = Module(modelPath)
        } catch (e: Exception) {
            Log.e("Executor", "Error loading model", e)
        }
    }

    private fun runInference() {
        if (module == null) {
            Log.e("Executor", "Model not loaded")
            return
        }

        // Create a sample input tensor. 
        // Replace with your model's actual input shape and data.
        val inputTensor = Tensor.fromBlob(floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f), longArrayOf(1, 4))

        try {
            // Run inference
            val outputTensor = module?.execute("forward", inputTensor)?.toTensor()

            // Process the output
            val outputData = outputTensor?.dataAsFloatArray
            Log.i("Executor", "Output: ${outputData?.joinToString()}")
        } catch (e: Exception) {
            Log.e("Executor", "Error during inference", e)
        }
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
            return file.absolutePath
        }
    }
}