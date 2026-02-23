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
            val modelPath = assetFilePath("smollm2_135m_q4.pte")
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

        // Create a sample input tensor with token IDs for the LLM.
        // In a real app, you would use a tokenizer to convert text to tokens.
        val inputTokens = longArrayOf(1, 2, 3) // Example input tokens
        val inputTensor = Tensor.fromBlob(inputTokens, longArrayOf(1, inputTokens.size.toLong()))

        try {
            // Run inference. The method name might be different for your model.
            val outputTensor = module?.execute("forward", inputTensor)?.toTensor()

            // Process the output
            val outputData = outputTensor?.dataAsLongArray
            Log.i("Executor", "Output token IDs: ${outputData?.joinToString()}")
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