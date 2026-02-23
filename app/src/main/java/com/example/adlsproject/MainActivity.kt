package com.example.adlsproject

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.adlsproject.databinding.ActivityMainBinding
import org.pytorch.executorch.EValue
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

        binding.sendButton.setOnClickListener {
            val prompt = binding.promptInput.text.toString()
            runInference(prompt)
        }
    }

    private fun loadModel() {
        try {
            val modelPath = assetFilePath("model.pte")
            module = Module.load(modelPath)
        } catch (e: Exception) {
            Log.e("Executor", "Error loading model", e)
        }
    }

    private fun runInference(prompt: String) {
        if (module == null) {
            Log.e("Executor", "Model not loaded")
            binding.responseText.text = "Error: Model not loaded"
            return
        }

        // In a real app, you would use a tokenizer to convert text to tokens.
        val inputTokens = longArrayOf(1, 2, 3) // Example input tokens from prompt
        val inputTensor = Tensor.fromBlob(inputTokens, longArrayOf(1, inputTokens.size.toLong()))

        try {
            // Run inference. The method name might be different for your model.
            val outputTensor = module?.execute("forward", EValue.from(inputTensor))?.get(0)?.toTensor()

            // Process the output
            val outputData = outputTensor?.dataAsLongArray
            val response = outputData?.joinToString() ?: "No response"
            binding.responseText.text = response
            Log.i("Executor", "Output token IDs: $response")
        } catch (e: Exception) {
            Log.e("Executor", "Error during inference", e)
            binding.responseText.text = "Error: ${e.message}"
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
