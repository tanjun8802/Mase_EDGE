package com.example.adlsproject

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.adlsproject.databinding.ActivityMainBinding
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity(), LlmCallback {

    private lateinit var binding: ActivityMainBinding
    private var llm: LlmModule? = null
    private val sb = StringBuilder()

    private val modelUrl =
        "https://huggingface.co/ethanc8/stories110M-executorch-v0.2/resolve/main/xnnpack_llama2.pte"

    private val tokUrl =
        "https://huggingface.co/ethanc8/stories110M-executorch-v0.2/resolve/main/tokenizer.bin"

    // Keep these names on disk
    private val modelFile by lazy { File(filesDir, "model.pte") }
    private val tokFile by lazy { File(filesDir, "tokeniser.bin") }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.sendButton.isEnabled = false
        binding.responseText.text = "Preparing model..."

        Thread {
            try {
                ensureFiles()
                loadModel()
                runOnUiThread {
                    binding.responseText.text = "Ready."
                    binding.sendButton.isEnabled = true
                }
            } catch (e: Exception) {
                Log.e("LLM", "Setup failed", e)
                runOnUiThread {
                    binding.responseText.text = "Setup error: ${e.message}"
                }
            }
        }.start()

        binding.sendButton.setOnClickListener {
            val prompt = binding.promptInput.text.toString()
            if (prompt.isBlank()) return@setOnClickListener
            generate(prompt)
        }
    }

    private fun ensureFiles() {
        if (!tokFile.exists() || tokFile.length() == 0L) {
            ModelDownloader.downloadResumable(tokUrl, tokFile) { done, total ->
                runOnUiThread {
                    binding.responseText.text = progressText("Downloading tokenizer", done, total)
                }
            }
        }

        if (!modelFile.exists() || modelFile.length() == 0L) {
            ModelDownloader.downloadResumable(modelUrl, modelFile) { done, total ->
                runOnUiThread {
                    binding.responseText.text = progressText("Downloading model", done, total)
                }
            }
        }
    }

    private fun loadModel() {
        llm = LlmModule(modelFile.absolutePath, tokFile.absolutePath, 0.2f)
        val code = llm!!.load()
        if (code != 0) error("LlmModule.load() failed with code $code")
        Log.i("LLM", "LlmModule loaded")
    }

    private fun generate(prompt: String) {
        val module = llm ?: run {
            binding.responseText.text = "Model not loaded"
            return
        }

        sb.clear()
        runOnUiThread { binding.responseText.text = "Generating..." }

        val formatted = "Once upon a time, $prompt\n"

        Thread {
            try {
                module.generate(formatted, 128, this)
            } catch (e: Exception) {
                Log.e("LLM", "Generate failed", e)
                runOnUiThread { binding.responseText.text = "Error: ${e.message}" }
            }
        }.start()
    }

    override fun onResult(result: String) {
        sb.append(result)
        runOnUiThread { binding.responseText.text = sb.toString() }
    }

    override fun onStats(p0: Float) {
        Log.i("LLM", "tok/s=$p0")
    }

    private fun progressText(prefix: String, done: Long, total: Long): String {
        val doneMb = done / (1024.0 * 1024.0)
        return if (total > 0) {
            val pct = (100.0 * done / total).roundToInt()
            "$prefix: $pct% (${String.format("%.1f", doneMb)} MB)"
        } else {
            "$prefix: ${String.format("%.1f", doneMb)} MB"
        }
    }
}
