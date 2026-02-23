package com.example.adlsproject

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.adlsproject.databinding.ActivityMainBinding
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity(), LlmCallback {

    private lateinit var binding: ActivityMainBinding
    private var llm: LlmModule? = null
    private val isGenerating = AtomicBoolean(false)
    private val sb = StringBuilder()
    private val modelUrl = "https://huggingface.co/soup2k3/stories110M-executorch-06/resolve/main/stories110M_et06.pte"
    private val tokUrl   = "https://huggingface.co/soup2k3/stories110M-executorch-06/resolve/main/tokenizer.bin"

    private val modelFile by lazy { File(filesDir, "model.pte") }
    private val tokFile   by lazy { File(filesDir, "tokenizer.bin") }

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
                runOnUiThread { binding.responseText.text = "Setup error: ${e.message}" }
            }
        }.start()

        binding.sendButton.setOnClickListener {
            val prompt = binding.promptInput.text.toString()
            if (prompt.isBlank()) return@setOnClickListener
            generate(prompt)
        }
    }

    private fun ensureFiles() {
        // Force re-download if file is suspiciously small (old v0.2 model was ~440MB, new is ~170MB)
        if (modelFile.exists() && modelFile.length() < 100_000_000L) {
            Log.w("LLM", "Stale model detected (${modelFile.length()} bytes), deleting")
            modelFile.delete()
        }

        if (!tokFile.exists() || tokFile.length() == 0L) {
            ModelDownloader.downloadResumable(tokUrl, tokFile) { done, total ->
                runOnUiThread { binding.responseText.text = progressText("Downloading tokenizer", done, total) }
            }
        }
        if (!modelFile.exists() || modelFile.length() == 0L) {
            ModelDownloader.downloadResumable(modelUrl, modelFile) { done, total ->
                runOnUiThread { binding.responseText.text = progressText("Downloading model", done, total) }
            }
        }
    }


    private fun loadModel() {
        // 1 = TEXT_ONLY model category, required by LlmModule in executorch-android:0.6.0
        llm = LlmModule(1, modelFile.absolutePath, tokFile.absolutePath, 0.2f)
        val code = llm!!.load()
        if (code != 0) error("LlmModule.load() returned $code")
        Log.i("LLM", "Model loaded")
    }

    private fun generate(userPrompt: String) {
        val module = llm ?: run { binding.responseText.text = "Model not loaded"; return }

        if (!isGenerating.compareAndSet(false, true)) {
            try { module.stop() } catch (_: Throwable) {}
            return
        }

        sb.clear()
        runOnUiThread {
            binding.sendButton.isEnabled = false
            binding.responseText.text = "Generating..."
        }

        val formatted = "Once upon a time, ${userPrompt.trim()}\n"

        Thread {
            try {
                // With the correctly exported v0.6 model, 256 tokens is safe
                val rc = module.generate(formatted, 256, this)
                if (rc != 0) runOnUiThread { binding.responseText.text = "generate() failed: $rc" }
            } catch (t: Throwable) {
                Log.e("LLM", "Generate failed", t)
                runOnUiThread { binding.responseText.text = "Error: ${t.message}" }
            } finally {
                isGenerating.set(false)
                runOnUiThread { binding.sendButton.isEnabled = true }
            }
        }.start()
    }

    override fun onResult(result: String) {
        sb.append(result)
        runOnUiThread { binding.responseText.text = sb.toString() }
    }

    override fun onStats(tps: Float) {
        Log.i("LLM", "tok/s = $tps")
    }

    private fun progressText(prefix: String, done: Long, total: Long): String {
        val mb = done / (1024.0 * 1024.0)
        return if (total > 0) "$prefix: ${(100.0 * done / total).roundToInt()}% (${"%.1f".format(mb)} MB)"
        else "$prefix: ${"%.1f".format(mb)} MB"
    }
}
