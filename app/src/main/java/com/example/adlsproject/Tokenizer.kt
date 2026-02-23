package com.example.adlsproject

import android.content.res.AssetManager
import org.json.JSONObject


class Tokenizer(private val assetManager: AssetManager) {

    private val encoder: Map<String, Int>
    private val decoder: Map<Int, String>
    private val merges: List<Pair<String, String>>
    private val byteEncoder: Map<Int, String>
    private val byteDecoder: Map<String, Int>
    val bosTokenId: Int
    val eosTokenId: Int

    init {
        val vocabJson = assetManager.open("vocab.json")
            .bufferedReader().use { it.readText() }
        val vocabObj = JSONObject(vocabJson)
        val mutableEncoder = mutableMapOf<String, Int>()
        for (key in vocabObj.keys()) {
            mutableEncoder[key] = vocabObj.getInt(key)
        }
        encoder = mutableEncoder
        decoder = encoder.entries.associate { (k, v) -> v to k }

        val mergeLines = assetManager.open("merges.txt")
            .bufferedReader().use { it.readLines() }
        merges = mergeLines
            .drop(1)
            .filter { it.isNotBlank() }
            .map {
                val parts = it.split(" ")
                Pair(parts[0], parts[1])
            }

        // Build byte encoder/decoder once and cache it
        byteEncoder = buildByteEncoder()
        byteDecoder = byteEncoder.entries.associate { (k, v) -> v to k }

        bosTokenId = encoder["<|im_start|>"] ?: encoder["<s>"] ?: 1
        eosTokenId = encoder["<|im_end|>"] ?: encoder["</s>"] ?: 2
    }

    fun encode(text: String, addBos: Boolean = true): LongArray {
        // Apply chat template for SmolLM2 instruction format
        val formatted = if (addBos) {
            "<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n"
        } else {
            text
        }

        val tokens = bpeEncode(formatted)
        val ids = tokens.mapNotNull { encoder[it] }.map { it.toLong() }.toMutableList()

        // Do NOT prepend bosTokenId separately — it's already in the chat template above
        // Only add raw BOS if not using chat template
        return ids.toLongArray()
    }

    fun decode(tokenIds: LongArray): String {
        // Decode BPE tokens back to bytes then to string
        val tokenStr = tokenIds.joinToString("") { id ->
            decoder[id.toInt()] ?: ""
        }
        // Convert BPE byte-level chars back to actual bytes
        return try {
            val bytes = tokenStr.map { c ->
                (byteDecoder[c.toString()] ?: c.code).toByte()
            }.toByteArray()
            String(bytes, Charsets.UTF_8)
        } catch (e: Exception) {
            // Fallback: simple replacement
            tokenStr.replace("Ġ", " ").replace("Ċ", "\n")
        }
    }

    private fun bpeEncode(text: String): List<String> {
        if (text.isEmpty()) return emptyList()

        val result = mutableListOf<String>()

        // GPT-2 style pre-tokenization: split on spaces keeping the space attached
        // Simple word splitting that handles the Ġ (space) prefix convention
        val words = splitIntoWords(text)

        for (word in words) {
            val encoded = bpeEncodeWord(word)
            result.addAll(encoded)
        }

        return result
    }

    private fun splitIntoWords(text: String): List<String> {
        // GPT-2 pre-tokenization: attach leading space to following word
        val words = mutableListOf<String>()
        val regex = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        for (match in regex.findAll(text)) {
            words.add(match.value)
        }
        return words
    }

    private fun bpeEncodeWord(word: String): List<String> {
        // Convert characters to byte-level representation
        var symbols = word.map { c ->
            // Leading space becomes Ġ in GPT-2 BPE
            if (c == ' ') "Ġ" else (byteEncoder[c.code] ?: c.toString())
        }.toMutableList()

        if (symbols.isEmpty()) return emptyList()

        val mergeRanks = merges.withIndex().associate { (i, pair) ->
            Pair(pair.first, pair.second) to i
        }

        while (symbols.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1

            for (i in 0 until symbols.size - 1) {
                val rank = mergeRanks[Pair(symbols[i], symbols[i + 1])] ?: Int.MAX_VALUE
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if (bestIdx == -1 || bestRank == Int.MAX_VALUE) break

            val merged = symbols[bestIdx] + symbols[bestIdx + 1]
            symbols[bestIdx] = merged
            symbols.removeAt(bestIdx + 1)
        }

        return symbols
    }

    private fun buildByteEncoder(): Map<Int, String> {
        val bs = mutableListOf<Int>()
        ('!'.code..'~'.code).forEach { bs.add(it) }
        ('¡'.code..'¬'.code).forEach { bs.add(it) }
        ('®'.code..'ÿ'.code).forEach { bs.add(it) }

        val cs = bs.toMutableList()
        var n = 0
        (0 until 256).forEach { b ->
            if (b !in bs) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        return bs.zip(cs).associate { (b, c) -> b to c.toChar().toString() }
    }
}
