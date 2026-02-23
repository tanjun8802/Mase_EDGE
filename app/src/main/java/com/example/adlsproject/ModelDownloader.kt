package com.example.adlsproject

import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream

object ModelDownloader {

    private val client = OkHttpClient.Builder()
        .retryOnConnectionFailure(true)
        .build()

    /**
     * Resumable download with HTTP Range.
     * - If dest exists, requests bytes=<existing>- and appends.
     * - If server ignores Range and returns 200, it overwrites.
     */
    fun downloadResumable(
        url: String,
        dest: File,
        onProgress: ((downloaded: Long, total: Long) -> Unit)? = null
    ) {
        dest.parentFile?.mkdirs()

        val existing = if (dest.exists()) dest.length() else 0L

        val reqBuilder = Request.Builder().url(url)
        if (existing > 0L) reqBuilder.header("Range", "bytes=$existing-")

        client.newCall(reqBuilder.build()).execute().use { resp ->
            if (!resp.isSuccessful) throw RuntimeException("HTTP ${resp.code} for $url")

            val body = resp.body ?: throw RuntimeException("Empty body for $url")

            // If 206, contentLength is remaining bytes; if 200, it's full size.
            val incomingLen = body.contentLength()
            val total = when {
                incomingLen <= 0L -> -1L
                resp.code == 206 -> existing + incomingLen
                else -> incomingLen
            }

            // If server ignored Range (200), start from scratch
            val append = (resp.code == 206 && existing > 0L)

            body.byteStream().use { input ->
                val out = if (append) {
                    FileOutputStream(dest, true)
                } else {
                    FileOutputStream(dest, false)
                }

                out.use { output ->
                    val buf = ByteArray(1024 * 1024) // 1MB
                    var read: Int
                    var downloaded = if (append) existing else 0L

                    while (input.read(buf).also { read = it } != -1) {
                        output.write(buf, 0, read)
                        downloaded += read
                        onProgress?.invoke(downloaded, total)
                    }
                    output.flush()
                }
            }
        }
    }
}
