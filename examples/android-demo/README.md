# PaddleOCR-rs Android Example

## Overview

This example demonstrates how to use PaddleOCR-rs in an Android application via JNI (Java Native Interface).

## Prerequisites

1. Android Studio with NDK installed
2. Rust toolchain with Android target:
   ```bash
   rustup target add aarch64-linux-android
   ```
3. Android NDK (r26b or later recommended)

**Note:** Only `aarch64-linux-android` (ARM64) is supported by ort-sys prebuilt binaries. Other Android targets (armv7, x86_64) are NOT supported.

## Building the Native Library

### Using the build script (Linux/macOS):

```bash
# Build for ARM64 (only supported target)
./build-android.sh aarch64-linux-android --release
```

### Using the PowerShell script (Windows):

```powershell
# Build for ARM64 (only supported target)
.\build-android.ps1 -Target aarch64-linux-android -Release
```

### Using Cargo directly:

```bash
cargo build --release --target aarch64-linux-android --features ffi
```

## Project Structure

```
android-demo/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/paddleocr/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   └── PaddleOcrBridge.kt
│   │   │   ├── jniLibs/
│   │   │   │   └── arm64-v8a/
│   │   │   │       └── libpaddleocr_rs_onnx.so
│   │   │   ├── assets/
│   │   │   │   ├── ch_PP-OCRv4_det_infer.onnx
│   │   │   │   ├── ch_PP-OCRv4_rec_infer.onnx
│   │   │   │   └── ppocr_keys_v1.txt
│   │   │   └── res/
│   │   └── test/
│   └── build.gradle.kts
├── build.gradle.kts
└── settings.gradle.kts
```

## JNI Bridge (Kotlin)

```kotlin
package com.example.paddleocr

import android.content.Context
import java.io.File
import java.io.FileOutputStream

class PaddleOcrBridge {
    
    companion object {
        init {
            System.loadLibrary("paddleocr_rs_onnx")
        }
        
        // Copy assets to internal storage for native access
        fun copyAssetsToInternal(context: Context) {
            val models = listOf(
                "ch_PP-OCRv4_det_infer.onnx",
                "ch_PP-OCRv4_rec_infer.onnx",
                "ppocr_keys_v1.txt"
            )
            
            for (model in models) {
                val file = File(context.filesDir, model)
                if (!file.exists()) {
                    context.assets.open(model).use { input ->
                        FileOutputStream(file).use { output ->
                            input.copyTo(output)
                        }
                    }
                }
            }
        }
    }
    
    // Native methods
    external fun createEngine(
        detModelPath: String,
        recModelPath: String,
        keysPath: String,
        device: Int = 0 // 0 = CPU, 4 = NNAPI
    ): Long
    
    external fun destroyEngine(handle: Long)
    
    external fun recognize(
        handle: Long,
        imagePath: String,
        order: Int = 0 // 0 = Horizontal, 1 = Vertical, 2 = Score
    ): Array<OcrResult>
    
    data class OcrResult(
        val text: String,
        val confidence: Float,
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float
    )
}
```

## Usage Example (Kotlin)

```kotlin
package com.example.paddleocr

import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class MainActivity : AppCompatActivity() {
    
    private var engineHandle: Long = 0
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Copy models to internal storage
        PaddleOcrBridge.copyAssetsToInternal(this)
        
        // Create OCR engine
        val detModel = File(filesDir, "ch_PP-OCRv4_det_infer.onnx")
        val recModel = File(filesDir, "ch_PP-OCRv4_rec_infer.onnx")
        val keys = File(filesDir, "ppocr_keys_v1.txt")
        
        engineHandle = PaddleOcrBridge().createEngine(
            detModel.absolutePath,
            recModel.absolutePath,
            keys.absolutePath,
            0 // CPU
        )
        
        // Perform OCR on an image
        val imageFile = File(filesDir, "test.png")
        val results = PaddleOcrBridge().recognize(engineHandle, imageFile.absolutePath)
        
        for (result in results) {
            println("Text: ${result.text}")
            println("Confidence: ${result.confidence}")
            println("Position: (${result.x}, ${result.y}) ${result.width}x${result.height}")
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        if (engineHandle != 0L) {
            PaddleOcrBridge().destroyEngine(engineHandle)
        }
    }
}
```

## Model Files

Place the following model files in `app/src/main/assets/`:

1. `ch_PP-OCRv4_det_infer.onnx` - Text detection model
2. `ch_PP-OCRv4_rec_infer.onnx` - Text recognition model
3. `ppocr_keys_v1.txt` - Character dictionary

Download from: https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_infer.tar

## NNAPI Acceleration

To use NNAPI acceleration (for supported devices):

```kotlin
engineHandle = PaddleOcrBridge().createEngine(
    detModel.absolutePath,
    recModel.absolutePath,
    keys.absolutePath,
    4 // NNAPI
)
```

Note: NNAPI support varies by device and Android version. The engine will fallback to CPU if NNAPI is not available.

## Performance Tips

1. Use ARM64 (aarch64-linux-android) for best performance
2. NNAPI can provide significant speedup on supported devices
3. Consider using quantized models for mobile deployment
4. Cache the engine instance - creating it is expensive
5. Use background threads for OCR operations

## Troubleshooting

### "Library not found" error
- Ensure the `.so` file is in the correct `jniLibs/arm64-v8a/` directory
- Check that your device is ARM64 (most modern Android devices are)

### "Model load failed" error
- Verify model files are in the assets directory
- Check file permissions in internal storage

### NNAPI fallback to CPU
- Check device Android version (NNAPI requires Android 8.1+)
- Verify device GPU/NPU supports NNAPI operations
- Check logcat for NNAPI-related warnings