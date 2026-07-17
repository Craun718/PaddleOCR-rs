# Hardware Acceleration

[← Back to API Overview](api-overview.md)

PaddleOCR-rs supports hardware acceleration via ONNX Runtime execution providers (EPs).

## AccelerationDevice Enum

```rust
pub enum AccelerationDevice {
    Cpu,        // CPU-only inference (default, always available)
    DirectML,   // DirectML acceleration (Windows, DirectX 12 GPU)
    Cuda,       // CUDA acceleration (NVIDIA GPU)
    OpenVINO,   // Intel OpenVINO acceleration (Windows/Linux)
    Nnapi,      // Android NNAPI acceleration (Android)
    Coreml,     // Apple CoreML acceleration (macOS/iOS)
    Cann,       // Huawei CANN / Ascend NPU acceleration (Linux)
}
```

## Platform Requirements

| Device | Platform | Requirements |
|--------|----------|-------------|
| **Cpu** | All | No additional requirements (always available) |
| **DirectML** | Windows 10/11 | DirectX 12 compatible GPU |
| **CUDA** | Windows/Linux | NVIDIA GPU with CUDA toolkit installed |
| **OpenVINO** | Windows/Linux | Intel CPU/GPU with OpenVINO toolkit installed |
| **NNAPI** | Android | Android device with NNAPI support |
| **CoreML** | macOS/iOS | macOS 10.13+ or iOS 11+ device |
| **CANN** | Linux | Huawei Ascend NPU with CANN toolkit installed |

## EP Availability by Feature

| EP | Status | Feature Required |
|----|--------|-----------------|
| ✅ **CPU** | Always available | None |
| ✅ **DirectML** | Included in default Windows builds | No feature needed |
| ❌ **CUDA** | Requires feature flag | `cuda` feature + CUDA toolkit |
| ❌ **OpenVINO** | Requires feature flag | `openvino` feature + custom ORT build |
| ❌ **NNAPI** | Requires feature flag | `nnapi` feature + custom ORT build |
| ❌ **CoreML** | Requires feature flag | `coreml` feature + custom ORT build |
| ❌ **CANN** | Requires feature flag | `cann` feature + custom ORT build |

## Enabling EPs via Cargo Features

```toml
[dependencies]
paddleocr_rs_onnx = { version = "0.1", features = ["cuda"] }

# Multiple EPs (fallback order: first enabled EP first)
paddleocr_rs_onnx = { version = "0.1", features = ["cuda", "directml"] }
```

## Checking EP Availability

Use `is_available()` to test whether a specific device can be used, and `available_devices()` to list all usable devices in the current environment:

```rust
use paddleocr_rs_onnx::AccelerationDevice;

// Check a specific device
if AccelerationDevice::Cuda.is_available() {
    println!("CUDA is available");
}

// List all available devices
let available = AccelerationDevice::available_devices();
for device in &available {
    println!("[available] {}", device);
}
```

The check covers two layers:
1. **Compile-time** — whether the corresponding Cargo feature is enabled
2. **Runtime** — whether the ONNX Runtime execution provider can be initialized successfully

## Runtime Behavior

If a requested EP is not available at runtime (not compiled into the ONNX Runtime binary, missing libraries, or unsupported platform), a warning is logged and inference falls back to CPU automatically.

## Usage Example

```rust
use paddleocr_rs_onnx::{OcrEngine, AccelerationDevice};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

// The engine will try the specified device and fall back to CPU if unavailable
let engine = OcrEngine::new_with_device(
    &det_model, &rec_model, &keys,
    AccelerationDevice::DirectML,
)?;

// Check which device is actually in use
println!("Using: {:?}", engine.device());
```
