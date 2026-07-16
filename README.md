# PaddleOCR-rs

[English](README.md) | [中文](README_CN.md)

ONNX-based OCR engine using PaddleOCR models, written in Rust.

## Features

- **Text Detection & Recognition** — Complete OCR pipeline with DBNet detection and CRNN recognition
- **Document Orientation Classification** — PP-LCNet classifier for 0°/90°/180°/270° rotation detection and auto-correction
- **Hardware Acceleration** — ONNX Runtime with CPU, DirectML, CUDA, OpenVINO, NNAPI, CoreML, CANN support
- **Cross-platform** — Windows, Linux, macOS, Android (via FFI), iOS (via FFI)
- **Concurrent Processing** — rayon parallel execution + session pooling
- **Fine-grained Control** — Step-by-step API: detect → recognize, or full pipeline with ordering modes
- **Minimal Dependencies** — Only ONNX models needed, no external runtime libraries

## Quick Start

```rust
use paddleocr_rs_onnx::{OcrEngine, OrderBy};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;

let image = image::open("test.png")?;
let blocks = engine.recognize_all(&image, OrderBy::Horizontal)?;
for block in &blocks {
    println!("{} ({:.2}%)", block.text, block.confidence * 100.0);
}
```

## Installation

```toml
[dependencies]
paddleocr_rs_onnx = "0.2"
```

## API Documentation

- [API Overview](docs/en/api-overview.md) — Modules, structs, return types
- [OcrEngine](docs/en/ocr-engine.md) — Detection and recognition API
- [DocOrientationClassifier](docs/en/doc-orientation.md) — Orientation detection and correction
- [Hardware Acceleration](docs/en/acceleration.md) — AccelerationDevice, platform requirements, EP features
- [FFI API](src/ffi.rs) — C-compatible API

## Mobile Platform Support

PaddleOCR-rs supports Android and iOS platforms via C FFI interface.

### Android

- **CPU**: ✅ Supported (via ONNX Runtime)
- **NNAPI**: ✅ Supported (via nnapi feature)
- **Targets**: aarch64-linux-android, armv7-linux-androideabi, x86_64-linux-android

### iOS

- **CPU**: ✅ Supported (via ONNX Runtime)
- **CoreML**: ✅ Supported (via coreml feature)
- **Targets**: aarch64-apple-ios, aarch64-apple-ios-sim

### Building for Mobile

```bash
# Android
./build-android.sh aarch64-linux-android --release

# iOS
./build-ios.sh aarch64-apple-ios --release
```

### FFI Interface

Enable the `ffi` feature to expose C-compatible API:

```toml
[dependencies]
paddleocr_rs_onnx = { version = "0.2", features = ["ffi"] }
```

See `src/ffi.rs` for the complete FFI API documentation.

### Example Projects

- [xc-ocr-onnx](https://github.com/Craun718/xc-ocr-onnx) — Tauri 2 desktop GUI app, supports image / DOCX / PDF OCR with dynamic model switching
- `examples/android-demo/` — Android Kotlin example
- `examples/ios-demo/` — iOS Swift example

## Comparison with Other Rust PaddleOCR Implementations

This project is one of several Rust implementations of PaddleOCR. Below is a comprehensive comparison of the three main implementations:

### Acceleration Hardware Support Comparison

| Platform/Backend | PaddleOCR-rs                         | [mg-chao/paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs) | [zibo-chen/rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) |
| ---------------- | ------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Windows**      |                                      |                                                                   |                                                                           |
| CUDA             | ✅ CUDA (via `cuda` feature)         | ✅ CUDA                                                           | ✅ CUDA                                                                   |
| DirectML         | ✅ DirectML                          | ✅ DirectML                                                       | ❌                                                                        |
| OpenVINO         | ✅ OpenVINO (via `openvino` feature) | ❌                                                                | ❌                                                                        |
| **Linux**        |                                      |                                                                   |                                                                           |
| CUDA             | ✅ CUDA (via `cuda` feature)         | ✅ CUDA                                                           | ✅ CUDA                                                                   |
| CANN             | ✅ CANN (via `cann` feature)         | ✅ CANN                                                           | ❌                                                                        |
| OpenVINO         | ✅ OpenVINO (via `openvino` feature) | ❌                                                                | ❌                                                                        |
| **macOS**        |                                      |                                                                   |                                                                           |
| Metal            | ✅ Metal (via `metal` feature)       | ❌                                                                | ✅ Metal                                                                  |
| CoreML           | ✅ CoreML (via `coreml` feature)     | ❌                                                                | ✅ CoreML                                                                 |
| **Android**      |                                      |                                                                   |                                                                           |
| NNAPI            | ✅ NNAPI (via `nnapi` feature)       | ❌                                                                | ❌                                                                        |
| CPU              | ✅ CPU                               | ✅ CPU                                                            | ✅ CPU                                                                    |
| **iOS**          |                                      |                                                                   |                                                                           |
| CoreML           | ✅ CoreML (via `coreml` feature)     | ❌                                                                | ✅ CoreML                                                                 |
| CPU              | ✅ CPU                               | ✅ CPU                                                            | ✅ CPU                                                                    |

### Comprehensive Comparison

| Feature                    | PaddleOCR-rs                                                      | [mg-chao/paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs)             | [zibo-chen/rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) |
| -------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Model Format**           | ✅ ONNX only                                                      | ✅ ONNX format                                                                | ✅ MNN format                                                             |
| **Backend/Runtime**        | ONNX Runtime (feature flags)                                      | ONNX Runtime (via ort crate)                                                  | MNN Framework (via mnn-rs)                                                |
| **Document Orientation**   | ✅ PP-LCNet classifier (0/90/180/270°)                            | ✅ PP-OCR v2.0 classifier (0/180°)                                            | ✅ PP-LCNet classifier (0/90/180/270° + 0/180°)                           |
| **Concurrency**            | ✅ rayon parallel + session pooling                               | ✅ rayon parallel + batch inference (6 images for recognition/classification) | ⚠️ rayon in pre/post-processing, inference is single-threaded             |
| **Image Preprocessing**    | ✅ Rust native image processing                                   | ✅ Pure Rust implementation (or optional OpenCV)                              | ✅ Rust native (image + imageproc + ndarray)                              |
| **Model Format Support**   | ✅ ONNX only                                                      | ✅ ONNX format                                                                | ✅ MNN format                                                             |
| **Platform Compatibility** | ✅ Excellent (ONNX Runtime cross-platform, single binary)         | ✅ Excellent (ONNX Runtime cross-platform)                                    | ✅ Good (MNN supports multiple platforms)                                 |
| **External Interfaces**    | ✅ Rust API + C FFI API (via `ffi` feature)                       | ✅ YAML config + CLI (rapidocr)                                               | ✅ C API (cdylib) + CLI (newbee-ocr-cli)                                  |
| **Text Processing**        | ✅ Sorting modes (Horizontal/Vertical/Score)                      | ✅ Word-level boxes + BiDi text                                               | ✅ FP16 inference + async support                                         |
| **Memory/Type Safety**     | ✅ Memory-safe Rust + strong typing + automatic memory management | ✅ Memory-safe Rust + strong typing                                           | ✅ Memory-safe Rust (mnn-rs) + ⚠️ C API partial                           |
| **Error Handling & API**   | ✅ Rust Result types + modern idioms                              | ✅ Rust Result types (thiserror)                                              | ✅ Rust Result types (thiserror)                                          |
| **Concurrency Safety**     | ✅ Thread-safe by design                                          | ✅ Thread-safe (Arc + Mutex)                                                  | ⚠️ Requires careful handling                                              |

## Thanks

This project is built upon the work of the following projects:

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Provides models
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) - Provides runtime reference
- [MAAFramework](https://github.com/MaaAssistantArknights/MAAFramework) - Provides architecture reference

## License

MIT
