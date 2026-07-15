# PaddleOCR-rs

[English](README.md) | [中文](README_CN.md)

ONNX-based OCR engine using PaddleOCR models, written in Rust.

## Modules

- `lib.rs` — `OcrEngine` (detection + recognition) + `DocOrientationClassifier`
- `det.rs` — Text detection via DBNet
- `rec.rs` — Text recognition via CRNN
- `decode.rs` — CTC greedy decoding
- `cls.rs` — Document orientation classification (PP-LCNet)

## API

### Main Structs

#### `OcrEngine`

OCR engine that contains text detection and recognition functionality.

```rust
pub fn new(det_model: &[u8], rec_model: &[u8], keys_data: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
```

| Parameter | Description |
|-----------|-------------|
| `det_model` | Detection model ONNX file byte data |
| `rec_model` | Recognition model ONNX file byte data |
| `keys_data` | Dictionary file byte data (character set) |

```rust
pub fn detect_text_regions(&self, image: &DynamicImage) -> Result<Vec<TextRegion>, String>
```

Detect text regions in the image.

```rust
pub fn recognize_text(&self, image: &DynamicImage, region: &TextRegion) -> Result<DecodedText, String>
```

Recognize text in a specific region.

```rust
pub fn recognize_all(&self, image: &DynamicImage, order: OrderBy) -> Result<Vec<OcrBlock>, String>
```

Complete OCR process: detection + recognition, returns all recognition results.

#### `DocOrientationClassifier`

Document orientation classifier for detecting the rotation angle of document images.

```rust
pub fn new(model_data: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
```

| Parameter | Description |
|-----------|-------------|
| `model_data` | Orientation classification model ONNX file byte data |

```rust
pub fn classify(&self, image: &DynamicImage) -> Result<OrientationResult, String>
```

Classify document orientation.

```rust
pub fn correct_orientation(&self, image: &DynamicImage) -> Result<(DynamicImage, OrientationResult), String>
```

Automatically correct document orientation, returns the corrected image and orientation information.

### Return Types

#### `TextRegion`

```rust
pub struct TextRegion {
    pub bbox: [[f32; 2]; 4],   // Four corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    pub confidence: f32,        // Confidence (0.0 ~ 1.0)
}
```

#### `OcrBlock`

```rust
pub struct OcrBlock {
    pub text: String,           // Recognized text
    pub confidence: f32,        // Confidence (0.0 ~ 1.0)
    pub x: f32,                 // Bounding box top-left x coordinate
    pub y: f32,                 // Bounding box top-left y coordinate
    pub width: f32,             // Bounding box width
    pub height: f32,            // Bounding box height
}
```

#### `DecodedText`

```rust
pub struct DecodedText {
    pub text: String,           // Decoded text
    pub score: f32,             // Average confidence
}
```

#### `OrientationResult`

```rust
pub struct OrientationResult {
    pub orientation: DocOrientation,  // Detected orientation
    pub confidence: f32,              // Confidence (0.0 ~ 1.0)
}
```

#### `DocOrientation`

```rust
pub enum DocOrientation {
    Upright,    // 0° - Normal orientation
    Rotate90,   // 90° - Needs clockwise rotation of 90°
    Rotate180,  // 180° - Needs rotation of 180°
    Rotate270,  // 270° - Needs counterclockwise rotation of 90°
}
```

#### `OrderBy`

```rust
pub enum OrderBy {
    Horizontal,  // Arrange horizontally (top to bottom, left to right)
    Vertical,    // Arrange vertically (right to left, top to bottom)
    Score,       // Arrange by confidence descending
}
```

## Usage

```rust
use paddleocr_rs_onnx::{OcrEngine, DocOrientationClassifier, OrderBy};

// Read model files
let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

// Create OCR engine
let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;

// Complete recognition
let image = image::open("test.png")?;
let blocks = engine.recognize_all(&image, OrderBy::Horizontal)?;
for block in &blocks {
    println!("{} ({:.2}%)", block.text, block.confidence * 100.0);
}

// Orientation correction
let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;
let classifier = DocOrientationClassifier::new(&cls_model)?;
let (corrected, result) = classifier.correct_orientation(&image)?;
println!("Orientation: {}°, Confidence: {:.2}%", result.orientation.angle(), result.confidence * 100.0);
```

## License

MIT

## Thanks

This project is built upon the work of the following projects:

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Provides models
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) - Provides runtime reference
- [MAAFramework](https://github.com/MaaAssistantArknights/MAAFramework) - Provides architecture reference

## Comparison with Other Rust PaddleOCR Implementations

This project is one of several Rust implementations of PaddleOCR. Below is a comprehensive comparison of the three main implementations:

### Acceleration Hardware Support Comparison

| Platform/Backend     | This project (PaddleOCR-rs) | [mg-chao/paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs) | [zibo-chen/rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) |
| -------------------- | ---------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Windows**          |                              |                                                                   |                                                                           |
| CPU                  | ✅ ONNX Runtime (default)    | ✅ ONNX Runtime (default)                                         | ✅ MNN (default)                                                          |
| CUDA (NVIDIA)        | ❌ Not enabled               | ✅ CUDA                                                           | ✅ CUDA                                                                   |
| DirectML (GPU)       | ❌ Not enabled               | ✅ DirectML                                                       | ❌                                                                        |
| **Linux**            |                              |                                                                   |                                                                           |
| CPU                  | ✅ ONNX Runtime (default)    | ✅ ONNX Runtime (default)                                         | ✅ MNN (default)                                                          |
| CUDA (NVIDIA)        | ❌ Not enabled               | ✅ CUDA                                                           | ✅ CUDA                                                                   |
| CANN (Ascend)        | ❌ Not enabled               | ✅ CANN                                                           | ❌                                                                        |
| OpenCL               | ❌                           | ❌                                                                | ✅ OpenCL                                                                 |
| Vulkan               | ❌                           | ❌                                                                | ✅ Vulkan                                                                 |
| **macOS**            |                              |                                                                   |                                                                           |
| CPU                  | ✅ ONNX Runtime (default)    | ✅ ONNX Runtime (default)                                         | ✅ MNN (default)                                                          |
| CoreML (Apple GPU)   | ❌ Not enabled               | ❌                                                                | ✅ CoreML                                                                 |
| Metal (Apple GPU)    | ❌                           | ❌                                                                | ✅ Metal                                                                  |
| **Android**          |                              |                                                                   |                                                                           |
| CPU                  | ❌ Not supported             | ❌ Not supported                                                  | ✅ MNN (default)                                                          |
| OpenCL               | ❌                           | ❌                                                                | ✅ OpenCL                                                                 |
| Vulkan               | ❌                           | ❌                                                                | ✅ Vulkan                                                                 |
| **iOS**              |                              |                                                                   |                                                                           |
| CPU                  | ❌ Not supported             | ❌ Not supported                                                  | ✅ MNN (default)                                                          |
| CoreML (Apple GPU)   | ❌                           | ❌                                                                | ✅ CoreML                                                                 |
| Metal (Apple GPU)    | ❌                           | ❌                                                                | ✅ Metal                                                                  |
| **Other**            |                              |                                                                   |                                                                           |
| OpenGL               | ❌                           | ❌                                                                | ✅ OpenGL                                                                 |
| Accelerator Count    | 0                           | 3 (CUDA + DirectML + CANN)                                        | 6 (CUDA + Metal + CoreML + OpenCL + Vulkan + OpenGL)                     |

### Overview

| Aspect | This project (PaddleOCR-rs) | [mg-chao/paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs) | [zibo-chen/rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) |
|--------|------------------------------|------------------------|---------------------------|
| **Inference backend** | ONNX Runtime via ort crate (pure Rust, no FFI) | ONNX Runtime via ort crate | MNN inference framework via mnn-rs crate |
| **Document orientation classification** | Included (PP-LCNet model, auto rotation correction) | Included (PP-OCR mobile v2.0, 180° rotation) | Included (PP-LCNet model, 0/90/180/270° + 0/180°) |
| **Model format** | Standard ONNX format | Standard ONNX format | MNN model format |
| **Architecture** | Pure Rust with ONNX Runtime | Rust + ONNX Runtime | Rust + MNN bindings |
| **External dependencies** | Minimal dependency set (ort + core crates, no extra download logic) | Full-featured set (ort with explicit download-binaries + YAML/JSON/reqwest etc.) | MNN library (via mnn-rs) |
| **API design** | OcrEngine with fine-grained control: detect_text_regions, recognize_text, recognize_all | Rich pipeline API with YAML config, auto model download, word-level boxes | Multi-level API: Det/Rec, OcrEngine, C API |
| **Model loading** | Loads ONNX models directly from byte slices | Loads ONNX models (auto-download from ModelScope) | Loads MNN models from file paths |
| **Concurrency** | rayon for parallel processing + session pooling | rayon for parallel processing + batch inference | Mainly single-threaded (rayon in pre/post-processing) |
| **Image processing** | image + imageproc crates | Pure Rust image processing (or optional OpenCV) | Rust-native (image + imageproc + ndarray) |
| **Dependencies** | Minimal: ort, image, imageproc, geo, serde, rayon, log, parking_lot | ort + many crates (rayon, nalgebra, geo, serde_yaml, reqwest, turbojpeg) | mnn-rs + many crates (image, imageproc, ndarray, rayon, libc) |
| **Error handling** | Result types with detailed error messages | thiserror-based PaddleOcrError enum (14 variants) | thiserror-based OcrError enum (11 variants) |
| **Platform support** | Cross-platform (ONNX Runtime supports Windows, Linux, macOS) | Cross-platform (ONNX Runtime supports Windows, Linux, macOS) | Cross-platform (MNN supports Windows, Linux, macOS, Android, iOS) |
| **Build complexity** | Simple cargo build | Simple cargo build (auto-downloads ONNX Runtime) | Requires MNN setup |
| **Deployment** | Single binary, no external libs | Requires ONNX Runtime (auto-downloaded) | Requires MNN runtime |
| **Model portability** | Standard ONNX format | Standard ONNX format | MNN format (needs conversion) |
| **Ease of use** | Simple API, minimal setup | Simple setup with auto model download | Requires MNN setup |
| **Performance** | Optimized via ONNX Runtime | Optimized via ONNX Runtime | MNN framework performance |
| **Maintenance** | Active development | Active development (v0.7.0) | Community maintained |
| **Ecosystem** | ⚠️ Limited (ONNX only) | ⚠️ Limited (ONNX, RapidOCR) | ⚠️ Limited (MNN only) |
| **GPU Acceleration** | ❌ Not enabled | ✅ CUDA/DirectML/CANN | ✅ Metal/OpenCL/OpenGL/Vulkan/CUDA/CoreML |
| **External Interfaces** | ❌ Rust API only | ✅ YAML config + CLI (rapidocr) | ✅ C API (cdylib) + CLI (newbee-ocr-cli) |
| **Output Formats** | ❌ Plain text only | ✅ JSON + Markdown + Visualization image | ❌ Plain text only |
| **Text Processing** | ✅ Sorting modes (Horizontal/Vertical/Score) | ✅ Word-level boxes + BiDi text | ✅ FP16 inference + async support |

### Advantages of Each Project

| Project | Key Advantages |
|---------|----------------|
| **This project (PaddleOCR-rs)** | ✅ Cross-platform support (Windows, Linux, macOS)<br>✅ No external dependencies (only ONNX models)<br>✅ Document orientation classification<br>✅ Fine-grained API control<br>✅ Concurrent processing with session pooling<br>✅ Multiple ordering modes<br>✅ Full image recognition fallback |
| **mg-chao/paddle-ocr-rs** | ✅ Rich pipeline API with YAML configuration<br>✅ ONNX Runtime inference with auto model download<br>✅ Word-level bounding boxes<br>✅ Batch processing (recognition + classification)<br>✅ Multiple output formats (JSON, Markdown, visualization)<br>✅ RapidOCR ecosystem compatibility |
| **zibo-chen/rust-paddle-ocr** | ✅ MNN inference for efficient deployment<br>✅ Multi-level API (Det/Rec, OcrEngine, C API)<br>✅ Cross-language C bindings via cbindgen<br>✅ Partial parallel preprocessing<br>✅ CLI tool included<br>✅ Support for PP-OCR models converted to MNN |

### Detailed Comparison

#### Functionality Differences

| Feature | This project (PaddleOCR-rs) | mg-chao/paddle-ocr-rs | zibo-chen/rust-paddle-ocr |
|---------|------------------------------|------------------------|---------------------------|
| **Text Detection** | ✅ DBNet implementation | ✅ DBNet implementation | ✅ DBNet implementation |
| **Text Recognition** | ✅ CRNN implementation | ✅ CRNN implementation | ✅ CRNN implementation |
| **Document Orientation** | ✅ PP-LCNet classifier (0/90/180/270°) | ✅ PP-OCR v2.0 classifier (0/180°) | ✅ PP-LCNet classifier (0/90/180/270° + 0/180°) |
| **Batch Processing** | ✅ Parallel processing with rayon | ✅ Batch inference (rec:6, cls:6) | ⚠️ Partially rayon in pre/post |
| **Session Management** | ✅ Session pooling for concurrency | ❌ Not available | ❌ Not available |
| **Image Preprocessing** | ✅ Rust-native image processing | ✅ Pure Rust (or optional OpenCV) | ✅ Rust-native (imageproc, ndarray) |
| **Model Format Support** | ✅ ONNX only | ✅ ONNX format | ✅ MNN format |
| **API Granularity** | ✅ Fine-grained control | ✅ Rich pipeline control | ✅ Multi-level (raw to high-level) |
| **Error Handling** | ✅ Detailed error messages | ✅ Detailed thiserror enum | ✅ Detailed thiserror enum |

#### Performance Differences

| Aspect | This project (PaddleOCR-rs) | mg-chao/paddle-ocr-rs | zibo-chen/rust-paddle-ocr |
|--------|------------------------------|------------------------|---------------------------|
| **Inference Speed** | Good (ONNX Runtime optimized) | Good (ONNX Runtime optimized) | Good (MNN framework) |
| **Memory Usage** | Moderate (Rust safety overhead) | Moderate (Rust overhead) | Moderate |
| **Startup Time** | Fast (No PaddlePaddle loading) | Fast (auto-downloaded ONNX Runtime) | Slower (MNN initialization) |
| **Parallel Processing** | ✅ Excellent (rayon + session pooling) | ✅ Available (rayon + batch inference) | ⚠️ Partial (rayon in pre/post) |
| **Batch Processing** | ✅ Excellent (Parallel execution) | ✅ Available (batch rec & cls) | ❌ Not available |
| **Resource Efficiency** | Good (Rust safety guarantees) | Good (Rust safety guarantees) | Good (MNN performance) |

#### Capability Differences

| Capability | This project (PaddleOCR-rs) | mg-chao/paddle-ocr-rs | zibo-chen/rust-paddle-ocr |
|------------|------------------------------|------------------------|---------------------------|
| **Cross-platform** | ✅ Excellent (ONNX Runtime) | ✅ Excellent (ONNX Runtime) | ✅ Good (MNN support) |
| **Easy Deployment** | ✅ Excellent (Single binary) | ✅ Good (auto-download) | ⚠️ Requires MNN setup |
| **Model Flexibility** | ✅ Good (ONNX standard) | ✅ Good (ONNX standard) | ⚠️ Limited to MNN format |
| **Developer Experience** | ✅ Good (Rust API) | ✅ Good (YAML config, auto-download) | ✅ Good (multi-level API, CLI) |
| **Documentation** | ✅ Good | ✅ Excellent | ✅ Good |
| **Community Support** | Growing | Active | Growing |

#### Technical Implementation

| Aspect | This project (PaddleOCR-rs) | mg-chao/paddle-ocr-rs | zibo-chen/rust-paddle-ocr |
|--------|------------------------------|------------------------|---------------------------|
| **Language Safety** | ✅ Memory-safe Rust | ✅ Memory-safe Rust | ✅ Memory-safe Rust (mnn-rs) + ⚠️ C API |
| **Type Safety** | ✅ Strong typing | ✅ Strong typing | ✅ Strong typing |
| **Error Propagation** | ✅ Rust Result types (thiserror) | ✅ Rust Result types (thiserror) | ✅ Rust Result types (thiserror) |
| **Memory Management** | ✅ Automatic (Rust) | ✅ Automatic (Rust) | ✅ Automatic (Rust) |
| **Concurrency Safety** | ✅ Thread-safe by design | ✅ Thread-safe (Arc + Mutex) | ⚠️ Requires careful handling |
| **API Design** | ✅ Modern Rust idioms | ✅ Modern Rust idioms | ✅ Modern Rust + C API |

### Dependencies Comparison

| Category | This project | mg-chao/paddle-ocr-rs | zibo-chen/rust-paddle-ocr |
|----------|--------------|------------------------|---------------------------|
| **OCR Core** | ort (ONNX Runtime) | ort (ONNX Runtime) | mnn (MNN framework) |
| **Image Processing** | image, imageproc | Pure Rust (custom) or optional OpenCV | image, imageproc, ndarray |
| **Geometry** | geo | geo-clipper, geo-types, nalgebra | libc |
| **Concurrency** | rayon, parking_lot | rayon, num_cpus | rayon, crossbeam-channel |
| **Serialization** | serde | serde, serde_json, serde_yaml | serde, serde_json |
| **Logging** | log | log | log |

### Summary

**This project (PaddleOCR-rs)** excels in:
- Cross-platform compatibility and easy deployment
- No external dependencies (only ONNX models needed)
- Advanced features including orientation classification (0/90/180/270°)
- Fine-grained API control and concurrent processing
- Memory safety and type safety with Rust

**mg-chao/paddle-ocr-rs** excels in:
- Rich pipeline API with YAML configuration and auto model download
- ONNX Runtime inference with batch processing
- Word-level bounding boxes and BiDi text support
- Multiple output formats (JSON, Markdown, visualization)
- RapidOCR compatibility and established documentation

**zibo-chen/rust-paddle-ocr** excels in:
- MNN inference framework for efficient deployment
- Multi-level API (raw Det/Rec, OcrEngine, C bindings)
- Cross-language C API via cbindgen
- CLI tool for direct OCR recognition
- Partial parallel preprocessing for performance

### Use Case Recommendations

| Use Case | Recommended Project |
|----------|---------------------|
| Cross-platform deployment | **PaddleOCR-rs** |
| Minimal dependencies | **PaddleOCR-rs** |
| Advanced OCR features (orientation, etc.) | **PaddleOCR-rs** |
| Rich pipeline with auto model download | **mg-chao/paddle-ocr-rs** |
| Batch recognition and classification | **mg-chao/paddle-ocr-rs** |
| Multi-language OCR with BiDi support | **mg-chao/paddle-ocr-rs** |
| Lightweight deployment on mobile | **zibo-chen/rust-paddle-ocr** |
| Cross-language C/C++ integration | **zibo-chen/rust-paddle-ocr** |

For the latest features and updates, please refer to the respective repositories.

### Note

The original repository may have evolved since this fork was created. For the latest features, please refer to the [upstream repository](https://github.com/mg-chao/paddle-ocr-rs).


