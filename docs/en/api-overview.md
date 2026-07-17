# API Overview

[← Back to README](../../README.md)

## Modules

- `lib.rs` — `OcrEngine` (detection + recognition) + `DocOrientationClassifier`
- `det.rs` — Text detection via DBNet
- `rec.rs` — Text recognition via CRNN
- `decode.rs` — CTC greedy decoding
- `cls.rs` — Document orientation classification (PP-LCNet)

## Main Structs

| Struct | Description | Docs |
|--------|-------------|------|
| `OcrEngine` | OCR engine (detection + recognition) | [ocr-engine.md](ocr-engine.md) |
| `DocOrientationClassifier` | Document orientation classifier | [doc-orientation.md](doc-orientation.md) |

## Return Types

### `TextRegion`

```rust
pub struct TextRegion {
    pub bbox: [[f32; 2]; 4],   // Four corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    pub confidence: f32,        // Confidence (0.0 ~ 1.0)
}
```

### `OcrBlock`

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

### `DecodedText`

```rust
pub struct DecodedText {
    pub text: String,           // Decoded text
    pub score: f32,             // Average confidence
}
```

### `OrientationResult`

```rust
pub struct OrientationResult {
    pub orientation: DocOrientation,  // Detected orientation
    pub confidence: f32,              // Confidence (0.0 ~ 1.0)
}
```

### `DocOrientation`

```rust
pub enum DocOrientation {
    Upright,    // 0° - Normal orientation
    Rotate90,   // 90° - Needs clockwise rotation of 90°
    Rotate180,  // 180° - Needs rotation of 180°
    Rotate270,  // 270° - Needs counterclockwise rotation of 90°
}
```

### `OrderBy`

```rust
pub enum OrderBy {
    Horizontal,  // Arrange horizontally (top to bottom, left to right)
    Vertical,    // Arrange vertically (right to left, top to bottom)
    Score,       // Arrange by confidence descending
}
```

### `AccelerationDevice`

Hardware acceleration device for ONNX Runtime inference.

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

Key methods:
- `is_available(&self) -> bool` — Check if the device is available (compile-time feature + runtime EP check)
- `available_devices() -> Vec<AccelerationDevice>` — List all available devices in the current environment

See [acceleration.md](acceleration.md) for detailed platform requirements, availability checking, and runtime behavior.
