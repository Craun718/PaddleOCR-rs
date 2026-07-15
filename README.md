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
