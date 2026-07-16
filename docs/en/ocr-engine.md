# OcrEngine

[← Back to API Overview](api-overview.md)

OCR engine that contains text detection and recognition functionality.

## Constructor

### `new`

```rust
pub fn new(
    det_model: &[u8],
    rec_model: &[u8],
    keys_data: &[u8],
) -> Result<Self, Box<dyn std::error::Error>>
```

| Parameter | Description |
|-----------|-------------|
| `det_model` | Detection model ONNX file byte data |
| `rec_model` | Recognition model ONNX file byte data |
| `keys_data` | Dictionary file byte data (character set) |

### `new_with_device`

Create engine with hardware acceleration:

```rust
pub fn new_with_device(
    det_model: &[u8],
    rec_model: &[u8],
    keys_data: &[u8],
    device: AccelerationDevice,
) -> Result<Self, Box<dyn std::error::Error>>
```

| Parameter | Description |
|-----------|-------------|
| `det_model` | Detection model ONNX file byte data |
| `rec_model` | Recognition model ONNX file byte data |
| `keys_data` | Dictionary file byte data (character set) |
| `device` | Hardware acceleration device (see `AccelerationDevice` enum) |

### `device`

Get the acceleration device:

```rust
pub fn device(&self) -> AccelerationDevice
```

## Methods

### `detect_text_regions`

Detect text regions in the image.

```rust
pub fn detect_text_regions(
    &self,
    image: &DynamicImage,
) -> Result<Vec<TextRegion>, String>
```

### `recognize_text`

Recognize text in a specific region.

```rust
pub fn recognize_text(
    &self,
    image: &DynamicImage,
    region: &TextRegion,
) -> Result<DecodedText, String>
```

### `recognize_all`

Complete OCR process: detection + recognition, returns all recognition results.

```rust
pub fn recognize_all(
    &self,
    image: &DynamicImage,
    order: OrderBy,
) -> Result<Vec<OcrBlock>, String>
```

## Examples

### Basic Usage

```rust
use paddleocr_rs_onnx::{OcrEngine, OrderBy};

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
```

### With Hardware Acceleration

```rust
use paddleocr_rs_onnx::{OcrEngine, OrderBy, AccelerationDevice};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

// Create with DirectML acceleration (Windows)
let engine = OcrEngine::new_with_device(
    &det_model, &rec_model, &keys,
    AccelerationDevice::DirectML,
)?;

// Or CUDA acceleration (NVIDIA GPU)
let engine = OcrEngine::new_with_device(
    &det_model, &rec_model, &keys,
    AccelerationDevice::Cuda,
)?;

let image = image::open("test.png")?;
let blocks = engine.recognize_all(&image, OrderBy::Score)?;
```

### Step-by-Step: Detection Only

```rust
use paddleocr_rs_onnx::OcrEngine;

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;
let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;

let image = image::open("test.png")?;
let regions = engine.detect_text_regions(&image)?;

for region in &regions {
    println!("bbox: {:?}, confidence: {:.2}%", region.bbox, region.confidence * 100.0);
}
```

### Step-by-Step: Recognition Only

```rust
use paddleocr_rs_onnx::OcrEngine;

// After detection...
let regions = engine.detect_text_regions(&image)?;
if let Some(region) = regions.first() {
    let result = engine.recognize_text(&image, region)?;
    println!("{} ({:.2}%)", result.text, result.score * 100.0);
}
```
