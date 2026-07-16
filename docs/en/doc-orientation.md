# DocOrientationClassifier

[← Back to API Overview](api-overview.md)

Document orientation classifier for detecting the rotation angle of document images.

## Constructor

### `new`

```rust
pub fn new(
    model_data: &[u8],
) -> Result<Self, Box<dyn std::error::Error>>
```

| Parameter | Description |
|-----------|-------------|
| `model_data` | Orientation classification model ONNX file byte data |

### `new_with_device`

Create classifier with hardware acceleration:

```rust
pub fn new_with_device(
    model_data: &[u8],
    device: AccelerationDevice,
) -> Result<Self, Box<dyn std::error::Error>>
```

| Parameter | Description |
|-----------|-------------|
| `model_data` | Orientation classification model ONNX file byte data |
| `device` | Hardware acceleration device (see `AccelerationDevice` enum) |

### `device`

Get the acceleration device:

```rust
pub fn device(&self) -> AccelerationDevice
```

## Methods

### `classify`

Classify document orientation.

```rust
pub fn classify(
    &self,
    image: &DynamicImage,
) -> Result<OrientationResult, String>
```

### `correct_orientation`

Automatically correct document orientation, returns the corrected image and orientation information.

```rust
pub fn correct_orientation(
    &self,
    image: &DynamicImage,
) -> Result<(DynamicImage, OrientationResult), String>
```

## Examples

### Basic Usage

```rust
use paddleocr_rs_onnx::DocOrientationClassifier;

let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;
let classifier = DocOrientationClassifier::new(&cls_model)?;

let image = image::open("document.png")?;
let result = classifier.classify(&image)?;

println!("Orientation: {:?}, Confidence: {:.2}%", result.orientation, result.confidence * 100.0);
```

### Auto-Correct Orientation

```rust
use paddleocr_rs_onnx::DocOrientationClassifier;

let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;
let classifier = DocOrientationClassifier::new(&cls_model)?;

let image = image::open("rotated_document.png")?;
let (corrected_image, result) = classifier.correct_orientation(&image)?;

println!("Detected: {}°, Confidence: {:.2}%",
    result.orientation.angle(), result.confidence * 100.0);

// Save corrected image
corrected_image.save("corrected_document.png")?;
```

### With OCR Pipeline

```rust
use paddleocr_rs_onnx::{OcrEngine, DocOrientationClassifier, OrderBy};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;
let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;

let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;
let classifier = DocOrientationClassifier::new(&cls_model)?;

let image = image::open("document.png")?;

// Correct orientation first, then OCR
let (corrected, _) = classifier.correct_orientation(&image)?;
let blocks = engine.recognize_all(&corrected, OrderBy::Horizontal)?;

for block in &blocks {
    println!("{}", block.text);
}
```
