# OcrEngine

[← 返回 API 概览](api-overview.md)

OCR 引擎，包含文本检测和识别功能。

## 构造方法

### `new`

```rust
pub fn new(
    det_model: &[u8],
    rec_model: &[u8],
    keys_data: &[u8],
) -> Result<Self, Box<dyn std::error::Error>>
```

| 参数 | 说明 |
|------|------|
| `det_model` | 检测模型 ONNX 文件的字节数据 |
| `rec_model` | 识别模型 ONNX 文件的字节数据 |
| `keys_data` | 字典文件的字节数据（字符集） |

### `new_with_device`

使用硬件加速创建引擎：

```rust
pub fn new_with_device(
    det_model: &[u8],
    rec_model: &[u8],
    keys_data: &[u8],
    device: AccelerationDevice,
) -> Result<Self, Box<dyn std::error::Error>>
```

| 参数 | 说明 |
|------|------|
| `det_model` | 检测模型 ONNX 文件的字节数据 |
| `rec_model` | 识别模型 ONNX 文件的字节数据 |
| `keys_data` | 字典文件的字节数据（字符集） |
| `device` | 硬件加速设备（参见 `AccelerationDevice` 枚举） |

### `device`

获取加速设备：

```rust
pub fn device(&self) -> AccelerationDevice
```

## 方法

### `detect_text_regions`

检测图像中的文本区域。

```rust
pub fn detect_text_regions(
    &self,
    image: &DynamicImage,
) -> Result<Vec<TextRegion>, String>
```

### `recognize_text`

识别指定区域的文本。

```rust
pub fn recognize_text(
    &self,
    image: &DynamicImage,
    region: &TextRegion,
) -> Result<DecodedText, String>
```

### `recognize_all`

完整 OCR 流程：检测 + 识别，返回所有识别结果。

```rust
pub fn recognize_all(
    &self,
    image: &DynamicImage,
    order: OrderBy,
) -> Result<Vec<OcrBlock>, String>
```

## 示例

### 基本用法

```rust
use paddleocr_rs_onnx::{OcrEngine, OrderBy};

// 读取模型文件
let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

// 创建 OCR 引擎
let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;

// 完整识别
let image = image::open("test.png")?;
let blocks = engine.recognize_all(&image, OrderBy::Horizontal)?;
for block in &blocks {
    println!("{} ({:.2}%)", block.text, block.confidence * 100.0);
}
```

### 使用硬件加速

```rust
use paddleocr_rs_onnx::{OcrEngine, OrderBy, AccelerationDevice};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

// 使用 DirectML 加速（Windows）
let engine = OcrEngine::new_with_device(
    &det_model, &rec_model, &keys,
    AccelerationDevice::DirectML,
)?;

// 或 CUDA 加速（NVIDIA GPU）
let engine = OcrEngine::new_with_device(
    &det_model, &rec_model, &keys,
    AccelerationDevice::Cuda,
)?;

let image = image::open("test.png")?;
let blocks = engine.recognize_all(&image, OrderBy::Score)?;
```

### 分步使用：仅检测

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

### 分步使用：仅识别

```rust
use paddleocr_rs_onnx::OcrEngine;

// 检测之后...
let regions = engine.detect_text_regions(&image)?;
if let Some(region) = regions.first() {
    let result = engine.recognize_text(&image, region)?;
    println!("{} ({:.2}%)", result.text, result.score * 100.0);
}
```
