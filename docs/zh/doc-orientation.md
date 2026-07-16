# DocOrientationClassifier

[← 返回 API 概览](api-overview.md)

文档方向分类器，用于检测文档图像的旋转角度。

## 构造方法

### `new`

```rust
pub fn new(
    model_data: &[u8],
) -> Result<Self, Box<dyn std::error::Error>>
```

| 参数 | 说明 |
|------|------|
| `model_data` | 方向分类模型 ONNX 文件的字节数据 |

### `new_with_device`

使用硬件加速创建分类器：

```rust
pub fn new_with_device(
    model_data: &[u8],
    device: AccelerationDevice,
) -> Result<Self, Box<dyn std::error::Error>>
```

| 参数 | 说明 |
|------|------|
| `model_data` | 方向分类模型 ONNX 文件的字节数据 |
| `device` | 硬件加速设备（参见 `AccelerationDevice` 枚举） |

### `device`

获取加速设备：

```rust
pub fn device(&self) -> AccelerationDevice
```

## 方法

### `classify`

分类文档方向。

```rust
pub fn classify(
    &self,
    image: &DynamicImage,
) -> Result<OrientationResult, String>
```

### `correct_orientation`

自动校正文档方向，返回校正后的图像和方向信息。

```rust
pub fn correct_orientation(
    &self,
    image: &DynamicImage,
) -> Result<(DynamicImage, OrientationResult), String>
```

## 示例

### 基本用法

```rust
use paddleocr_rs_onnx::DocOrientationClassifier;

let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;
let classifier = DocOrientationClassifier::new(&cls_model)?;

let image = image::open("document.png")?;
let result = classifier.classify(&image)?;

println!("方向: {:?}, 置信度: {:.2}%", result.orientation, result.confidence * 100.0);
```

### 自动校正方向

```rust
use paddleocr_rs_onnx::DocOrientationClassifier;

let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;
let classifier = DocOrientationClassifier::new(&cls_model)?;

let image = image::open("rotated_document.png")?;
let (corrected_image, result) = classifier.correct_orientation(&image)?;

println!("检测到: {}°, 置信度: {:.2}%",
    result.orientation.angle(), result.confidence * 100.0);

// 保存校正后的图像
corrected_image.save("corrected_document.png")?;
```

### 结合 OCR 流程

```rust
use paddleocr_rs_onnx::{OcrEngine, DocOrientationClassifier, OrderBy};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;
let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;

let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;
let classifier = DocOrientationClassifier::new(&cls_model)?;

let image = image::open("document.png")?;

// 先校正方向，再进行 OCR
let (corrected, _) = classifier.correct_orientation(&image)?;
let blocks = engine.recognize_all(&corrected, OrderBy::Horizontal)?;

for block in &blocks {
    println!("{}", block.text);
}
```
