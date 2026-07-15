# PaddleOCR-rs

[English](README.md) | [中文](README_CN.md)

基于 ONNX 的 OCR 引擎，使用 PaddleOCR 模型，使用 Rust 编写。

## 模块

- `lib.rs` — `OcrEngine`（检测 + 识别）+ `DocOrientationClassifier`
- `det.rs` — 基于 DBNet 的文本检测
- `rec.rs` — 基于 CRNN 的文本识别
- `decode.rs` — CTC 贪心解码
- `cls.rs` — 文档方向分类（PP-LCNet）

## API 接口

### 主要结构体

#### `OcrEngine`

OCR 引擎，包含文本检测和识别功能。

```rust
pub fn new(det_model: &[u8], rec_model: &[u8], keys_data: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
```

| 参数 | 说明 |
|------|------|
| `det_model` | 检测模型 ONNX 文件的字节数据 |
| `rec_model` | 识别模型 ONNX 文件的字节数据 |
| `keys_data` | 字典文件的字节数据（字符集） |

```rust
pub fn detect_text_regions(&self, image: &DynamicImage) -> Result<Vec<TextRegion>, String>
```

检测图像中的文本区域。

```rust
pub fn recognize_text(&self, image: &DynamicImage, region: &TextRegion) -> Result<DecodedText, String>
```

识别指定区域的文本。

```rust
pub fn recognize_all(&self, image: &DynamicImage, order: OrderBy) -> Result<Vec<OcrBlock>, String>
```

完整 OCR 流程：检测 + 识别，返回所有识别结果。

#### `DocOrientationClassifier`

文档方向分类器，用于检测文档图像的旋转角度。

```rust
pub fn new(model_data: &[u8]) -> Result<Self, Box<dyn std::error::Error>>
```

| 参数 | 说明 |
|------|------|
| `model_data` | 方向分类模型 ONNX 文件的字节数据 |

```rust
pub fn classify(&self, image: &DynamicImage) -> Result<OrientationResult, String>
```

分类文档方向。

```rust
pub fn correct_orientation(&self, image: &DynamicImage) -> Result<(DynamicImage, OrientationResult), String>
```

自动校正文档方向，返回校正后的图像和方向信息。

### 返回值类型

#### `TextRegion`

```rust
pub struct TextRegion {
    pub bbox: [[f32; 2]; 4],   // 四个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    pub confidence: f32,        // 置信度 (0.0 ~ 1.0)
}
```

#### `OcrBlock`

```rust
pub struct OcrBlock {
    pub text: String,           // 识别出的文本
    pub confidence: f32,        // 置信度 (0.0 ~ 1.0)
    pub x: f32,                 // 边界框左上角 x 坐标
    pub y: f32,                 // 边界框左上角 y 坐标
    pub width: f32,             // 边界框宽度
    pub height: f32,            // 边界框高度
}
```

#### `DecodedText`

```rust
pub struct DecodedText {
    pub text: String,           // 解码后的文本
    pub score: f32,             // 平均置信度
}
```

#### `OrientationResult`

```rust
pub struct OrientationResult {
    pub orientation: DocOrientation,  // 检测到的方向
    pub confidence: f32,              // 置信度 (0.0 ~ 1.0)
}
```

#### `DocOrientation`

```rust
pub enum DocOrientation {
    Upright,    // 0° - 正常方向
    Rotate90,   // 90° - 需要顺时针旋转 90°
    Rotate180,  // 180° - 需要旋转 180°
    Rotate270,  // 270° - 需要逆时针旋转 90°
}
```

#### `OrderBy`

```rust
pub enum OrderBy {
    Horizontal,  // 按水平顺序排列（从上到下，从左到右）
    Vertical,    // 按垂直顺序排列（从右到左，从上到下）
    Score,       // 按置信度降序排列
}
```

## 使用方法

```rust
use paddleocr_rs_onnx::{OcrEngine, DocOrientationClassifier, OrderBy};

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

// 方向校正
let cls_model = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;
let classifier = DocOrientationClassifier::new(&cls_model)?;
let (corrected, result) = classifier.correct_orientation(&image)?;
println!("方向: {}°, 置信度: {:.2}%", result.orientation.angle(), result.confidence * 100.0);
```

## 许可证

MIT

## 致谢

本项目基于以下项目构建：

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 提供模型
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) - 提供运行时参考
- [MAAFramework](https://github.com/MaaAssistantArknights/MAAFramework) - 提供架构参考
