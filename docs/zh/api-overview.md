# API 概览

[← 返回 README](../../README_CN.md)

## 模块

- `lib.rs` — `OcrEngine`（检测 + 识别）+ `DocOrientationClassifier`
- `det.rs` — 基于 DBNet 的文本检测
- `rec.rs` — 基于 CRNN 的文本识别
- `decode.rs` — CTC 贪心解码
- `cls.rs` — 文档方向分类（PP-LCNet）

## 主要结构体

| 结构体 | 说明 | 文档 |
|--------|------|------|
| `OcrEngine` | OCR 引擎（检测 + 识别） | [ocr-engine.md](ocr-engine.md) |
| `DocOrientationClassifier` | 文档方向分类器 | [doc-orientation.md](doc-orientation.md) |

## 返回值类型

### `TextRegion`

```rust
pub struct TextRegion {
    pub bbox: [[f32; 2]; 4],   // 四个角点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    pub confidence: f32,        // 置信度 (0.0 ~ 1.0)
}
```

### `OcrBlock`

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

### `DecodedText`

```rust
pub struct DecodedText {
    pub text: String,           // 解码后的文本
    pub score: f32,             // 平均置信度
}
```

### `OrientationResult`

```rust
pub struct OrientationResult {
    pub orientation: DocOrientation,  // 检测到的方向
    pub confidence: f32,              // 置信度 (0.0 ~ 1.0)
}
```

### `DocOrientation`

```rust
pub enum DocOrientation {
    Upright,    // 0° - 正常方向
    Rotate90,   // 90° - 需要顺时针旋转 90°
    Rotate180,  // 180° - 需要旋转 180°
    Rotate270,  // 270° - 需要逆时针旋转 90°
}
```

### `OrderBy`

```rust
pub enum OrderBy {
    Horizontal,  // 按水平顺序排列（从上到下，从左到右）
    Vertical,    // 按垂直顺序排列（从右到左，从上到下）
    Score,       // 按置信度降序排列
}
```

### `AccelerationDevice`

ONNX Runtime 推理的硬件加速设备。

```rust
pub enum AccelerationDevice {
    Cpu,        // 仅 CPU 推理（默认，始终可用）
    DirectML,   // DirectML 加速（Windows，DirectX 12 GPU）
    Cuda,       // CUDA 加速（NVIDIA GPU）
    OpenVINO,   // Intel OpenVINO 加速（Windows/Linux）
    Nnapi,      // Android NNAPI 加速（Android）
    Coreml,     // Apple CoreML 加速（macOS/iOS）
    Cann,       // 华为 CANN / 昇腾 NPU 加速（Linux）
}
```

关键方法：
- `is_available(&self) -> bool` — 检查设备是否可用（编译期 feature + 运行时 EP 检查）
- `available_devices() -> Vec<AccelerationDevice>` — 列出当前环境下所有可用设备

详见 [acceleration.md](acceleration.md) 了解平台要求、可用性检查和运行时行为。
