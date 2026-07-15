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

| 参数        | 说明                         |
| ----------- | ---------------------------- |
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

| 参数         | 说明                             |
| ------------ | -------------------------------- |
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
    pub confidence: f32,     // 置信度 (0.0 ~ 1.0)
}
```

#### `OcrBlock`

```rust
pub struct OcrBlock {
    pub text: String,     // 识别出的文本
    pub confidence: f32,     // 置信度 (0.0 ~ 1.0)
    pub x: f32,        // 边界框左上角 x 坐标
    pub y: f32,        // 边界框左上角 y 坐标
    pub width: f32,       // 边界框宽度
    pub height: f32,      // 边界框高度
}
```

#### `DecodedText`

```rust
pub struct DecodedText {
    pub text: String,     // 解码后的文本
    pub score: f32,       // 平均置信度
}
```

#### `OrientationResult`

```rust
pub struct OrientationResult {
    pub orientation: DocOrientation,  // 检测到的方向
    pub confidence: f32,        // 置信度 (0.0 ~ 1.0)
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

````rust
pub enum OrderBy {
    Horizontal,  // 按水平顺序排列（从上到下，从左到右）
    Vertical,    // 按垂直顺序排列（从右到左，从上到下）
    Score,    // 按置信度降序排列

#### `AccelerationDevice`

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
    DirectML,   // DirectML 加速（Windows，DirectX 12 GPU）
    Cuda,       // CUDA 加速（NVIDIA GPU）
    OpenVINO,   // Intel OpenVINO 加速（Windows/Linux）
    Nnapi,      // Android NNAPI 加速（Android）
    Coreml,     // Apple CoreML 加速（macOS/iOS）
    Cann,       // 华为 CANN / 昇腾 NPU 加速（Linux）
}
````

**要求：**

- **Cpu**: 无额外要求（始终可用）
- **DirectML**: Windows 10/11，需要支持 DirectX 12 的 GPU
- **CUDA**: 需要安装 CUDA 工具包的 NVIDIA GPU
- **OpenVINO**: 安装了 OpenVINO 工具包的 Intel CPU/GPU
- **NNAPI**: 支持 NNAPI 的 Android 设备
- **CoreML**: macOS 10.13+ 或 iOS 11+ 设备
- **CANN**: 安装了 CANN 工具包的华为昇腾 NPU

**运行时行为：** 如果请求的 EP 在运行时不可用（未编译到 ORT 二进制文件中、缺少运行时库或不支持当前平台），系统会记录警告并自动回退到 CPU 推理。

**按 feature 控制的 EP 可用性：**
- ✅ **CPU** — 始终可用
- ✅ **DirectML** — 包含在默认 Windows 构建中（无需 feature）
- ❌ **CUDA** — 需要 `cuda` feature + CUDA 工具包
- ❌ **OpenVINO** — 需要 `openvino` feature + 自定义 ORT 构建
- ❌ **NNAPI** — 需要 `nnapi` feature + 自定义 ORT 构建
- ❌ **CoreML** — 需要 `coreml` feature + 自定义 ORT 构建
- ❌ **CANN** — 需要 `cann` feature + 自定义 ORT 构建

**通过 Cargo features 启用 EP：**
```toml
[dependencies]
paddleocr_rs_onnx = { version = "0.1", features = ["cuda"] }

# 多个 EP（回退顺序：先启用的 EP 优先）
paddleocr_rs_onnx = { version = "0.1", features = ["cuda", "directml"] }
```
}
```

## Usage
## 使用方法

```rust
use paddleocr_rs_onnx::{OcrEngine, DocOrientationClassifier, OrderBy, AccelerationDevice};

// 读取模型文件
let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

// 创建 OCR 引擎
let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;

// 或使用硬件加速创建
let engine = OcrEngine::new_with_device(&det_model, &rec_model, &keys, AccelerationDevice::DirectML)?;
let engine = OcrEngine::new_with_device(&det_model, &rec_model, &keys, AccelerationDevice::Cuda)?;

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
````

## 许可证

MIT

## 致谢

本项目基于以下项目构建：

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 提供模型
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) - 提供运行时参考
- [MAAFramework](https://github.com/MaaAssistantArknights/MAAFramework) - 提供架构参考

## 与其他 Rust PaddleOCR 实现的对比

本项目是 PaddleOCR 的几种 Rust 实现之一。以下是三个主要实现的全面对比：

### 加速硬件支持对比

| 平台/后端           | 本项目 (PaddleOCR-rs)   | [mg-chao/paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs) | [zibo-chen/rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) |
| ------------------- | ----------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Windows**         |                         |                                                                   |                                                                           |
| CPU                 | ✅ ONNX Runtime（默认） | ✅ ONNX Runtime（默认）                                           | ✅ MNN（默认）                                                            |
| CUDA (NVIDIA)       | ❌ 未启用               | ✅ CUDA                                                           | ✅ CUDA                                                                   |
| DirectML (通用 GPU) | ❌ 未启用               | ✅ DirectML                                                       | ❌                                                                        |
| **Linux**           |                         |                                                                   |                                                                           |
| CPU                 | ✅ ONNX Runtime（默认） | ✅ ONNX Runtime（默认）                                           | ✅ MNN（默认）                                                            |
| CUDA (NVIDIA)       | ❌ 未启用               | ✅ CUDA                                                           | ✅ CUDA                                                                   |
| CANN (昇腾)         | ❌ 未启用               | ✅ CANN                                                           | ❌                                                                        |
| OpenCL              | ❌                      | ❌                                                                | ✅ OpenCL                                                                 |
| Vulkan              | ❌                      | ❌                                                                | ✅ Vulkan                                                                 |
| **macOS**           |                         |                                                                   |                                                                           |
| CPU                 | ✅ ONNX Runtime（默认） | ✅ ONNX Runtime（默认）                                           | ✅ MNN（默认）                                                            |
| CoreML (Apple GPU)  | ❌ 未启用               | ❌                                                                | ✅ CoreML                                                                 |
| Metal (Apple GPU)   | ❌                      | ❌                                                                | ✅ Metal                                                                  |
| **Android**         |                         |                                                                   |                                                                           |
| CPU                 | ❌ 不支持               | ❌ 不支持                                                         | ✅ MNN（默认）                                                            |
| OpenCL              | ❌                      | ❌                                                                | ✅ OpenCL                                                                 |
| Vulkan              | ❌                      | ❌                                                                | ✅ Vulkan                                                                 |
| **iOS**             |                         |                                                                   |                                                                           |
| CPU                 | ❌ 不支持               | ❌ 不支持                                                         | ✅ MNN（默认）                                                            |
| CoreML (Apple GPU)  | ❌                      | ❌                                                                | ✅ CoreML                                                                 |
| Metal (Apple GPU)   | ❌                      | ❌                                                                | ✅ Metal                                                                  |
| **其他**            |                         |                                                                   |                                                                           |
| OpenGL              | ❌                      | ❌                                                                | ✅ OpenGL                                                                 |
| 硬件后端总数        | 0 个加速后端            | 3 个加速后端（CUDA + DirectML + CANN）                            | 6 个加速后端（CUDA + Metal + CoreML + OpenCL + Vulkan + OpenGL）          |

### 全面对比

| 方面               | 本项目 (PaddleOCR-rs)                                         | [mg-chao/paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs) | [zibo-chen/rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) |
| ------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **技术架构**       | 纯 Rust + ONNX Runtime（通过 ort crate，无 FFI）              | Rust + ONNX Runtime                                               | Rust + MNN 绑定（通过 mnn-rs crate）                                      |
| **模型支持**       | 标准 ONNX 格式                                                | 标准 ONNX 格式                                                    | MNN 模型格式                                                              |
| **依赖管理**       | 轻量依赖集（ort + 核心库，无额外下载逻辑）                    | 全功能依赖集（ort 显式启用自动下载 + YAML/JSON/reqwest 等）       | 需要 MNN 库（通过 mnn-rs）                                                |
| **API 与易用性**   | OcrEngine 细粒度控制 + Result 错误处理 + 最小化设置           | 丰富的流水线 API（YAML 配置、自动下载、单词级边框）               | 多层 API（Det/Rec、OcrEngine、C API）                                     |
| **部署与平台**     | ✅ 跨平台（Windows/Linux/macOS），单一二进制文件，cargo build | 跨平台（ONNX Runtime 支持 Windows/Linux/macOS），自动下载         | 跨平台（MNN 支持 Windows/Linux/macOS/Android/iOS），需要 MNN 环境         |
| **性能**           | 通过 ONNX Runtime 优化                                        | 通过 ONNX Runtime 优化                                            | MNN 框架性能                                                              |
| **文档方向分类**   | ✅ PP-LCNet 分类器（0/90/180/270°）                           | ✅ PP-OCR v2.0 分类器（0/180°）                                   | ✅ PP-LCNet 分类器（0/90/180/270° + 0/180°）                              |
| **并发能力**       | ✅ rayon 并行 + 会话池                                        | ✅ rayon 并行 + 批量推理（识别/分类各 6 张）                      | ⚠️ 预处理/后处理使用 rayon，推理部分单线程                                |
| **图像预处理**     | ✅ Rust 原生图像处理                                          | ✅ 纯 Rust 实现（或可选 OpenCV）                                  | ✅ Rust 原生（image + imageproc + ndarray）                               |
| **模型格式支持**   | ✅ 仅支持 ONNX                                                | ✅ ONNX 格式                                                      | ✅ MNN 格式                                                               |
| **API 与错误处理** | ✅ 细粒度控制 + 详细错误消息                                  | ✅ 丰富流水线控制 + thiserror 枚举（14 种变体）                   | ✅ 多层 API + thiserror 枚举（11 种变体）                                 |
| **平台兼容性**     | ✅ 优秀（ONNX Runtime 跨平台，单一二进制）                    | ✅ 优秀（ONNX Runtime 跨平台）                                    | ✅ 良好（MNN 支持多平台）                                                 |
| **生态支持**       | ⚠️ 有限（仅 ONNX）                                            | ⚠️ 有限（ONNX, RapidOCR）                                         | ⚠️ 有限（仅 MNN）                                                         |
| **GPU 加速**       | ❌ 未启用                                                     | ✅ CUDA/DirectML/CANN（通过 ort features）                        | ✅ Metal/OpenCL/OpenGL/Vulkan/CUDA/CoreML（6 后端）                       |
| **外部接口**       | ❌ 仅 Rust API                                                | ✅ YAML 配置 + CLI(rapidocr)                                      | ✅ C API(cdylib) + CLI(newbee-ocr-cli)                                    |
| **输出格式**       | ❌ 仅文本输出                                                 | ✅ JSON + Markdown + 可视化图片                                   | ❌ 仅文本输出                                                             |
| **文本处理增强**   | ✅ 排序模式（水平/垂直/置信度）                               | ✅ 单词级边框 + BiDi 文本                                         | ✅ FP16 推理 + async 异步                                                 |
| **内存/类型安全**  | ✅ 内存安全 Rust + 强类型 + 自动内存管理                      | ✅ 内存安全 Rust + 强类型                                         | ✅ 内存安全 Rust（mnn-rs）+ ⚠️ C API 部分                                 |
| **错误处理与 API** | ✅ Rust Result 类型 + 现代惯用法                              | ✅ Rust Result 类型（thiserror）                                  | ✅ Rust Result 类型（thiserror）                                          |
| **并发安全性**     | ✅ 设计上就是线程安全的                                       | ✅ 线程安全（Arc + Mutex）                                        | ⚠️ 需要小心处理                                                           |
