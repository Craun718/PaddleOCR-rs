# PaddleOCR-rs

[English](README.md) | [中文](README_CN.md)

基于 ONNX 的 OCR 引擎，使用 PaddleOCR 模型，使用 Rust 编写。

## 特性

- **文本检测与识别** — 完整的 OCR 流程，包含 DBNet 检测和 CRNN 识别
- **文档方向分类** — PP-LCNet 分类器，支持 0°/90°/180°/270° 旋转检测和自动校正
- **硬件加速** — ONNX Runtime，支持 CPU、DirectML、CUDA、OpenVINO、NNAPI、CoreML、CANN
- **跨平台** — Windows、Linux、macOS、Android（FFI）、iOS（FFI）
- **并发处理** — rayon 并行执行 + 会话池
- **细粒度控制** — 分步 API：检测 → 识别，或使用排序模式的完整流程
- **最小依赖** — 仅需 ONNX 模型，无需外部运行时库

## 快速开始

```rust
use paddleocr_rs_onnx::{OcrEngine, OrderBy};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

let engine = OcrEngine::new(&det_model, &rec_model, &keys)?;

let image = image::open("test.png")?;
let blocks = engine.recognize_all(&image, OrderBy::Horizontal)?;
for block in &blocks {
    println!("{} ({:.2}%)", block.text, block.confidence * 100.0);
}
```

## 安装

```toml
[dependencies]
paddleocr_rs_onnx = "0.2"
```

## API 文档

- [API 概览](docs/zh/api-overview.md) — 模块、结构体、返回类型
- [OcrEngine](docs/zh/ocr-engine.md) — 检测和识别 API
- [DocOrientationClassifier](docs/zh/doc-orientation.md) — 方向检测和校正
- [硬件加速](docs/zh/acceleration.md) — AccelerationDevice、平台要求、EP feature
- [FFI API](src/ffi.rs) — C 兼容 API

## 移动端平台支持

PaddleOCR-rs 通过 C FFI 接口支持 Android 和 iOS 平台。

### Android

- **CPU**: ✅ 支持（通过 ONNX Runtime）
- **NNAPI**: ✅ 支持（通过 nnapi feature）
- **Targets**: aarch64-linux-android、armv7-linux-androideabi（需要自行编译）、x86_64-linux-android（需要自行编译）

### iOS

- **CPU**: ✅ 支持（通过 ONNX Runtime）
- **CoreML**: ✅ 支持（通过 coreml feature）
- **Targets**: aarch64-apple-ios、aarch64-apple-ios-sim

### 构建移动端

```bash
# Android
./build-android.sh aarch64-linux-android --release

# iOS
./build-ios.sh aarch64-apple-ios --release
```

### 示例项目

- [xc-ocr-onnx](https://github.com/Craun718/xc-ocr-onnx) — Tauri 2 桌面 GUI 应用，支持图片 / DOCX / PDF OCR，可动态切换模型
- `examples/android-demo/` — Android Kotlin 示例
- `examples/ios-demo/` — iOS Swift 示例

## 与其他 Rust PaddleOCR 实现的对比

本项目是 PaddleOCR 的多个 Rust 实现之一。以下是三个主要实现的全面对比：

### 硬件加速支持对比

| 平台/后端         | 本项目 (PaddleOCR-rs) | [paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs) | [rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr) |
| ----------------- | --------------------- | --------------------------------------------------------- | --------------------------------------------------------------- |
| **Windows**       |                       |                                                           |                                                                 |
| CUDA              | ✅                    | ✅                                                        | ✅                                                              |
| DirectML          | ✅                    | ✅                                                        | ❌                                                              |
| OpenVINO          | ✅                    | ❌                                                        | ❌                                                              |
| **Linux**         |                       |                                                           |                                                                 |
| CUDA              | ✅                    | ✅                                                        | ✅                                                              |
| CANN              | ✅                    | ✅                                                        | ❌                                                              |
| OpenVINO          | ✅                    | ❌                                                        | ❌                                                              |
| **macOS**         |                       |                                                           |                                                                 |
| CoreML            | ✅                    | ❌                                                        | ✅ CoreML                                                       |
| Metal (Apple GPU) | ❌                    | ❌                                                        | ✅ Metal                                                        |
| **其他**          |                       |                                                           |                                                                 |
| OpenGL            | ❌                    | ❌                                                        | ✅ OpenGL                                                       |

### 总览

| 方面              | 本项目 (PaddleOCR-rs)                                                        | [paddle-ocr-rs](https://github.com/mg-chao/paddle-ocr-rs)         | [rust-paddle-ocr](https://github.com/zibo-chen/rust-paddle-ocr)   |
| ----------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| **推理后端**      | ONNX Runtime via ort crate（纯 Rust，无 FFI）                                | ONNX Runtime via ort crate                                        | MNN 推理框架 via mnn-rs crate                                     |
| **文档方向分类**  | ✅ 包含（PP-LCNet 模型，自动旋转校正）                                       | ✅ 包含（PP-OCR mobile v2.0，180° 旋转）                          | ✅ 包含（PP-LCNet 模型，0/90/180/270° + 0/180°）                  |
| **模型格式**      | 标准 ONNX 格式                                                               | 标准 ONNX 格式                                                    | MNN 模型格式                                                      |
| **架构**          | 纯 Rust + ONNX Runtime                                                       | Rust + ONNX Runtime                                               | Rust + MNN 绑定                                                   |
| **外部依赖**      | 轻量依赖集（ort + 核心库，无额外下载逻辑）                                   | 全功能依赖集（ort 显式启用自动下载 + YAML/JSON/reqwest 等）       | 需要 MNN 库（通过 mnn-rs）                                        |
| **API 设计**      | OcrEngine 细粒度控制 + Result 错误处理 + 最小化设置                          | 丰富的流水线 API（YAML 配置、自动下载、单词级边框）               | 多层 API（Det/Rec、OcrEngine、C API）                             |
| **部署与平台**    | ✅ 跨平台（Windows/Linux/macOS/Android/iOS），FFI 支持移动端，单一二进制文件 | 跨平台（ONNX Runtime 支持 Windows/Linux/macOS），自动下载外部依赖 | 跨平台（MNN 支持 Windows/Linux/macOS/Android/iOS），需要 MNN 环境 |
| **文档方向分类**  | ✅ PP-LCNet 分类器（0/90/180/270°）                                          | ✅ PP-OCR v2.0 分类器（0/180°）                                   | ✅ PP-LCNet 分类器（0/90/180/270° + 0/180°）                      |
| **并发能力**      | ✅ rayon 并行 + 会话池                                                       | ✅ rayon 并行 + 批量推理（识别/分类各 6 张）                      | ⚠️ 预处理/后处理使用 rayon，推理部分单线程                        |
| **模型格式支持**  | ONNX 格式                                                                    | ONNX 格式                                                         | MNN 格式                                                          |
| **外部接口**      | ✅ Rust API + C FFI API（通过 `ffi` feature）                                | ✅ YAML 配置 + CLI(rapidocr)                                      | ✅ C API(cdylib) + CLI(newbee-ocr-cli)                            |
| **内存/类型安全** | ✅ 内存安全 Rust + 强类型                                                    | ✅ 内存安全 Rust + 强类型                                         | ✅ 内存安全 Rust（mnn-rs）+ ⚠️ C API 部分                         |
| **并发安全性**    | ✅ 线程安全                                                                  | ✅ 线程安全                                                       | ⚠️ 需要小心处理                                                   |

## 致谢

本项目基于以下项目的工作：

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 提供模型
- [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) - 提供运行时参考
- [MAAFramework](https://github.com/MaaAssistantArknights/MAAFramework) - 提供架构参考

## 许可证

MIT
