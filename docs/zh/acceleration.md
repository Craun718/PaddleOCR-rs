# 硬件加速

[← 返回 API 概览](api-overview.md)

PaddleOCR-rs 通过 ONNX Runtime 执行提供程序（EP）支持硬件加速。

## AccelerationDevice 枚举

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

## 平台要求

| 设备 | 平台 | 要求 |
|------|------|------|
| **Cpu** | 所有平台 | 无额外要求（始终可用） |
| **DirectML** | Windows 10/11 | 支持 DirectX 12 的 GPU |
| **CUDA** | Windows/Linux | 安装了 CUDA 工具包的 NVIDIA GPU |
| **OpenVINO** | Windows/Linux | 安装了 OpenVINO 工具包的 Intel CPU/GPU |
| **NNAPI** | Android | 支持 NNAPI 的 Android 设备 |
| **CoreML** | macOS/iOS | macOS 10.13+ 或 iOS 11+ 设备 |
| **CANN** | Linux | 安装了 CANN 工具包的华为昇腾 NPU |

## EP 可用性与 Feature Flag

| EP | 状态 | 所需 Feature |
|----|------|-------------|
| ✅ **CPU** | 始终可用 | 无需 |
| ✅ **DirectML** | 默认 Windows 构建中包含 | 无需 |
| ❌ **CUDA** | 需要 feature flag | `cuda` feature + CUDA 工具包 |
| ❌ **OpenVINO** | 需要 feature flag | `openvino` feature + 自定义 ORT 构建 |
| ❌ **NNAPI** | 需要 feature flag | `nnapi` feature + 自定义 ORT 构建 |
| ❌ **CoreML** | 需要 feature flag | `coreml` feature + 自定义 ORT 构建 |
| ❌ **CANN** | 需要 feature flag | `cann` feature + 自定义 ORT 构建 |

## 通过 Cargo Feature 启用 EP

```toml
[dependencies]
paddleocr_rs_onnx = { version = "0.1", features = ["cuda"] }

# 多个 EP（回退顺序：第一个启用的 EP 优先）
paddleocr_rs_onnx = { version = "0.1", features = ["cuda", "directml"] }
```

## 运行时行为

如果请求的 EP 在运行时不可用（未编译到 ORT 二进制文件中、缺少运行时库或不支持当前平台），会记录警告日志并自动回退到 CPU 推理。

## 使用示例

```rust
use paddleocr_rs_onnx::{OcrEngine, AccelerationDevice};

let det_model = std::fs::read("ch_PP-OCRv4_det_infer.onnx")?;
let rec_model = std::fs::read("ch_PP-OCRv4_rec_infer.onnx")?;
let keys = std::fs::read("ppocr_keys_v1.txt")?;

// 引擎会尝试使用指定设备，如果不可用则回退到 CPU
let engine = OcrEngine::new_with_device(
    &det_model, &rec_model, &keys,
    AccelerationDevice::DirectML,
)?;

// 检查实际使用的设备
println!("使用设备: {:?}", engine.device());
```
