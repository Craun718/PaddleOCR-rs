# 模型获取

你可以根据 PaddleOCR 的[官方文档](https://www.paddleocr.ai/v2.10.0/infer_deploy/paddle2onnx.html)下载 PaddlePaddle 格式的模型文件自行转换为 ONNX 格式的模型文件，也可以直接下载我们提供的[已经转换好的模型包](https://github.com/Craun718/PaddleOCR-rs/releases)。

## 文件结构

### OCR 模型

已经转换好的模型包内含有三个文件。

```bash
det.onnx # 检测模型
keys.txt # 字典文件
rec.onnx # 识别模型
```

使用时可以自由组合不同版本的识别模型和检测模型，但是识别模型和字典文件的版本**必须**是相同的。

### 方向分类模型

参见[官方说明](https://paddlepaddle.github.io/PaddleX/main/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html)

```bash
PP-LCNet_x1_0_doc_ori.onnx
```

## 鸣谢

感谢 MaaXYZ(https://github.com/MaaXYZ) 提供的 [onnx 模型文件](https://github.com/MaaXYZ/MaaCommonAssets/tree/main/OCR)
感谢 PaddlePaddle 提供的 [模型转换工具](https://www.paddleocr.ai/v2.10.0/infer_deploy/paddle2onnx.html)

## 参考内容

有关 PaddleOCR 的识别流程可以参考[官方文档](https://www.paddleocr.ai/latest/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
