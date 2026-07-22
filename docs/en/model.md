# Model Acquisition

You can follow PaddleOCR's [official documentation](https://www.paddleocr.ai/v2.10.0/infer_deploy/paddle2onnx.html) to download PaddlePaddle-format model files and convert them to ONNX yourself, or you can directly download the [pre-converted model packages](https://github.com/Craun718/PaddleOCR-rs/releases) provided by us.

## File Structure

### OCR Models

Each pre-converted model package contains three files.

```bash
det.onnx # Detection model
keys.txt # Dictionary file
rec.onnx # Recognition model
```

You can freely combine recognition and detection models from different versions when using them, but the recognition model and dictionary file versions **must** match.

### Orientation Classification Model

See the [official documentation](https://paddlepaddle.github.io/PaddleX/main/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html).

```bash
PP-LCNet_x1_0_doc_ori.onnx
```

## Acknowledgements

Thanks to MaaXYZ (https://github.com/MaaXYZ) for providing the [ONNX model files](https://github.com/MaaXYZ/MaaCommonAssets/tree/main/OCR).

Thanks to PaddlePaddle for providing the [model conversion tool](https://www.paddleocr.ai/v2.10.0/infer_deploy/paddle2onnx.html).

## References

For details about PaddleOCR's recognition workflow, see the [official documentation](https://www.paddleocr.ai/latest/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html).
