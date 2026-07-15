# PaddleOCR-rs

ONNX-based OCR engine using PaddleOCR models, written in Rust.

## Modules

- `lib.rs` — `OcrEngine` (detection + recognition) + `DocOrientationClassifier`
- `det.rs` — Text detection via DBNet
- `rec.rs` — Text recognition via CRNN
- `decode.rs` — CTC greedy decoding
- `cls.rs` — Document orientation classification (PP-LCNet)

## Usage

```rust
use paddleocr_rs_onnx::{OcrEngine, DocOrientationClassifier, OrderBy};

let engine = OcrEngine::new(&det_model_bytes, &rec_model_bytes, &keys_bytes)?;
let blocks = engine.recognize_all(&image, OrderBy::Horizontal)?; // Vec<OcrBlock>

let classifier = DocOrientationClassifier::new(&model_bytes)?;
let (corrected, result) = classifier.correct_orientation(&image)?;
```

## License

MIT
