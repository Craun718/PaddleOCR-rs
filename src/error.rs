use thiserror::Error;

/// Unified error type for PaddleOCR-rs operations.
///
/// This enum covers all failure modes across the OCR pipeline: image loading,
/// model loading, inference, preprocessing, CTC decoding, and geometric
/// transformations.
///
/// # Example
///
/// ```ignore
/// use paddleocr_rs_onnx::{OcrEngine, PaddleOcrError};
///
/// match OcrEngine::new(&det, &rec, &keys) {
///     Ok(engine) => { /* use engine */ }
///     Err(PaddleOcrError::Model(e)) => eprintln!("Model load failed: {}", e),
///     Err(e) => eprintln!("Other error: {}", e),
/// }
/// ```
#[derive(Error, Debug)]
pub enum PaddleOcrError {
    /// Image loading or processing error.
    #[error("image error: {message}")]
    Image { message: String },

    /// ONNX model loading or session creation error.
    #[error("model error: {0}")]
    Model(#[from] ort::Error),

    /// ONNX inference execution error.
    #[error("inference failed: {message}")]
    Inference { message: String },

    /// Image preprocessing error (resize, pad, normalize, etc.).
    #[error("preprocessing failed: {message}")]
    Preprocessing { message: String },

    /// CTC or text decoding error.
    #[error("decoding failed: {message}")]
    Decoding { message: String },

    /// UTF-8 conversion error when reading the character dictionary.
    #[error("UTF-8 conversion failed: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// Degenerate text region detected (bbox too small or near-collinear).
    #[error("degenerate text region: {reason}")]
    DegenerateRegion { reason: String },

    /// Perspective projection or warp transformation failed.
    #[error("projection failed: {reason}")]
    Projection { reason: String },

    /// General-purpose error wrapper.
    #[error("general error: {0}")]
    General(String),
}

impl From<image::ImageError> for PaddleOcrError {
    fn from(err: image::ImageError) -> Self {
        PaddleOcrError::Image {
            message: err.to_string(),
        }
    }
}

impl From<Box<dyn std::error::Error>> for PaddleOcrError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        PaddleOcrError::General(err.to_string())
    }
}
