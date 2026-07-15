use thiserror::Error;

/// PaddleOCR-rs统一错误类型
#[derive(Error, Debug)]
pub enum PaddleOcrError {
    /// 图像处理错误
    #[error("image error: {message}")]
    Image { message: String },

    /// 模型加载错误
    #[error("model error: {0}")]
    Model(#[from] ort::Error),

    /// 推理错误
    #[error("inference failed: {message}")]
    Inference { message: String },

    /// 预处理错误
    #[error("preprocessing failed: {message}")]
    Preprocessing { message: String },

    /// 解码错误
    #[error("decoding failed: {message}")]
    Decoding { message: String },

    /// UTF-8转换错误
    #[error("UTF-8 conversion failed: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    /// 退化文本区域
    #[error("degenerate text region: {reason}")]
    DegenerateRegion { reason: String },

    /// 投影变换失败
    #[error("projection failed: {reason}")]
    Projection { reason: String },

    /// 泛型错误包装
    #[error("general error: {0}")]
    General(String),
}

// 为常见错误类型实现From trait
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
