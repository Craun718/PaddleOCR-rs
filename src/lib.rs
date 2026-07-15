use parking_lot::{Mutex, Condvar};
use std::collections::VecDeque;
use rayon::prelude::*;
use log::{info, warn};

mod det;
mod decode;
mod rec;
mod cls;
mod error;
pub use error::PaddleOcrError;

pub use cls::{DocOrientation, OrientationResult, classify_orientation};

use image::DynamicImage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    pub bbox: [[f32; 2]; 4],
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrBlock {
    pub text: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderBy {
    Horizontal,
    Vertical,
    Score,
}

/// Hardware acceleration device for ONNX Runtime inference.
///
/// Controls which execution provider is used for model inference:
/// - `Cpu` — CPU-only inference (default, always available)
/// - `DirectML` — DirectML acceleration (Windows, DirectX 12 GPU)
/// - `Cuda` — CUDA acceleration (NVIDIA GPU)
/// - `OpenVINO` — Intel OpenVINO acceleration (Windows/Linux)
/// - `Nnapi` — Android NNAPI acceleration (Android)
/// - `Coreml` — Apple CoreML acceleration (macOS/iOS)
/// - `Cann` — Huawei CANN / Ascend NPU acceleration (Linux)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccelerationDevice {
    /// CPU-only inference (default, always available)
    Cpu,
    /// DirectML acceleration (Windows, DirectX 12 GPU)
    DirectML,
    /// CUDA acceleration (NVIDIA GPU)
    Cuda,
    /// Intel OpenVINO acceleration (Windows/Linux)
    OpenVINO,
    /// Android NNAPI acceleration (Android)
    Nnapi,
    /// Apple CoreML acceleration (macOS/iOS)
    Coreml,
    /// Huawei CANN / Ascend NPU acceleration (Linux)
    Cann,
}

impl Default for AccelerationDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

impl std::fmt::Display for AccelerationDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::DirectML => write!(f, "DirectML"),
            Self::Cuda => write!(f, "CUDA"),
            Self::OpenVINO => write!(f, "OpenVINO"),
            Self::Nnapi => write!(f, "NNAPI"),
            Self::Coreml => write!(f, "CoreML"),
            Self::Cann => write!(f, "CANN"),
        }
    }
}

impl AccelerationDevice {
    /// Parse a device string (case-insensitive).
    ///
    /// Supported values: "cpu", "cuda"/"nvidia", "directml"/"dml",
    /// "openvino"/"open-vino"/"ov", "nnapi", "coreml"/"apple",
    /// "cann"/"ascend"/"huawei".
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "cuda" | "nvidia" => Some(Self::Cuda),
            "directml" | "dml" => Some(Self::DirectML),
            "openvino" | "open-vino" | "ov" => Some(Self::OpenVINO),
            "nnapi" => Some(Self::Nnapi),
            "coreml" | "apple" => Some(Self::Coreml),
            "cann" | "ascend" | "huawei" => Some(Self::Cann),
            _ => None,
        }
    }
}

/// Configure execution providers on a session builder.
///
/// If the requested EP is not available (not compiled into ORT, missing runtime
/// libraries, or unsupported on this platform), a warning is logged and the
/// session falls back to CPU.
fn configure_session_builder(
    builder: ort::session::builder::SessionBuilder,
    device: AccelerationDevice,
) -> Result<ort::session::builder::SessionBuilder, PaddleOcrError> {
    match device {
        AccelerationDevice::Cpu => Ok(builder),
        AccelerationDevice::DirectML => {
            info!("[ep] configuring DirectML execution provider");
            let ep = ort::ep::DirectML::default().build();
            match builder.clone().with_execution_providers([ep]) {
                Ok(b) => Ok(b),
                Err(e) => {
                    warn!("[ep] DirectML unavailable, falling back to CPU: {}", e);
                    Ok(builder)
                }
            }
        }
        AccelerationDevice::Cuda => {
            info!("[ep] configuring CUDA execution provider");
            let ep = ort::ep::CUDA::default().build();
            match builder.clone().with_execution_providers([ep]) {
                Ok(b) => Ok(b),
                Err(e) => {
                    warn!("[ep] CUDA unavailable, falling back to CPU: {}", e);
                    Ok(builder)
                }
            }
        }
        AccelerationDevice::OpenVINO => {
            info!("[ep] configuring OpenVINO execution provider");
            let ep = ort::ep::OpenVINO::default().build();
            match builder.clone().with_execution_providers([ep]) {
                Ok(b) => Ok(b),
                Err(e) => {
                    warn!("[ep] OpenVINO unavailable, falling back to CPU: {}", e);
                    Ok(builder)
                }
            }
        }
        AccelerationDevice::Nnapi => {
            info!("[ep] configuring NNAPI execution provider");
            let ep = ort::ep::NNAPI::default().build();
            match builder.clone().with_execution_providers([ep]) {
                Ok(b) => Ok(b),
                Err(e) => {
                    warn!("[ep] NNAPI unavailable, falling back to CPU: {}", e);
                    Ok(builder)
                }
            }
        }
        AccelerationDevice::Coreml => {
            info!("[ep] configuring CoreML execution provider");
            let ep = ort::ep::CoreML::default().build();
            match builder.clone().with_execution_providers([ep]) {
                Ok(b) => Ok(b),
                Err(e) => {
                    warn!("[ep] CoreML unavailable, falling back to CPU: {}", e);
                    Ok(builder)
                }
            }
        }
        AccelerationDevice::Cann => {
            info!("[ep] configuring CANN execution provider");
            let ep = ort::ep::CANN::default().build();
            match builder.clone().with_execution_providers([ep]) {
                Ok(b) => Ok(b),
                Err(e) => {
                    warn!("[ep] CANN unavailable, falling back to CPU: {}", e);
                    Ok(builder)
                }
            }
        }
    }
}

pub struct OcrEngine {
    det_session: Mutex<ort::session::Session>,
    rec_sessions: Mutex<VecDeque<ort::session::Session>>,
    rec_sessions_cvar: Condvar,
    det_input: String,
    det_output: String,
    rec_input: String,
    rec_output: String,
    rec_height: u32,
    rec_width: Option<u32>,
    keys: Vec<String>,
    device: AccelerationDevice,
}

impl OcrEngine {
    pub fn new(det_model: &[u8], rec_model: &[u8], keys_data: &[u8]) -> Result<Self, PaddleOcrError> {
        Self::new_with_device(det_model, rec_model, keys_data, AccelerationDevice::default())
    }

    pub fn new_with_device(
        det_model: &[u8],
        rec_model: &[u8],
        keys_data: &[u8],
        device: AccelerationDevice,
    ) -> Result<Self, PaddleOcrError> {
        info!("[ocr] creating engine with device: {}", device);

        let det_session = configure_session_builder(ort::session::Session::builder()?, device)?
            .commit_from_memory(det_model)?;
        let rec_session = configure_session_builder(ort::session::Session::builder()?, device)?
            .commit_from_memory(rec_model)?;

        let det_input = det_session.inputs()[0].name().to_string();
        let det_output = det_session.outputs()[0].name().to_string();
        let rec_input = rec_session.inputs()[0].name().to_string();
        let rec_output = rec_session.outputs()[0].name().to_string();
        let rec_shape = rec_session.inputs()[0]
            .dtype()
            .tensor_shape()
            .cloned()
            .unwrap_or_else(|| vec![1_i64, 3, 48, 320].into());
        let rec_height = rec_shape.get(2).copied().unwrap_or(48).max(1) as u32;
        let rec_width = rec_shape
            .get(3)
            .copied()
            .filter(|dim| *dim > 0)
            .map(|dim| dim as u32);

        let keys_str = std::str::from_utf8(keys_data)?;
        let keys: Vec<String> = keys_str.lines().map(|s| s.to_string()).collect();

        // debug: validate keys vs model output
        let model_classes = rec_session.outputs()[0]
            .dtype().tensor_shape()
            .and_then(|s| s.last())
            .copied()
            .unwrap_or(0) as usize;
        info!(
            "[ocr] keys: {} lines, model output classes: {} (blank + {} chars)",
            keys.len(),
            model_classes,
            model_classes.saturating_sub(1),
        );
        if keys.len() + 1 != model_classes {
            warn!(
                "[ocr] WARNING: keys({}) + 1(blank) = {} != model_classes({})",
                keys.len(),
                keys.len() + 1,
                model_classes,
            );
        }

        // build session pool for concurrent recognition
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let pool_size = (cores / 2).max(1);
        info!("[ocr] creating rec session pool of size {} (cores={})", pool_size, cores);

        let mut sessions = VecDeque::with_capacity(pool_size);
        sessions.push_back(rec_session);
        for _ in 1..pool_size {
            sessions.push_back(
                configure_session_builder(ort::session::Session::builder()?, device)?
                    .commit_from_memory(rec_model)?,
            );
        }

        Ok(Self {
            det_session: Mutex::new(det_session),
            rec_sessions: Mutex::new(sessions),
            rec_sessions_cvar: Condvar::new(),
            det_input,
            det_output,
            rec_input,
            rec_output,
            rec_height,
            rec_width,
            keys,
            device,
        })
    }

    /// Returns the acceleration device used by this engine.
    pub fn device(&self) -> AccelerationDevice {
        self.device
    }

    pub fn detect_text_regions(&self, image: &DynamicImage) -> Result<Vec<TextRegion>, PaddleOcrError> {
        let mut session = self.det_session.lock();
        det::detect_text_regions(&mut session, image, &self.det_input, &self.det_output)
    }

    pub fn recognize_text(&self, image: &DynamicImage, region: &TextRegion) -> Result<decode::DecodedText, PaddleOcrError> {
        let (data, width) = rec::preprocess_region(image, region, self.rec_height, self.rec_width)?;

        // Pop a session from the pool, blocking until one is available
        let mut session = {
            let mut pool = self.rec_sessions.lock();
            loop {
                if let Some(s) = pool.pop_front() {
                    break s;
                }
                self.rec_sessions_cvar.wait(&mut pool);
            }
        };

        let result = rec::run_recognition(&mut session, &data, width, self.rec_height, &self.rec_input, &self.rec_output)?;

        // Always return session to pool and notify waiters
        {
            let mut pool = self.rec_sessions.lock();
            pool.push_back(session);
            self.rec_sessions_cvar.notify_one();
        }

        let probs = result;
        Ok(decode::ctc_decode(&probs, &self.keys))
    }

    /// Complete OCR pipeline: detection + recognition.
    /// `order` controls the output ordering of text blocks.
    pub fn recognize_all(&self, image: &DynamicImage, order: OrderBy) -> Result<Vec<OcrBlock>, PaddleOcrError> {
        let regions = self.detect_text_regions(image)?;

        if regions.is_empty() {
            // Try full-image recognition as fallback
            return self.recognize_full_image(image)
                .map(|b| b.into_iter().collect())
                .map_err(|e| PaddleOcrError::General(e.to_string()));
        }

        let mut blocks: Vec<OcrBlock> = regions
            .par_iter()
            .filter_map(|region| {
                match self.recognize_text(image, region) {
                    Ok(decoded) => {
                        if decoded.text.is_empty() {
                            None
                        } else {
                            let (x, y, width, height) = bbox_to_rect(&region.bbox);
                            Some(OcrBlock {
                                text: decoded.text,
                                confidence: decoded.score,
                                x,
                                y,
                                width,
                                height,
                            })
                        }
                    }
                    Err(PaddleOcrError::DegenerateRegion { .. }) => {
                        // Skip degenerate regions silently
                        None
                    }
                    Err(_) => None,
                }
            })
            .collect();

        match order {
            OrderBy::Horizontal => {
                blocks.sort_by(|a, b| {
                    a.y.partial_cmp(&b.y)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
            OrderBy::Vertical => {
                blocks.sort_by(|a, b| {
                    a.x.partial_cmp(&b.x)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
                });
            }
            OrderBy::Score => {
                blocks.sort_by(|a, b| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        Ok(blocks)
    }

    fn recognize_full_image(&self, image: &DynamicImage) -> Result<Option<OcrBlock>, PaddleOcrError> {
        let width = image.width() as f32;
        let height = image.height() as f32;
        if width < 1.0 || height < 1.0 {
            return Ok(None);
        }

        let full_region = TextRegion {
            bbox: [
                [0.0, 0.0],
                [width - 1.0, 0.0],
                [width - 1.0, height - 1.0],
                [0.0, height - 1.0],
            ],
            confidence: 0.0,
        };
        let decoded = self.recognize_text(image, &full_region)?;
        if decoded.text.is_empty() {
            return Ok(None);
        }

        Ok(Some(OcrBlock {
            text: decoded.text,
            confidence: decoded.score,
            x: 0.0,
            y: 0.0,
            width,
            height,
        }))
    }
}

fn bbox_to_rect(bbox: &[[f32; 2]; 4]) -> (f32, f32, f32, f32) {
    let min_x = bbox.iter().map(|p| p[0]).reduce(f32::min).unwrap_or(0.0);
    let min_y = bbox.iter().map(|p| p[1]).reduce(f32::min).unwrap_or(0.0);
    let max_x = bbox.iter().map(|p| p[0]).reduce(f32::max).unwrap_or(0.0);
    let max_y = bbox.iter().map(|p| p[1]).reduce(f32::max).unwrap_or(0.0);
    (min_x, min_y, max_x - min_x, max_y - min_y)
}

/// Document orientation classifier using PP-LCNet_x1_0_doc_ori model.
///
/// This classifier detects the orientation of document images (0°, 90°, 180°, 270°).
/// Useful for preprocessing documents before OCR when the scan/capture orientation
/// is unknown.
///
/// # Example
///
/// ```ignore
/// let model_data = std::fs::read("PP-LCNet_x1_0_doc_ori.onnx")?;
/// let classifier = DocOrientationClassifier::new(&model_data)?;
/// let result = classifier.classify(&image)?;
/// println!("Orientation: {}°, confidence: {}", result.orientation.angle(), result.confidence);
/// ```
pub struct DocOrientationClassifier {
    session: Mutex<ort::session::Session>,
    input_name: String,
    output_name: String,
    device: AccelerationDevice,
}

impl DocOrientationClassifier {
    /// Create a new orientation classifier from ONNX model data.
    ///
    /// # Arguments
    /// * `model_data` - Raw ONNX model bytes (PP-LCNet_x1_0_doc_ori.onnx)
    ///
    /// # Returns
    /// A new classifier instance ready to classify images.
    pub fn new(model_data: &[u8]) -> Result<Self, PaddleOcrError> {
        Self::new_with_device(model_data, AccelerationDevice::default())
    }

    /// Create a new orientation classifier with a specific acceleration device.
    pub fn new_with_device(model_data: &[u8], device: AccelerationDevice) -> Result<Self, PaddleOcrError> {
        info!("[doc_ori] creating classifier with device: {}", device);

        let session = configure_session_builder(ort::session::Session::builder()?, device)?
            .commit_from_memory(model_data)?;

        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        // Log model info
        let input_shape = session.inputs()[0]
            .dtype()
            .tensor_shape()
            .cloned()
            .unwrap_or_default();
        let output_shape = session.outputs()[0]
            .dtype()
            .tensor_shape()
            .cloned()
            .unwrap_or_default();
        info!(
            "[doc_ori] model loaded: input {:?}, output {:?}",
            input_shape, output_shape
        );

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
            device,
        })
    }

    /// Returns the acceleration device used by this classifier.
    pub fn device(&self) -> AccelerationDevice {
        self.device
    }

    /// Classify the orientation of a document image.
    ///
    /// # Arguments
    /// * `image` - The document image to classify
    ///
    /// # Returns
    /// An `OrientationResult` containing the detected orientation and confidence score.
    pub fn classify(&self, image: &DynamicImage) -> Result<OrientationResult, PaddleOcrError> {
        let mut session = self.session.lock();
        cls::classify_orientation(&mut session, image, &self.input_name, &self.output_name)
    }

    /// Rotate the image to correct its orientation.
    ///
    /// This is a convenience method that classifies the orientation and
    /// returns a correctly oriented image.
    ///
    /// # Arguments
    /// * `image` - The document image to correct
    ///
    /// # Returns
    /// A tuple of (corrected_image, orientation_result).
    pub fn correct_orientation(&self, image: &DynamicImage) -> Result<(DynamicImage, OrientationResult), PaddleOcrError> {
        let result = self.classify(image)?;

        let corrected = match result.orientation {
            DocOrientation::Upright => image.clone(),
            DocOrientation::Rotate90 => {
                // Rotate 90° counter-clockwise to correct
                image.rotate270()
            }
            DocOrientation::Rotate180 => {
                // Rotate 180° to correct
                image.rotate180()
            }
            DocOrientation::Rotate270 => {
                // Rotate 270° counter-clockwise (90° clockwise) to correct
                image.rotate90()
            }
        };

        Ok((corrected, result))
    }
}

