use parking_lot::{Mutex, Condvar};
use std::collections::VecDeque;
use rayon::prelude::*;
use log::{info, warn};

mod det;
mod decode;
mod rec;
mod cls;

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
}

impl OcrEngine {
    pub fn new(det_model: &[u8], rec_model: &[u8], keys_data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let det_session = ort::session::Session::builder()?
            .commit_from_memory(det_model)?;
        let rec_session = ort::session::Session::builder()?
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
                ort::session::Session::builder()?.commit_from_memory(rec_model)?,
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
        })
    }

    pub fn detect_text_regions(&self, image: &DynamicImage) -> Result<Vec<TextRegion>, String> {
        let mut session = self.det_session.lock();
        det::detect_text_regions(&mut session, image, &self.det_input, &self.det_output)
            .map_err(|e| e.to_string())
    }

    pub fn recognize_text(&self, image: &DynamicImage, region: &TextRegion) -> Result<decode::DecodedText, String> {
        let (data, width) = rec::preprocess_region(image, region, self.rec_height, self.rec_width)
            .map_err(|e| e.to_string())?;

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

        let result = rec::run_recognition(&mut session, &data, width, self.rec_height, &self.rec_input, &self.rec_output)
            .map_err(|e| e.to_string());

        // Always return session to pool and notify waiters
        {
            let mut pool = self.rec_sessions.lock();
            pool.push_back(session);
            self.rec_sessions_cvar.notify_one();
        }

        let probs = result?;
        Ok(decode::ctc_decode(&probs, &self.keys))
    }

    pub fn recognize_all(&self, image: &DynamicImage, order_by: OrderBy) -> Result<Vec<OcrBlock>, String> {
        let regions = self.detect_text_regions(image)?;
        if regions.is_empty() {
            if let Some(block) = self.recognize_full_image(image)? {
                return Ok(vec![block]);
            }
            return Ok(Vec::new());
        }

        let mut blocks: Vec<OcrBlock> = regions
            .par_iter()
            .filter_map(|region| {
                let decoded = match self.recognize_text(image, region) {
                    Ok(d) => d,
                    Err(e) => {
                        warn!("[ocr] skipping region {:?}: {}", region.bbox, e);
                        return None;
                    }
                };
                let (x, y, w, h) = bbox_to_rect(&region.bbox);
                Some(OcrBlock {
                    text: decoded.text,
                    confidence: decoded.score,
                    x,
                    y,
                    width: w,
                    height: h,
                })
            })
            .collect();

        match order_by {
            OrderBy::Horizontal => blocks.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y))),
            OrderBy::Vertical => blocks.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x))),
            OrderBy::Score => blocks.sort_by(|a, b| b.confidence.total_cmp(&a.confidence)),
        }
        Ok(blocks)
    }

    fn recognize_full_image(&self, image: &DynamicImage) -> Result<Option<OcrBlock>, String> {
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
}

impl DocOrientationClassifier {
    /// Create a new orientation classifier from ONNX model data.
    ///
    /// # Arguments
    /// * `model_data` - Raw ONNX model bytes (PP-LCNet_x1_0_doc_ori.onnx)
    ///
    /// # Returns
    /// A new classifier instance ready to classify images.
    pub fn new(model_data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let session = ort::session::Session::builder()?
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
        })
    }

    /// Classify the orientation of a document image.
    ///
    /// # Arguments
    /// * `image` - The document image to classify
    ///
    /// # Returns
    /// An `OrientationResult` containing the detected orientation and confidence score.
    pub fn classify(&self, image: &DynamicImage) -> Result<OrientationResult, String> {
        let mut session = self.session.lock();
        cls::classify_orientation(&mut session, image, &self.input_name, &self.output_name)
            .map_err(|e| e.to_string())
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
    pub fn correct_orientation(&self, image: &DynamicImage) -> Result<(DynamicImage, OrientationResult), String> {
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