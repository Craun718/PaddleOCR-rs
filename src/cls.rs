//! Document orientation classification using PP-LCNet_x1_0_doc_ori model.
//!
//! Classifies the orientation of document images into four classes:
//! 0 (upright), 90, 180, and 270 degrees. Useful for preprocessing
//! scanned documents or photos where the capture orientation is unknown.
//!
//! The preprocessing pipeline is:
//! 1. Resize so the short edge equals 256 pixels
//! 2. Center crop to 224x224
//! 3. Normalize with ImageNet mean/std
//! 4. Convert to CHW format

use crate::error::PaddleOcrError;

// Document orientation classification using PP-LCNet_x1_0_doc_ori ONNX model.
//
// This module provides functionality to classify the orientation of document images.
// The model outputs 4 classes representing rotation angles: 0°, 90°, 180°, 270°.

use image::{DynamicImage, RgbImage};
use ort::session::Session;

/// Document orientation angles classified by the model.
///
/// The model outputs one of four rotation classes representing the
/// original orientation of the document before any rotation was applied.
///
/// # Example
///
/// ```ignore
/// let orientation = DocOrientation::from_class_index(2);
/// assert_eq!(orientation, DocOrientation::Rotate180);
/// assert_eq!(orientation.angle(), 180);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocOrientation {
    /// Document is upright (0° rotation needed)
    Upright,
    /// Document needs 90° clockwise rotation
    Rotate90,
    /// Document needs 180° rotation
    Rotate180,
    /// Document needs 270° clockwise rotation (or 90° counter-clockwise)
    Rotate270,
}

impl DocOrientation {
    /// Get the rotation angle in degrees.
    ///
    /// Returns 0, 90, 180, or 270.
    pub fn angle(&self) -> u32 {
        match self {
            DocOrientation::Upright => 0,
            DocOrientation::Rotate90 => 90,
            DocOrientation::Rotate180 => 180,
            DocOrientation::Rotate270 => 270,
        }
    }

    /// Convert from class index (0-3).
    fn from_class_index(idx: usize) -> Self {
        match idx {
            0 => DocOrientation::Upright,
            1 => DocOrientation::Rotate90,
            2 => DocOrientation::Rotate180,
            3 => DocOrientation::Rotate270,
            _ => DocOrientation::Upright, // Default fallback
        }
    }
}

/// Result of document orientation classification.
///
/// Contains the detected orientation and the classifier's confidence score.
///
/// # Example
///
/// ```ignore
/// let result = classifier.classify(&image)?;
/// println!("Orientation: {:?}, confidence: {:.2}", result.orientation, result.confidence);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct OrientationResult {
    /// The detected orientation.
    pub orientation: DocOrientation,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

/// Default preprocessing parameters for PP-LCNet_x1_0_doc_ori model.
/// These match the PaddleOCR configuration.
const RESIZE_SHORT: u32 = 256;
const CROP_SIZE: u32 = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Preprocess image for orientation classification.
///
/// Steps:
/// 1. Resize so that the short edge equals RESIZE_SHORT (256)
/// 2. Center crop to CROP_SIZE x CROP_SIZE (224x224)
/// 3. Normalize with ImageNet mean/std
/// 4. Convert to CHW format
fn preprocess(image: &DynamicImage) -> Result<Vec<f32>, PaddleOcrError> {
    let (orig_w, orig_h) = (image.width(), image.height());
    if orig_w == 0 || orig_h == 0 {
        return Err(PaddleOcrError::Image { message: "Image has zero dimensions".to_string() });
    }

    // Step 1: Resize by short edge
    let short_edge = orig_w.min(orig_h);
    let scale = RESIZE_SHORT as f32 / short_edge as f32;
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let resized = image.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);

    // Step 2: Center crop
    let crop_x = ((new_w - CROP_SIZE) / 2).max(0) as u32;
    let crop_y = ((new_h - CROP_SIZE) / 2).max(0) as u32;

    // Ensure we have valid crop dimensions
    let actual_crop_w = CROP_SIZE.min(new_w);
    let actual_crop_h = CROP_SIZE.min(new_h);

    // Create a centered crop, padding if necessary
    let mut cropped = RgbImage::new(CROP_SIZE, CROP_SIZE);
    let rgb = resized.to_rgb8();

    // Copy the cropped region, handling edge cases
    for y in 0..actual_crop_h {
        for x in 0..actual_crop_w {
            let src_x = crop_x + x;
            let src_y = crop_y + y;
            if src_x < new_w && src_y < new_h {
                let pixel = rgb.get_pixel(src_x, src_y);
                cropped.put_pixel(x, y, *pixel);
            }
        }
    }

    // Step 3 & 4: Normalize and convert to CHW format
    let capacity: usize = 3 * CROP_SIZE as usize * CROP_SIZE as usize;
    let mut data = Vec::with_capacity(capacity);
    for c in 0..3 {
        for y in 0..CROP_SIZE {
            for x in 0..CROP_SIZE {
                let pixel = cropped.get_pixel(x, y);
                let val = pixel[c] as f32 / 255.0;
                data.push((val - MEAN[c]) / STD[c]);
            }
        }
    }

    Ok(data)
}

/// Apply softmax to logits and get the predicted class.
///
/// # Returns
///
/// A tuple of `(predicted_class_index, confidence_score)`.

fn softmax_argmax(logits: &[f32]) -> (usize, f32) {
    // Compute softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp() / exp_sum).collect();

    // Find argmax
    let (idx, prob) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap_or((0, &0.0_f32));

    (idx, *prob)
}

/// Run orientation classification on an image.
///
/// # Arguments
///
/// * `session` — ONNX runtime session for the orientation model.
/// * `image` — Input image to classify.
/// * `input_name` — Name of the input tensor in the ONNX model.
/// * `output_name` — Name of the output tensor in the ONNX model.
///
/// # Returns
///
/// An [`OrientationResult`] containing the detected orientation and confidence.
///
/// # Errors
///
/// Returns [`PaddleOcrError::Preprocessing`] if the image has zero dimensions,
/// or [`PaddleOcrError::Inference`] if the model execution fails.
///
/// # Example
///
/// ```ignore
/// let mut session = /* loaded ONNX session */;
/// let image = image::open("scan.png")?;
/// let result = classify_orientation(&mut session, &image, "input", "output")?;
/// println!("Detected: {:?} ({:.0}%)", result.orientation, result.confidence * 100.0);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn classify_orientation(
    session: &mut Session,
    image: &DynamicImage,
    input_name: &str,
    output_name: &str,
) -> Result<OrientationResult, PaddleOcrError> {
    let data = preprocess(image)?;

    // Create input tensor: [1, 3, 224, 224]
    let input_tensor = ort::value::Tensor::from_array((
        [1i64, 3, CROP_SIZE as i64, CROP_SIZE as i64],
        data,
    ))?;

    // Run inference
    let outputs = session.run(ort::inputs![input_name => input_tensor])?;

    // Extract output logits
    let (_, output_slice) = outputs[output_name].try_extract_tensor::<f32>()?;

    // The output shape is [1, 4] for 4 orientation classes
    let logits: &[f32] = output_slice;

    // Apply softmax and get prediction
    let (class_idx, confidence) = softmax_argmax(logits);
    let orientation = DocOrientation::from_class_index(class_idx);

    Ok(OrientationResult {
        orientation,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orientation_angle() {
        assert_eq!(DocOrientation::Upright.angle(), 0);
        assert_eq!(DocOrientation::Rotate90.angle(), 90);
        assert_eq!(DocOrientation::Rotate180.angle(), 180);
        assert_eq!(DocOrientation::Rotate270.angle(), 270);
    }

    #[test]
    fn test_from_class_index() {
        assert_eq!(DocOrientation::from_class_index(0), DocOrientation::Upright);
        assert_eq!(DocOrientation::from_class_index(1), DocOrientation::Rotate90);
        assert_eq!(DocOrientation::from_class_index(2), DocOrientation::Rotate180);
        assert_eq!(DocOrientation::from_class_index(3), DocOrientation::Rotate270);
    }

    #[test]
    fn test_softmax_argmax() {
        // Test with logits where class 2 should have highest probability
        let logits = [0.5, 1.0, 2.0, 0.3];
        let (idx, prob) = softmax_argmax(&logits);
        assert_eq!(idx, 2);
        assert!(prob > 0.5); // Class 2 should have highest confidence
    }
}



