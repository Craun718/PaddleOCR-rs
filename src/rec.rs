//! Text recognition module using SVTR/CRNN.
//!
//! Recognizes text within detected regions by:
//!
//! 1. **Preprocess** — Perspective rectification, resize to model input dimensions
//! 2. **Inference** — Run the recognition model to produce per-timestep class probabilities
//! 3. (Decoding is handled by [`crate::decode::ctc_decode`])
//!
//! The module also provides geometric utilities for quadrilateral validation
//! and perspective transformation.

use crate::error::PaddleOcrError;

use std::panic::{catch_unwind, AssertUnwindSafe};

use image::{DynamicImage, RgbImage};
use imageproc::geometric_transformations::{warp_into, Interpolation, Border, Projection};
use log::warn;
use ort::session::Session;
use crate::TextRegion;

fn point_dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// 2D cross product of vectors (a→b) × (a→c).
fn cross2d(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Signed area of a quadrilateral using the shoelace formula.
///
/// Negative area indicates clockwise (self-intersecting / concave) ordering.
/// Absolute value gives the actual area.
fn quad_area(pts: &[[f32; 2]; 4]) -> f32 {
    let mut area = 0.0f32;
    for i in 0..4 {
        let j = (i + 1) % 4;
        area += pts[i][0] * pts[j][1];
        area -= pts[j][0] * pts[i][1];
    }
    area * 0.5
}

/// Minimum absolute cross product of diagonals against edges —
/// detects near-degenerate quads (collinear / near-zero area).
const MIN_QUAD_AREA: f32 = 10.0;
/// Maximum absolute cross-product ratio (min/max) to reject
/// extremely thin quads that still have "area" by shoelace.
const THIN_QUAD_RATIO: f32 = 0.01;

/// Validates that a quad is usable for perspective projection:
/// - All points are finite
/// - Has sufficient area (not degenerate)
/// - Not excessively thin (avoids ill-conditioned homography)
fn is_valid_quad(bbox: &[[f32; 2]; 4]) -> bool {
    // 1. All coordinates finite
    for pt in bbox {
        if !pt[0].is_finite() || !pt[1].is_finite() {
            return false;
        }
    }

    // 2. Signed area via shoelace — must be positive (counter-clockwise) and large enough
    let area = quad_area(bbox);
    if area.abs() < MIN_QUAD_AREA {
        return false;
    }

    // 3. Check diagonal cross products to reject extremely thin quads
    //    cross products of each edge against the diagonal through that vertex
    let cross_products: [f32; 4] = [
        cross2d(bbox[0], bbox[1], bbox[3]).abs(),
        cross2d(bbox[1], bbox[2], bbox[0]).abs(),
        cross2d(bbox[2], bbox[3], bbox[1]).abs(),
        cross2d(bbox[3], bbox[0], bbox[2]).abs(),
    ];
    let max_cross = cross_products.iter().copied().fold(0.0f32, f32::max);
    let min_cross = cross_products.iter().copied().fold(f32::INFINITY, f32::min);
    if max_cross > 0.0 && min_cross / max_cross < THIN_QUAD_RATIO {
        return false;
    }

    true
}

/// Order bounding box points as top-left, top-right, bottom-right, bottom-left.
///
/// Uses sum/difference heuristics to identify corners.
fn order_bbox_points(bbox: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    let mut rect = [[0.0; 2]; 4];

    let sums = bbox.map(|p| p[0] + p[1]);
    let diffs = bbox.map(|p| p[0] - p[1]);

    let top_left = sums
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let bottom_right = sums
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(2);

    let remaining: Vec<usize> = (0..4)
        .filter(|idx| *idx != top_left && *idx != bottom_right)
        .collect();

    let top_right = remaining
        .iter()
        .copied()
        .max_by(|a, b| diffs[*a].partial_cmp(&diffs[*b]).unwrap())
        .unwrap_or(1);
    let bottom_left = remaining
        .iter()
        .copied()
        .min_by(|a, b| diffs[*a].partial_cmp(&diffs[*b]).unwrap())
        .unwrap_or(3);

    rect[0] = bbox[top_left];
    rect[1] = bbox[top_right];
    rect[2] = bbox[bottom_right];
    rect[3] = bbox[bottom_left];
    rect
}

/// Preprocess a text region for recognition.
///
/// Performs perspective rectification on the quadrilateral bounding box,
/// resizes the result to the model's expected input dimensions, and
/// normalizes pixel values.
///
/// # Arguments
///
/// * `image` — The original image.
/// * `region` — The [`TextRegion`] to preprocess.
/// * `rec_height` — Target height for the recognition model input.
/// * `rec_width` — Optional fixed width; if `None`, width is computed from
///   the aspect ratio of the rectified region.
///
/// # Returns
///
/// A tuple of `(flattened_chw_data, input_width)` where `flattened_chw_data`
/// is the normalized CHW-format pixel data and `input_width` is the actual
/// width used for the model input tensor.
///
/// # Errors
///
/// Returns [`PaddleOcrError::DegenerateRegion`] if the bounding box is invalid,
/// or [`PaddleOcrError::Projection`] if the perspective transformation fails.
pub fn preprocess_region(
    image: &DynamicImage,
    region: &TextRegion,
    rec_height: u32,
    rec_width: Option<u32>,
) -> Result<(Vec<f32>, i64), PaddleOcrError> {
    let bbox = order_bbox_points(region.bbox);

    // Reject degenerate quads that would produce an ill-conditioned homography
    if !is_valid_quad(&bbox) {
        warn!(
            "[rec] skipping degenerate bbox: {:?} (area too small or near-collinear)",
            region.bbox
        );
        return Err(PaddleOcrError::DegenerateRegion { reason: "degenerate text region bbox".to_string() });
    }

    let rw = (point_dist(bbox[0], bbox[1]).max(point_dist(bbox[3], bbox[2])).ceil() as u32).max(1);
    let rh = (point_dist(bbox[1], bbox[2]).max(point_dist(bbox[0], bbox[3])).ceil() as u32).max(1);

    let src = [
        (bbox[0][0], bbox[0][1]),
        (bbox[1][0], bbox[1][1]),
        (bbox[2][0], bbox[2][1]),
        (bbox[3][0], bbox[3][1]),
    ];
    let dst = [
        (0.0f32, 0.0f32),
        (rw as f32, 0.0f32),
        (rw as f32, rh as f32),
        (0.0f32, rh as f32),
    ];

    let proj = Projection::from_control_points(src, dst)
        .ok_or(PaddleOcrError::Projection { reason: "Failed to create projection".to_string() })?;

    let rgb = image.to_rgb8();
    let mut rectified = RgbImage::new(rw, rh);

    // Use Bilinear instead of Bicubic: bilinear samples a 2×2 neighbourhood
    // vs bicubic's 4×4, dramatically reducing the chance of hitting extreme
    // coordinates that trigger integer overflow in imageproc.
    // For OCR text rectification the quality difference is negligible.
    let warp_result = catch_unwind(AssertUnwindSafe(|| {
        warp_into(
            &rgb,
            proj,
            Interpolation::Bilinear,
            Border::Replicate,
            &mut rectified,
        );
    }));

    if let Err(panic) = warp_result {
        let msg = panic
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("unknown");
        warn!(
            "[rec] warp panicked for bbox {:?} ({}×{}): {} — skipping region",
            region.bbox, rw, rh, msg
        );
        return Err(PaddleOcrError::Projection { reason: format!("warp failed: {}", msg) });
    }

    if rectified.height() as f32 / rectified.width().max(1) as f32 >= 1.5 {
        rectified = image::imageops::rotate90(&rectified);
    }

    let ratio = if rectified.height() == 0 {
        rw as f32 / rh.max(1) as f32
    } else {
        rectified.width() as f32 / rectified.height() as f32
    };
    let resized_w = (rec_height as f32 * ratio).ceil() as u32;
    let resized_w = if let Some(target_w) = rec_width {
        resized_w.max(1).min(target_w.max(4))
    } else {
        resized_w.max(4)
    };
    let input_w = rec_width.unwrap_or(resized_w.max(4));

    let resized = image::imageops::resize(
        &rectified, resized_w, rec_height,
        image::imageops::FilterType::Triangle,
    );

    let input_w_usize = input_w as usize;
    let rec_height_usize = rec_height as usize;
    let mut data = vec![0.0f32; 3 * input_w_usize * rec_height_usize];
    for c in 0..3usize {
        for y in 0..rec_height_usize {
            for x in 0..resized_w as usize {
                let pixel = resized.get_pixel(x as u32, y as u32);
                let val = pixel[c] as f32 / 255.0;
                let offset = c * input_w_usize * rec_height_usize + y * input_w_usize + x;
                data[offset] = (val - 0.5) / 0.5;
            }
        }
    }

    Ok((data, input_w as i64))
}

/// Run the recognition model on preprocessed image data.
///
/// Executes the ONNX inference and returns the raw per-timestep class
/// probabilities for CTC decoding.
///
/// # Arguments
///
/// * `session` — The ONNX runtime session loaded with the recognition model.
/// * `data` — Flattened CHW-format image data from [`preprocess_region`].
/// * `width` — Width of the input tensor (from [`preprocess_region`]).
/// * `height` — Height of the input tensor (`rec_height`).
/// * `input_name` — Name of the model's input tensor.
/// * `output_name` — Name of the model's output tensor.
///
/// # Returns
///
/// A vector of timestep vectors, where each inner vector contains class
/// probabilities for that timestep. Shape: `[timesteps, num_classes]`.
///
/// # Errors
///
/// Returns [`PaddleOcrError::Inference`] if the model execution fails.
pub fn run_recognition(
    session: &mut Session,
    data: &[f32],
    width: i64,
    height: u32,
    input_name: &str,
    output_name: &str,
) -> Result<Vec<Vec<f32>>, PaddleOcrError> {
    let input_tensor = ort::value::Tensor::from_array((
        [1i64, 3, height as i64, width],
        data.to_vec(),
    ))?;

    let outputs = session.run(ort::inputs![input_name => input_tensor])?;

    let (output_shape, output_slice) = outputs[output_name].try_extract_tensor::<f32>()?;
    let timesteps = output_shape[1] as usize;
    let num_classes = output_shape[2] as usize;

    let mut result = Vec::with_capacity(timesteps);
    for t in 0..timesteps {
        let mut row = Vec::with_capacity(num_classes);
        let base = t * num_classes;
        for c in 0..num_classes as usize {
            row.push(output_slice[base + c]);
        }
        result.push(row);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{order_bbox_points, is_valid_quad, quad_area};

    #[test]
    fn orders_quad_points_clockwise_from_top_left() {
        let bbox = [
            [0.0, 10.0],
            [30.0, 0.0],
            [30.0, 10.0],
            [0.0, 0.0],
        ];

        let ordered = order_bbox_points(bbox);

        assert_eq!(ordered[0], [0.0, 0.0]);
        assert_eq!(ordered[1], [30.0, 0.0]);
        assert_eq!(ordered[2], [30.0, 10.0]);
        assert_eq!(ordered[3], [0.0, 10.0]);
    }

    #[test]
    fn valid_quad_passes() {
        let bbox = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]];
        assert!(is_valid_quad(&bbox));
    }

    #[test]
    fn too_small_area_rejected() {
        // area = 0.5 * (0*0 + 5*5 + 5*5 + 0*0 - (0*5 + 5*5 + 5*0 + 0*0))
        //       = 0.5 * (0 + 25 + 25 + 0 - (0 + 25 + 0 + 0)) = 0.5 * 25 = 12.5
        // That's above MIN_QUAD_AREA=10, so test a smaller one
        let bbox = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        // area = 4.0 < 10.0 → rejected
        assert!(!is_valid_quad(&bbox));
    }

    #[test]
    fn nan_coordinate_rejected() {
        let bbox = [[f32::NAN, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]];
        assert!(!is_valid_quad(&bbox));
    }

    #[test]
    fn inf_coordinate_rejected() {
        let bbox = [[f32::INFINITY, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]];
        assert!(!is_valid_quad(&bbox));
    }

    #[test]
    fn quad_area_calculation() {
        let bbox = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        assert_eq!(quad_area(&bbox), 100.0);
    }
}






