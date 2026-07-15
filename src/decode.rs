use log::warn;

/// Result of CTC decoding containing the recognized text and confidence score.
///
/// Produced by the CTC greedy decoder from raw model output probabilities.
///
/// # Example
///
/// ```ignore
/// // Typically obtained from OcrEngine::recognize_text
/// let decoded = engine.recognize_text(&image, &region)?;
/// println!("Text: {}, Score: {:.2}", decoded.text, decoded.score);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct DecodedText {
    /// The decoded text string.
    pub text: String,
    /// Average confidence score of the recognized characters in `[0.0, 1.0]`.
    pub score: f32,
}

/// CTC (Connectionist Temporal Classification) greedy decoder.
///
/// Decodes the raw probability output from the recognition model into text by:
/// 1. Taking the argmax at each timestep
/// 2. Collapsing repeated characters
/// 3. Removing the blank token (index 0)
/// 4. Computing the average confidence of kept characters
///
/// # Arguments
///
/// * `probs` — Model output probabilities, shape `[timesteps, num_classes]`.
///   Each inner vector is the softmax output for one timestep.
/// * `keys` — Character dictionary, one character per element. Index 0 is
///   reserved for the blank token; `keys[0]` maps to class index 1.
///
/// # Returns
///
/// A [`DecodedText`] with the recognized text and average confidence score.
///
/// # Example
///
/// ```ignore
/// use paddleocr_rs_onnx::decode::ctc_decode;
///
/// // Simulated model output: 3 timesteps, 5 classes (blank + 4 chars)
/// let probs = vec![
///     vec![0.1, 0.7, 0.1, 0.05, 0.05],  // timestep 0 -> 'H'
///     vec![0.05, 0.05, 0.8, 0.05, 0.05], // timestep 1 -> 'i'
///     vec![0.9, 0.02, 0.02, 0.03, 0.03], // timestep 2 -> blank
/// ];
/// let keys = vec!["H".into(), "i".into(), "!".into(), "x".into()];
/// let result = ctc_decode(&probs, &keys);
/// assert_eq!(result.text, "Hi");
/// ```
pub fn ctc_decode(probs: &[Vec<f32>], keys: &[String]) -> DecodedText {
    let blank_idx = 0;
    let mut text = String::new();
    let mut prev_char_idx = blank_idx;
    let mut dropped = 0u32;
    let mut confidences = Vec::new();

    for timestep in probs {
        let (max_idx, max_prob) = timestep
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or((blank_idx, 0.0));

        if max_idx != blank_idx && max_idx != prev_char_idx {
            let keys_idx = max_idx - 1;
            if let Some(token) = keys.get(keys_idx) {
                text.push_str(token);
                confidences.push(max_prob);
            } else {
                dropped += 1;
            }
        }
        prev_char_idx = max_idx;
    }

    if dropped > 0 {
        warn!(
            "[ctc_decode] dropped {} out-of-range indices (keys: {}, max max_idx seen: {}).",
            dropped,
            keys.len(),
            probs.iter()
                .flat_map(|t| t.iter().enumerate())
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0),
        );
    }

    let score = if confidences.is_empty() {
        0.0
    } else {
        confidences.iter().sum::<f32>() / confidences.len() as f32
    };

    DecodedText { text, score }
}
