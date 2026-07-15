#[derive(Debug, Clone)]
pub struct DecodedText {
    pub text: String,
    pub score: f32,
}

use log::warn;

/// CTC decode: blank=0, characters start at index 1.
/// Collapse repeats, remove blanks, and average kept character scores.
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
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
