/// A single decorrelation pass with its parameters.
///
/// WavPack applies multiple LMS-like decorrelation passes during encoding.
/// The decoder reverses them in opposite order.
#[derive(Debug, Clone)]
pub struct DecorrPass {
    /// Decorrelation term (1-8: delay, 17: linear extrap, 18: weighted extrap,
    /// -1/-2/-3: cross-channel).
    pub term: i32,
    /// Weight update delta (controls adaptation speed).
    pub delta: i32,
    /// Adaptive filter weights (1 for mono, 2 for stereo).
    pub weights: [i32; 2],
    /// History samples buffer for each channel.
    /// For terms 1-8: circular buffer of `term` previous samples.
    /// For terms 17/18: last 2 samples.
    /// For terms -1/-2/-3: last sample of each channel.
    pub samples: [[i32; 8]; 2],
}

impl DecorrPass {
    pub fn new(term: i32, delta: i32) -> Self {
        Self {
            term,
            delta,
            weights: [0; 2],
            samples: [[0; 8]; 2],
        }
    }
}

/// Parse decorrelation terms from the ID_DECORR_TERMS sub-block (0x02).
///
/// Each byte encodes: term = ((byte & 0x1f) - 5), delta = (byte >> 5).
/// Terms are stored in encoding order (first pass first), but the
/// decoder applies them in reverse.
pub fn parse_decorr_terms(data: &[u8]) -> Vec<DecorrPass> {
    let mut passes = Vec::with_capacity(data.len());
    for &byte in data {
        let raw_term = (byte & 0x1f) as i32 - 5;
        let delta = (byte >> 5) as i32;
        // Valid terms: 1-8, 17, 18, -1, -2, -3
        let term = match raw_term {
            1..=8 | 17 | 18 | -3..=-1 => raw_term,
            _ => continue, // skip invalid
        };
        passes.push(DecorrPass::new(term, delta));
    }
    passes
}

/// Parse decorrelation weights from the ID_DECORR_WEIGHTS sub-block (0x03).
///
/// Each weight is stored as a single signed byte with the restore formula:
/// `w = stored << 3; if w > 0 { w += (w + 64) >> 7 }`
pub fn parse_decorr_weights(data: &[u8], passes: &mut [DecorrPass], is_mono: bool) {
    let channels = if is_mono { 1 } else { 2 };
    let mut idx = 0;

    for pass in passes.iter_mut() {
        for ch in 0..channels {
            if idx < data.len() {
                pass.weights[ch] = restore_weight(data[idx] as i8);
                idx += 1;
            }
        }
    }
}

/// Restore a decorrelation weight from its packed single-byte form.
fn restore_weight(stored: i8) -> i32 {
    let mut w = (stored as i32) * 8;
    if w > 0 {
        w += (w + 64) >> 7;
    }
    // Negative weights: no adjustment (matches WavPack 5.x reference).
    // The store_weight/restore_weight round-trip is exact without it.
    w
}

/// Parse decorrelation samples from the ID_DECORR_SAMPLES sub-block (0x04).
///
/// Each sample is stored as a log2-encoded 16-bit value (same format as
/// entropy vars but 16-bit). The number of samples per pass depends on
/// the term and number of channels.
pub fn parse_decorr_samples(data: &[u8], passes: &mut [DecorrPass], is_mono: bool) {
    let mut pos = 0;

    for pass in passes.iter_mut() {
        let term = pass.term;

        if term > 8 {
            // Terms 17, 18: 2 samples per channel
            let channels = if is_mono { 1 } else { 2 };
            for ch in 0..channels {
                for s in 0..2 {
                    if pos + 2 <= data.len() {
                        pass.samples[ch][s] = restore_sample_log2(read_i16_le(data, pos));
                        pos += 2;
                    }
                }
            }
        } else if term < 0 {
            // Cross-channel terms: 1 sample per channel
            for ch in 0..2 {
                if pos + 2 <= data.len() {
                    pass.samples[ch][0] = restore_sample_log2(read_i16_le(data, pos));
                    pos += 2;
                }
            }
        } else {
            // Terms 1-8: `term` samples per channel, INTERLEAVED by sample index.
            // WavPack stores: A[0], B[0], A[1], B[1], ... (not grouped by channel).
            let channels = if is_mono { 1 } else { 2 };
            for s in 0..term as usize {
                for ch in 0..channels {
                    if pos + 2 <= data.len() {
                        pass.samples[ch][s] = restore_sample_log2(read_i16_le(data, pos));
                        pos += 2;
                    }
                }
            }
        }
    }
}

/// Restore a sample from its log2-packed signed 16-bit representation.
///
/// The packed value is a signed i16 where:
/// - Negative packed values represent negative samples: -restore(|packed|)
/// - Positive packed values: mantissa = packed & 0xFF, shift = packed >> 8
///   restored = (256 + mantissa) adjusted by (shift - 9)
fn restore_sample_log2(raw: i16) -> i32 {
    if raw == 0 {
        return 0;
    }
    if raw < 0 {
        // Handle i16::MIN carefully (negate as i32)
        return -restore_sample_log2_unsigned((-raw as i32) as u16);
    }
    restore_sample_log2_unsigned(raw as u16)
}

fn restore_sample_log2_unsigned(packed: u16) -> i32 {
    if packed == 0 {
        return 0;
    }
    let mantissa = (packed & 0xFF) as usize;
    let shift = (packed >> 8) as u32;
    let value = (crate::entropy::EXP2_TABLE[mantissa] as u32) | 0x100;
    if shift <= 9 {
        (value >> (9 - shift)) as i32
    } else {
        (value << (shift - 9)) as i32
    }
}

fn read_i16_le(data: &[u8], pos: usize) -> i16 {
    i16::from_le_bytes([data[pos], data[pos + 1]])
}

/// Apply decorrelation passes in reverse order to reconstruct original samples.
///
/// `residuals` contains the entropy-decoded residuals for this block.
/// For stereo, `residuals[0]` and `residuals[1]` are the two channels.
/// `passes` are modified in-place (weights and sample history are updated).
pub fn apply_decorrelation(
    passes: &mut [DecorrPass],
    samples: &mut [Vec<i32>],
    num_samples: usize,
    is_mono: bool,
) {
    let trace = std::env::var("WP_TRACE").is_ok();
    // Apply passes in reverse order (last pass in array = innermost, applied first by decoder)
    for pass_idx in (0..passes.len()).rev() {
        if trace && is_mono {
            eprintln!("  === PASS[{}] term={} w={} delta={} hist={:?} ===",
                pass_idx, passes[pass_idx].term, passes[pass_idx].weights[0],
                passes[pass_idx].delta,
                &passes[pass_idx].samples[0][..std::cmp::min(passes[pass_idx].term.unsigned_abs() as usize, 8).max(2)]);
            eprintln!("    pre:  s[0..3] = {:?}", &samples[0][..3.min(num_samples)]);
        }
        if is_mono {
            apply_pass_mono(&mut passes[pass_idx], &mut samples[0], num_samples);
        } else {
            apply_pass_stereo(&mut passes[pass_idx], samples, num_samples);
        }
        if trace && is_mono {
            eprintln!("    post: s[0..3] = {:?} w_after={}",
                &samples[0][..3.min(num_samples)], passes[pass_idx].weights[0]);
        }
    }
}

/// Apply a single decorrelation pass to mono samples.
fn apply_pass_mono(pass: &mut DecorrPass, samples: &mut [i32], num_samples: usize) {
    let term = pass.term;

    for i in 0..num_samples {
        let prediction = match term {
            1..=8 => {
                // Delay by `term` samples — uses circular buffer like the official WavPack decoder.
                // History samples[0..term] are stored chronologically (oldest first).
                if i >= term as usize {
                    samples[i - term as usize]
                } else {
                    pass.samples[0][i]
                }
            }
            17 => {
                // Linear extrapolation: 2*s(k-1) - s(k-2)
                let s1 = if i >= 1 {
                    samples[i - 1]
                } else {
                    pass.samples[0][0]
                };
                let s2 = if i >= 2 {
                    samples[i - 2]
                } else if i == 1 {
                    pass.samples[0][0]
                } else {
                    pass.samples[0][1]
                };
                2 * s1 - s2
            }
            18 => {
                // Weighted extrapolation: (3*s(k-1) - s(k-2)) / 2
                let s1 = if i >= 1 {
                    samples[i - 1]
                } else {
                    pass.samples[0][0]
                };
                let s2 = if i >= 2 {
                    samples[i - 2]
                } else if i == 1 {
                    pass.samples[0][0]
                } else {
                    pass.samples[0][1]
                };
                ((3i64 * s1 as i64 - s2 as i64) >> 1) as i32
            }
            _ => 0, // cross-channel terms don't apply to mono
        };

        let residual = samples[i];
        let weighted = apply_weight(pass.weights[0], prediction);
        let reconstructed = residual + weighted;

        if i < 3 && std::env::var("WP_TRACE").is_ok() {
            eprintln!("    [{i}] pred={prediction} w={} res={residual} weighted={weighted} out={reconstructed}",
                pass.weights[0]);
        }

        update_weight(&mut pass.weights[0], pass.delta, residual, prediction);

        samples[i] = reconstructed;
    }

    // Save last samples as history for next block
    let start = num_samples.saturating_sub(8);
    for s in 0..8.min(num_samples) {
        pass.samples[0][s] = samples[num_samples - 1 - s];
    }
    let _ = start; // suppress warning
}

/// Apply a single decorrelation pass to stereo samples.
fn apply_pass_stereo(pass: &mut DecorrPass, samples: &mut [Vec<i32>], num_samples: usize) {
    let term = pass.term;

    match term {
        -3..=-1 => {
            // Cross-channel decorrelation
            apply_cross_channel(pass, samples, num_samples);
        }
        _ => {
            // Independent per-channel decorrelation
            for ch in 0..2 {
                apply_pass_channel(pass, ch, &mut samples[ch], num_samples);
            }
        }
    }
}

/// Apply an independent decorrelation pass to a single channel.
fn apply_pass_channel(
    pass: &mut DecorrPass,
    ch: usize,
    samples: &mut [i32],
    num_samples: usize,
) {
    let term = pass.term;

    for i in 0..num_samples {
        let prediction = match term {
            1..=8 => {
                if i >= term as usize {
                    samples[i - term as usize]
                } else {
                    pass.samples[ch][i]
                }
            }
            17 => {
                let s1 = if i >= 1 {
                    samples[i - 1]
                } else {
                    pass.samples[ch][0]
                };
                let s2 = if i >= 2 {
                    samples[i - 2]
                } else if i == 1 {
                    pass.samples[ch][0]
                } else {
                    pass.samples[ch][1]
                };
                2 * s1 - s2
            }
            18 => {
                let s1 = if i >= 1 {
                    samples[i - 1]
                } else {
                    pass.samples[ch][0]
                };
                let s2 = if i >= 2 {
                    samples[i - 2]
                } else if i == 1 {
                    pass.samples[ch][0]
                } else {
                    pass.samples[ch][1]
                };
                ((3i64 * s1 as i64 - s2 as i64) >> 1) as i32
            }
            _ => 0,
        };

        let residual = samples[i];
        let reconstructed =
            residual + apply_weight(pass.weights[ch], prediction);

        update_weight(&mut pass.weights[ch], pass.delta, residual, prediction);

        samples[i] = reconstructed;
    }

    // Save history
    for s in 0..8.min(num_samples) {
        pass.samples[ch][s] = samples[num_samples - 1 - s];
    }
}

/// Apply cross-channel decorrelation (terms -1, -2, -3).
///
/// These terms create feedback between channels. Each processes BOTH channels
/// per sample, with the decoded value of one channel feeding into the other.
/// Cross-channel terms use `update_weight_clip` (clamped to ±1024).
///
/// Matches `decorr_stereo_pass` cases -1, -2, -3 from WavPack reference.
fn apply_cross_channel(pass: &mut DecorrPass, samples: &mut [Vec<i32>], num_samples: usize) {
    match pass.term {
        -1 => {
            // Channel A predicts from saved sample (previous B), then
            // channel B predicts from the just-decoded A.
            for i in 0..num_samples {
                let residual_a = samples[0][i];
                let decoded_a = residual_a + apply_weight(pass.weights[0], pass.samples[0][0]);
                update_weight_clip(&mut pass.weights[0], pass.delta, pass.samples[0][0], residual_a);
                samples[0][i] = decoded_a;

                let residual_b = samples[1][i];
                let decoded_b = residual_b + apply_weight(pass.weights[1], decoded_a);
                update_weight_clip(&mut pass.weights[1], pass.delta, decoded_a, residual_b);
                samples[1][i] = decoded_b;

                // Cross-feedback: next A predicts from this B
                pass.samples[0][0] = decoded_b;
            }
        }
        -2 => {
            // Channel B predicts from saved sample (previous A), then
            // channel A predicts from the just-decoded B.
            for i in 0..num_samples {
                let residual_b = samples[1][i];
                let decoded_b = residual_b + apply_weight(pass.weights[1], pass.samples[1][0]);
                update_weight_clip(&mut pass.weights[1], pass.delta, pass.samples[1][0], residual_b);
                samples[1][i] = decoded_b;

                let residual_a = samples[0][i];
                let decoded_a = residual_a + apply_weight(pass.weights[0], decoded_b);
                update_weight_clip(&mut pass.weights[0], pass.delta, decoded_b, residual_a);
                samples[0][i] = decoded_a;

                // Cross-feedback: next B predicts from this A
                pass.samples[1][0] = decoded_a;
            }
        }
        -3 => {
            // Both channels predict independently from saved cross-samples,
            // then swap: A's decoded goes to B's history and vice versa.
            for i in 0..num_samples {
                let residual_a = samples[0][i];
                let decoded_a = residual_a + apply_weight(pass.weights[0], pass.samples[0][0]);
                update_weight_clip(&mut pass.weights[0], pass.delta, pass.samples[0][0], residual_a);

                let residual_b = samples[1][i];
                let decoded_b = residual_b + apply_weight(pass.weights[1], pass.samples[1][0]);
                update_weight_clip(&mut pass.weights[1], pass.delta, pass.samples[1][0], residual_b);

                samples[0][i] = decoded_a;
                samples[1][i] = decoded_b;
                // Cross-save: A goes to B's history, B goes to A's history
                pass.samples[1][0] = decoded_a;
                pass.samples[0][0] = decoded_b;
            }
        }
        _ => unreachable!(),
    }
}

/// Apply weight to prediction, matching the official WavPack apply_weight macro.
///
/// For predictions that fit in i16 [-32768, 32767], uses the simple 32-bit formula.
/// For larger predictions, uses 64-bit to avoid overflow. Both produce identical
/// results (proven algebraically); 64-bit is simpler than WavPack's split formula.
#[inline]
fn apply_weight(weight: i32, sample: i32) -> i32 {
    ((weight as i64 * sample as i64 + 512) >> 10) as i32
}

/// Update a decorrelation weight using the LMS-like rule.
///
/// `w += delta * sgn(prediction) * sgn(residual)`
/// where sgn returns +delta, -delta, or 0.
///
/// Matches FFmpeg's inline formula in wv_unpack_mono/stereo:
///   `weight -= ((((T ^ A) >> 30) & 2) - 1) * delta`
/// Weight update without clamping — matches the official WavPack decoder (wvunpack).
/// FFmpeg uses UPDATE_WEIGHT_CLIP with clamping, but that produces wrong results
/// when verified against wvunpack. The WavPack format allows weights outside [-1024, 1024].
#[inline]
fn update_weight(weight: &mut i32, delta: i32, residual: i32, prediction: i32) {
    if residual != 0 && prediction != 0 {
        if (residual ^ prediction) >= 0 {
            *weight += delta;
        } else {
            *weight -= delta;
        }
    }
}

/// Update a decorrelation weight with clamping to ±1024.
///
/// Used for cross-channel terms (-1, -2, -3). Same sign logic as `update_weight`
/// but clamps the result. Matches WavPack's `update_weight_clip` macro.
#[inline]
fn update_weight_clip(weight: &mut i32, delta: i32, source: i32, result: i32) {
    if source != 0 && result != 0 {
        let s = (source ^ result) >> 31; // 0 for same sign, -1 for different
        let mut w = (*weight ^ s) + (delta - s);
        if w > 1024 {
            w = 1024;
        }
        *weight = (w ^ s) - s;
    }
}

/// Apply joint stereo inverse transform.
///
/// In joint stereo mode, the encoder stores mid/side:
///   channel 0 = L - R (difference)
///   channel 1 = R + ((L - R) >> 1) (side)
///
/// Inverse (matching FFmpeg's `L += (unsigned)(R -= (unsigned)(L >> 1))`):
///   R = side - (mid >> 1)
///   L = mid + R
pub fn undo_joint_stereo(left: &mut [i32], right: &mut [i32], num_samples: usize) {
    for i in 0..num_samples {
        right[i] -= left[i] >> 1;
        left[i] += right[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_weight_values() {
        // 0 → 0
        assert_eq!(restore_weight(0), 0);
        // 1 → 8 + (8 + 64) >> 7 = 8 + 0 = 8
        assert_eq!(restore_weight(1), 8);
        // -1 → -8 (no adjustment for negative)
        assert_eq!(restore_weight(-1), -8);
        // 64 → 512 + (512 + 64) >> 7 = 512 + 4 = 516
        assert_eq!(restore_weight(64), 516);
        // -33 → -264 (no adjustment for negative)
        assert_eq!(restore_weight(-33), -264);
        // -47 → -376 (no adjustment for negative)
        assert_eq!(restore_weight(-47), -376);
    }

    #[test]
    fn restore_sample_log2_zero() {
        assert_eq!(restore_sample_log2(0), 0);
    }

    #[test]
    fn update_weight_same_sign() {
        let mut w = 100;
        update_weight(&mut w, 2, 5, 3); // both positive
        assert_eq!(w, 102);
    }

    #[test]
    fn update_weight_diff_sign() {
        let mut w = 100;
        update_weight(&mut w, 2, -5, 3); // different signs
        assert_eq!(w, 98);
    }

    #[test]
    fn update_weight_zero_no_change() {
        let mut w = 100;
        update_weight(&mut w, 2, 0, 3); // residual is zero
        assert_eq!(w, 100);
    }

    #[test]
    fn update_weight_no_clamp() {
        let mut w = 1023;
        update_weight(&mut w, 5, 1, 1);
        assert_eq!(w, 1028); // no clamping — matches wvunpack reference
    }

    #[test]
    fn joint_stereo_roundtrip() {
        // Simulate encoding: mid = L - R, side = (L + R) / 2
        let orig_l: Vec<i32> = vec![100, 200, -50, 0];
        let orig_r: Vec<i32> = vec![80, 150, -30, 10];

        let mut mid: Vec<i32> = orig_l
            .iter()
            .zip(&orig_r)
            .map(|(&l, &r)| l - r)
            .collect();
        let mut side: Vec<i32> = orig_l
            .iter()
            .zip(&orig_r)
            .map(|(&l, &r)| (l + r) >> 1)
            .collect();

        undo_joint_stereo(&mut mid, &mut side, 4);

        // Check reconstruction
        for i in 0..4 {
            // May differ by 1 due to integer rounding
            assert!((mid[i] - orig_l[i]).abs() <= 1, "L[{i}]: {} vs {}", mid[i], orig_l[i]);
            assert!(
                (side[i] - orig_r[i]).abs() <= 1,
                "R[{i}]: {} vs {}",
                side[i],
                orig_r[i]
            );
        }
    }

    #[test]
    fn parse_terms_basic() {
        // term=18 → raw=23, delta=2 → byte = (2 << 5) | (23+5)... wait
        // term = (byte & 0x1f) - 5, so term=18 → (byte & 0x1f) = 23
        // delta = byte >> 5, delta=2 → byte = (2 << 5) | 23 = 64+23 = 87
        let data = [87u8];
        let passes = parse_decorr_terms(&data);
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].term, 18);
        assert_eq!(passes[0].delta, 2);
    }
}
