use crate::bitstream::BitstreamReader;

/// Number of adaptive medians per channel.
const NUM_MEDIANS: usize = 3;

/// Adaptation divisors for each median level: 128 >> n for median index n.
/// median[0] uses 128, median[1] uses 64, median[2] uses 32.
/// Matches FFmpeg's INC_MED/DEC_MED macros: (128U >> n).
const MEDIAN_DIVS: [u32; NUM_MEDIANS] = [128, 64, 32];

/// Lookup table for wp_exp2: converts log2-packed mantissa to linear value.
/// Entry i represents the fractional part of 2^(i/256).
/// From FFmpeg's ff_wp_exp2_table / WavPack reference's exp2_table.
#[rustfmt::skip]
pub const EXP2_TABLE: [u8; 256] = [
    0x00, 0x01, 0x01, 0x02, 0x03, 0x03, 0x04, 0x05, 0x06, 0x06, 0x07, 0x08, 0x08, 0x09, 0x0a, 0x0b,
    0x0b, 0x0c, 0x0d, 0x0e, 0x0e, 0x0f, 0x10, 0x10, 0x11, 0x12, 0x13, 0x13, 0x14, 0x15, 0x16, 0x16,
    0x17, 0x18, 0x19, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1d, 0x1e, 0x1f, 0x20, 0x20, 0x21, 0x22, 0x23,
    0x24, 0x24, 0x25, 0x26, 0x27, 0x28, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2c, 0x2d, 0x2e, 0x2f, 0x30,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3a, 0x3b, 0x3c, 0x3d,
    0x3e, 0x3f, 0x40, 0x41, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x48, 0x49, 0x4a, 0x4b,
    0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a,
    0x5b, 0x5c, 0x5d, 0x5e, 0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x87, 0x88, 0x89, 0x8a,
    0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b,
    0x9c, 0x9d, 0x9f, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad,
    0xaf, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc8, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf, 0xd0, 0xd2, 0xd3, 0xd4,
    0xd6, 0xd7, 0xd8, 0xd9, 0xdb, 0xdc, 0xdd, 0xde, 0xe0, 0xe1, 0xe2, 0xe4, 0xe5, 0xe6, 0xe8, 0xe9,
    0xea, 0xec, 0xed, 0xee, 0xf0, 0xf1, 0xf2, 0xf4, 0xf5, 0xf6, 0xf8, 0xf9, 0xfa, 0xfc, 0xfd, 0xff,
];

/// Entropy decoder state for one channel.
#[derive(Debug, Clone)]
pub struct EntropyState {
    /// The 3 adaptive medians. Effective value for coding = (median >> 4) + 1.
    pub medians: [u32; NUM_MEDIANS],
}

impl EntropyState {
    pub fn new() -> Self {
        Self {
            medians: [0; NUM_MEDIANS],
        }
    }

    /// Get effective median value for coding: (internal >> 4) + 1.
    /// Always >= 1 to prevent zero divisor.
    #[inline]
    fn get_median(&self, idx: usize) -> u32 {
        (self.medians[idx] >> 4) + 1
    }

    /// Increase median: median += ((median + DIV) / DIV) * 5
    #[inline]
    fn inc_median(&mut self, idx: usize) {
        let div = MEDIAN_DIVS[idx];
        self.medians[idx] += ((self.medians[idx] + div) / div) * 5;
    }

    /// Decrease median: median -= ((median + DIV - 2) / DIV) * 2
    #[inline]
    fn dec_median(&mut self, idx: usize) {
        let div = MEDIAN_DIVS[idx];
        let sub = ((self.medians[idx] + div - 2) / div) * 2;
        self.medians[idx] = self.medians[idx].saturating_sub(sub);
    }
}

/// Restore a median from its log2-packed 16-bit representation.
/// Matches FFmpeg's wp_exp2() / WavPack reference's wp_exp2s().
fn restore_median_log2(packed: u16) -> u32 {
    if packed == 0 {
        return 0;
    }
    let mantissa = (packed & 0xFF) as usize;
    let shift = (packed >> 8) as u32;
    let value = (EXP2_TABLE[mantissa] as u32) | 0x100;
    if shift <= 9 {
        value >> (9 - shift)
    } else {
        value << (shift - 9)
    }
}

/// Entropy decoder for WavPack bitstreams.
///
/// Algorithm faithfully matches ffmpeg's wavpack.c wv_get_value().
///
/// Key features:
/// - Zero-run mode when median[0] < 2 for all channels AND !zero AND !one
/// - Zero/One alternation state machine (raw unary encodes 2*zone + flag)
/// - Escape extension at raw unary == 16
/// - Exp-golomb encoding for zero-run counts
pub struct EntropyDecoder<'a> {
    pub bs: BitstreamReader<'a>,
    pub states: Vec<EntropyState>,
    /// Zero flag: next sample forced to zone 0 without reading bitstream.
    pub zero: bool,
    /// One flag: affects zone calculation for next normal sample.
    pub one: bool,
    /// Pending zero-run counter (shared across channels).
    /// Matches ffmpeg's ctx->zeroes.
    pub zeroes: u32,
    /// Enable detailed trace output for debugging.
    pub trace: bool,
}

impl<'a> EntropyDecoder<'a> {
    pub fn new(bitstream_data: &'a [u8], states: Vec<EntropyState>) -> Self {
        Self {
            bs: BitstreamReader::new(bitstream_data),
            states,
            zero: false,
            one: false,
            zeroes: 0,
            trace: false,
        }
    }

    /// Decode one frame of residuals (one sample per channel).
    pub fn decode_frame(&mut self, num_channels: usize) -> Vec<i32> {
        let mut frame = Vec::with_capacity(num_channels);
        for ch in 0..num_channels {
            frame.push(self.decode_residual(ch));
        }
        frame
    }

    /// Decode one residual for the given channel.
    ///
    /// Faithfully matches ffmpeg's wv_get_value() algorithm.
    fn decode_residual(&mut self, channel: usize) -> i32 {
        let t = self.trace;
        let bit_pos_start = self.bs.bit_position();

        // 1. Zero-run check: median[0] < 2 for ALL channels AND !zero AND !one
        let all_medians_low = self.states.iter().all(|s| s.medians[0] < 2);
        if t {
            eprintln!(
                "    ENTROPY: ch={} bit={} medians={:?} zero={} one={} zeroes={} all_low={}",
                channel, bit_pos_start, self.states[channel].medians,
                self.zero, self.one, self.zeroes, all_medians_low
            );
        }
        if all_medians_low && !self.zero && !self.one {
            if self.zeroes > 0 {
                self.zeroes -= 1;
                if t {
                    eprintln!("    ENTROPY: decrement zeroes → {}", self.zeroes);
                }
                if self.zeroes > 0 {
                    return 0;
                }
                if t {
                    eprintln!("    ENTROPY: zeroes exhausted, fall through to normal decode");
                }
            } else {
                let mut run = self.bs.read_unary();
                if t {
                    eprintln!("    ENTROPY: read zero-run unary={}", run);
                }
                if run >= 2 {
                    if run <= 32 {
                        let extra = self.bs.read_bits(run - 1);
                        run = extra | (1u32 << (run - 1));
                        if t {
                            eprintln!("    ENTROPY: exp-golomb extra={} → run={}", extra, run);
                        }
                    }
                }
                self.zeroes = run;
                if self.zeroes > 0 {
                    for state in &mut self.states {
                        state.medians = [0, 0, 0];
                    }
                    if t {
                        eprintln!("    ENTROPY: zero-run={}, clear medians, return 0", self.zeroes);
                    }
                    return 0;
                }
                if t {
                    eprintln!("    ENTROPY: zero-run=0, fall through to normal decode");
                }
            }
        }

        // 2. Normal decode with zero/one state machine
        let zone;
        if self.zero {
            zone = 0;
            self.zero = false;
            if t {
                eprintln!("    ENTROPY: zero flag → zone=0");
            }
        } else {
            let mut raw = self.bs.read_unary();
            if t {
                eprintln!("    ENTROPY: read unary raw={}", raw);
            }

            if raw == 16 {
                let t2 = self.bs.read_unary();
                if t2 < 2 {
                    raw += t2;
                } else if t2 <= 32 {
                    raw += self.bs.read_bits(t2 - 1) | (1u32 << (t2 - 1));
                }
                if t {
                    eprintln!("    ENTROPY: escape extension → raw={}", raw);
                }
            }

            if self.one {
                self.one = (raw & 1) != 0;
                zone = (raw >> 1) + 1;
            } else {
                self.one = (raw & 1) != 0;
                zone = raw >> 1;
            }
            self.zero = !self.one;
            if t {
                eprintln!(
                    "    ENTROPY: raw={} → zone={} (new: one={} zero={})",
                    raw, zone, self.one, self.zero
                );
            }
        }

        // 3. Decode magnitude from zone, updating medians
        let magnitude = self.decode_zone(channel, zone);
        if t {
            eprintln!(
                "    ENTROPY: zone={} → magnitude={} medians_after={:?}",
                zone, magnitude, self.states[channel].medians
            );
        }

        // 4. Sign bit (always read, even for magnitude 0 in ffmpeg)
        let sign = self.bs.read_bit();
        let result = if sign != 0 {
            !(magnitude as i32)
        } else {
            magnitude as i32
        };
        if t {
            eprintln!(
                "    ENTROPY: sign={} → result={} (bits consumed: {})",
                sign, result, self.bs.bit_position() - bit_pos_start
            );
        }
        result
    }

    /// Decode value from zone, updating medians.
    /// Matches ffmpeg's base/add/get_tail pattern.
    fn decode_zone(&mut self, channel: usize, zone: u32) -> u32 {
        let t = self.trace;
        let state = &self.states[channel];
        match zone {
            0 => {
                let m = state.get_median(0);
                if t { eprintln!("      ZONE0: GET_MED(0)={} k={}", m, m-1); }
                let remainder = self.read_golomb_remainder(m - 1);
                self.states[channel].dec_median(0);
                if t { eprintln!("      ZONE0: rem={} total={}", remainder, remainder); }
                remainder
            }
            1 => {
                let m0 = state.get_median(0);
                let m1 = state.get_median(1);
                if t { eprintln!("      ZONE1: GET_MED(0)={} GET_MED(1)={} k={}", m0, m1, m1-1); }
                let remainder = self.read_golomb_remainder(m1 - 1);
                self.states[channel].inc_median(0);
                self.states[channel].dec_median(1);
                if t { eprintln!("      ZONE1: base={} rem={} total={}", m0, remainder, m0+remainder); }
                m0 + remainder
            }
            2 => {
                let m0 = state.get_median(0);
                let m1 = state.get_median(1);
                let m2 = state.get_median(2);
                if t { eprintln!("      ZONE2: meds=({},{},{}) k={}", m0, m1, m2, m2-1); }
                let remainder = self.read_golomb_remainder(m2 - 1);
                self.states[channel].inc_median(0);
                self.states[channel].inc_median(1);
                self.states[channel].dec_median(2);
                if t { eprintln!("      ZONE2: base={} rem={} total={}", m0+m1, remainder, m0+m1+remainder); }
                m0 + m1 + remainder
            }
            _ => {
                let m0 = state.get_median(0);
                let m1 = state.get_median(1);
                let m2 = state.get_median(2);
                let base = m0 + m1 + m2 * (zone - 2);
                if t { eprintln!("      ZONE{}: meds=({},{},{}) base={} k={}", zone, m0, m1, m2, base, m2-1); }
                let remainder = self.read_golomb_remainder(m2 - 1);
                self.states[channel].inc_median(0);
                self.states[channel].inc_median(1);
                self.states[channel].inc_median(2);
                if t { eprintln!("      ZONE{}: rem={} total={}", zone, remainder, base+remainder); }
                base + remainder
            }
        }
    }

    /// Read adjusted-binary Golomb remainder (get_tail in ffmpeg).
    /// Returns a value in [0, k] inclusive.
    fn read_golomb_remainder(&mut self, k: u32) -> u32 {
        if k < 1 {
            return 0;
        }

        let p = 31 - k.leading_zeros(); // av_log2(k)
        let e = (1u64 << (p + 1)) as u32 - k - 1;

        let bit_before = self.bs.bit_position();
        let mut res = self.bs.read_bits(p);
        let initial_res = res;
        if res >= e {
            let extra_bit = self.bs.read_bit();
            res = res * 2 - e + extra_bit;
            if self.trace {
                eprintln!("        get_tail(k={}): p={} e={} read_bits({})={} >=e, extra_bit={} → res={} bit_pos={}",
                    k, p, e, p, initial_res, extra_bit, res, bit_before);
            }
        } else if self.trace {
            eprintln!("        get_tail(k={}): p={} e={} read_bits({})={} <e → res={} bit_pos={}",
                k, p, e, p, initial_res, res, bit_before);
        }
        res
    }
}

/// Parse entropy vars from ID_ENTROPY_VARS sub-block (0x05).
pub fn parse_entropy_vars(data: &[u8], num_channels: usize) -> Vec<EntropyState> {
    let mut states = Vec::with_capacity(num_channels);
    for ch in 0..num_channels {
        let offset = ch * 6;
        let mut medians = [0u32; NUM_MEDIANS];
        for i in 0..NUM_MEDIANS {
            let pos = offset + i * 2;
            if pos + 2 <= data.len() {
                let packed = u16::from_le_bytes([data[pos], data[pos + 1]]);
                medians[i] = restore_median_log2(packed);
            }
        }
        states.push(EntropyState { medians });
    }
    states
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_median_values() {
        assert_eq!(restore_median_log2(0), 0);
        // packed 0x0706: mantissa=6, shift=7, EXP2_TABLE[6]=4
        // value = 0x104 = 260, 260 >> (9-7) = 260 >> 2 = 65
        assert_eq!(restore_median_log2(0x0706), 65);
        let state = EntropyState { medians: [65, 0, 0] };
        // GET_MED = (65 >> 4) + 1 = 4 + 1 = 5
        assert_eq!(state.get_median(0), 5);
    }

    #[test]
    fn parse_mono_entropy_vars() {
        // From mono16 test file entropy vars: [0x06, 0x07, 0x49, 0x06, 0x06, 0x07]
        let data = [0x06, 0x07, 0x49, 0x06, 0x06, 0x07];
        let states = parse_entropy_vars(&data, 1);
        assert_eq!(states.len(), 1);
        // packed 0x0706 → 65, GET_MED = 5
        assert_eq!(states[0].get_median(0), 5);
        // packed 0x0649 → mantissa=73, shift=6, EXP2_TABLE[73]=0x38=56
        // value = 0x138 = 312, 312 >> (9-6) = 312 >> 3 = 39, GET_MED = (39>>4)+1 = 3
        assert_eq!(states[0].get_median(1), 3);
        // packed 0x0706 → 65, GET_MED = 5
        assert_eq!(states[0].get_median(2), 5);
    }

    #[test]
    fn median_update_inc_dec() {
        // Use restored median values (no << 4 scaling)
        let mut s = EntropyState { medians: [65, 39, 65] };
        // INC_MED(0): div=128, (65+128)/128 = 1, 1*5=5, 65+5=70
        s.inc_median(0);
        assert_eq!(s.medians[0], 70);
        s.medians[0] = 65;
        // DEC_MED(0): div=128, (65+128-2)/128 = 191/128 = 1, 1*2=2, 65-2=63
        s.dec_median(0);
        assert_eq!(s.medians[0], 63);
    }

    #[test]
    fn golomb_remainder_matches_get_tail() {
        // get_tail(5): k=5, p=2, e=(8-5-1)=2
        // Should return values in [0,5]
        // Test with known bit patterns
        let data = [0x00]; // all zero bits
        let states = vec![EntropyState::new()];
        let mut dec = EntropyDecoder::new(&data, states);
        // read_bits(2) = 0, 0 < 2 = e, so res = 0
        assert_eq!(dec.read_golomb_remainder(5), 0);
    }
}
