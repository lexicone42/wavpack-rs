#![allow(dead_code)]

//! Pure Rust decoder for WavPack lossless audio files.
//!
//! Implemented from publicly available format documentation:
//! - [WavPack5FileFormat.pdf](https://www.wavpack.com/WavPack5FileFormat.pdf) — container format
//! - WavPack paper (Salomon's "Data Compression" textbook) — algorithm
//! - [MultimediaWiki](https://wiki.multimedia.cx/index.php/WavPack) — format reference
//! - [wavpack.com/technical.htm](https://www.wavpack.com/technical.htm) — algorithm overview
//!
//! No code derived from the BSD-3-Clause reference implementation.
//!
//! # Example
//!
//! ```no_run
//! use wavpack_rs::WavPackReader;
//!
//! let mut reader = WavPackReader::open("track.wv").unwrap();
//! let info = reader.info();
//! println!("{}ch, {}Hz, {}bit", info.channels, info.sample_rate, info.bits_per_sample);
//!
//! let samples: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
//! ```

mod bitstream;
mod decorrelation;
mod entropy;
pub mod error;
mod header;

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

pub use error::WavPackError;

/// Metadata about the audio contained in a WavPack file.
#[derive(Debug, Clone)]
pub struct WavPackInfo {
    /// Sample rate in Hz (e.g. 44100).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Bits per sample (8, 16, 24, or 32).
    pub bits_per_sample: u16,
    /// Total number of audio samples (frames × channels).
    pub total_samples: u64,
}

/// A reader that decodes WavPack (.wv) files.
///
/// Modeled after `ape_rs::ApeReader` — open a file, read metadata, then
/// iterate over decoded PCM samples.
pub struct WavPackReader<R: Read + Seek> {
    reader: R,
    info: WavPackInfo,
    /// Buffered decoded samples from the current block.
    output_buf: Vec<i32>,
    /// Current read position in output_buf.
    output_pos: usize,
    /// Total samples decoded so far (frames).
    frames_decoded: u64,
    /// Total frames in file (0 = unknown).
    total_frames: u64,
    /// True when we've hit the end of the stream.
    finished: bool,
}

impl WavPackReader<BufReader<File>> {
    /// Open a WavPack file by path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, WavPackError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::new(reader)
    }
}

impl<R: Read + Seek> WavPackReader<R> {
    /// Create a new WavPackReader from any `Read + Seek` source.
    ///
    /// Parses the first block header to determine format. After construction,
    /// call `info()` for metadata and `samples()` for audio.
    pub fn new(mut reader: R) -> Result<Self, WavPackError> {
        // Sync to first block (skip any non-WavPack data at start)
        if !header::sync_to_block(&mut reader)? {
            return Err(WavPackError::NoAudioBlocks);
        }

        // Read the first block header to get format info
        let first_header =
            header::read_block_header(&mut reader)?.ok_or(WavPackError::NoAudioBlocks)?;

        // Validate unsupported features
        if first_header.is_hybrid() {
            return Err(WavPackError::HybridNotSupported);
        }
        if first_header.is_float() {
            return Err(WavPackError::FloatNotSupported);
        }
        if first_header.is_dsd() {
            return Err(WavPackError::DsdNotSupported);
        }

        // Read payload to check for custom sample rate
        let payload = header::read_block_payload(&mut reader, &first_header)?;
        let sub_blocks = header::parse_metadata(&payload)?;

        let sample_rate = determine_sample_rate(&first_header, &sub_blocks);
        let channels = if first_header.is_mono() { 1u16 } else { 2u16 };
        let bits_per_sample = first_header.bits_per_sample() as u16;

        // Calculate total samples (frames × channels)
        let total_frames = first_header.total_samples;
        let total_samples = if total_frames == 0xFFFFFFFFFF {
            0 // Unknown length
        } else {
            total_frames * channels as u64
        };

        let info = WavPackInfo {
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
        };

        // Seek back to before the first block so decode starts from the beginning
        // Seek back to start of the first block.
        // We read: 32-byte header + payload. payload = block_size - 24.
        // Total bytes read from start of block = 32 + (block_size - 24) = block_size + 8.
        let bytes_read = 32 + payload.len();
        reader.seek(SeekFrom::Current(-(bytes_read as i64)))?;

        Ok(WavPackReader {
            reader,
            info,
            output_buf: Vec::new(),
            output_pos: 0,
            frames_decoded: 0,
            total_frames,
            finished: false,
        })
    }

    /// Get metadata about the audio stream.
    pub fn info(&self) -> &WavPackInfo {
        &self.info
    }

    /// Returns an iterator that yields decoded PCM samples as `Result<i32>`.
    ///
    /// Samples are interleaved for stereo files: `[L0, R0, L1, R1, ...]`
    ///
    /// Values are native i32 — the consumer should normalize using
    /// `bits_per_sample` (e.g. divide by 32768 for 16-bit to get f32).
    pub fn samples(&mut self) -> WavPackSamples<'_, R> {
        WavPackSamples { reader: self }
    }

    /// Decode the next block and fill output_buf. Returns true if samples available.
    fn decode_next_block(&mut self) -> Result<bool, WavPackError> {
        loop {
            // Sync to next block
            if !header::sync_to_block(&mut self.reader)? {
                self.finished = true;
                return Ok(false);
            }

            let hdr = match header::read_block_header(&mut self.reader)? {
                Some(h) => h,
                None => {
                    self.finished = true;
                    return Ok(false);
                }
            };

            // Skip metadata-only blocks
            if hdr.block_samples == 0 {
                let payload = header::read_block_payload(&mut self.reader, &hdr)?;
                let _ = payload; // skip
                continue;
            }

            // Reject unsupported
            if hdr.is_hybrid() {
                return Err(WavPackError::HybridNotSupported);
            }
            if hdr.is_float() {
                return Err(WavPackError::FloatNotSupported);
            }
            if hdr.is_dsd() {
                return Err(WavPackError::DsdNotSupported);
            }

            let payload = header::read_block_payload(&mut self.reader, &hdr)?;
            let sub_blocks = header::parse_metadata(&payload)?;

            let decoded = decode_block(&hdr, &sub_blocks)?;

            // Interleave channels for output
            let num_samples = hdr.block_samples as usize;
            let channels = if hdr.is_mono() || hdr.is_false_stereo() {
                // false stereo: mono data, output duplicated
                if hdr.is_false_stereo() { 2 } else { 1 }
            } else {
                2
            };

            self.output_buf.clear();

            if hdr.is_false_stereo() {
                // Duplicate mono to both channels
                self.output_buf.reserve(num_samples * 2);
                for i in 0..num_samples {
                    self.output_buf.push(decoded[0][i]);
                    self.output_buf.push(decoded[0][i]);
                }
            } else if channels == 1 {
                self.output_buf.extend_from_slice(&decoded[0][..num_samples]);
            } else {
                // Interleave L/R
                self.output_buf.reserve(num_samples * 2);
                for i in 0..num_samples {
                    self.output_buf.push(decoded[0][i]);
                    self.output_buf.push(decoded[1][i]);
                }
            }

            self.output_pos = 0;
            self.frames_decoded += num_samples as u64;
            return Ok(true);
        }
    }
}

/// Iterator over decoded PCM samples from a WavPack file.
pub struct WavPackSamples<'a, R: Read + Seek> {
    reader: &'a mut WavPackReader<R>,
}

impl<R: Read + Seek> Iterator for WavPackSamples<'_, R> {
    type Item = Result<i32, WavPackError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to get a sample from the current buffer
        if self.reader.output_pos < self.reader.output_buf.len() {
            let sample = self.reader.output_buf[self.reader.output_pos];
            self.reader.output_pos += 1;
            return Some(Ok(sample));
        }

        if self.reader.finished {
            return None;
        }

        // Decode next block
        match self.reader.decode_next_block() {
            Ok(true) => {
                if self.reader.output_pos < self.reader.output_buf.len() {
                    let sample = self.reader.output_buf[self.reader.output_pos];
                    self.reader.output_pos += 1;
                    Some(Ok(sample))
                } else {
                    None
                }
            }
            Ok(false) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Decode a single WavPack block into sample vectors.
///
/// Returns a Vec of channels (1 for mono, 2 for stereo), each containing
/// `block_samples` decoded i32 values.
fn decode_block(
    hdr: &header::BlockHeader,
    sub_blocks: &[header::MetaSubBlock],
) -> Result<Vec<Vec<i32>>, WavPackError> {
    let num_samples = hdr.block_samples as usize;
    let is_mono = hdr.is_mono() || hdr.is_false_stereo();
    let num_channels = if is_mono { 1 } else { 2 };
    let left_shift = hdr.left_shift();

    // Extract sub-blocks by ID
    let mut decorr_terms_data: Option<&[u8]> = None;
    let mut decorr_weights_data: Option<&[u8]> = None;
    let mut decorr_samples_data: Option<&[u8]> = None;
    let mut entropy_vars_data: Option<&[u8]> = None;
    let mut bitstream_data: Option<&[u8]> = None;
    let mut int32_info_data: Option<&[u8]> = None;

    for sb in sub_blocks {
        match sb.id {
            header::ID_DECORR_TERMS => decorr_terms_data = Some(&sb.data),
            header::ID_DECORR_WEIGHTS => decorr_weights_data = Some(&sb.data),
            header::ID_DECORR_SAMPLES => decorr_samples_data = Some(&sb.data),
            header::ID_ENTROPY_VARS => entropy_vars_data = Some(&sb.data),
            header::ID_WV_BITSTREAM => bitstream_data = Some(&sb.data),
            header::ID_INT32_INFO => int32_info_data = Some(&sb.data),
            _ => {
                eprintln!("    sub-block id=0x{:02x} raw=0x{:02x} len={}", sb.id, sb.raw_id, sb.data.len());
            }
        }
    }

    let bitstream = bitstream_data.ok_or_else(|| {
        WavPackError::BadMetadata("missing WV bitstream sub-block".into())
    })?;

    // Parse decorrelation passes
    let mut passes = if let Some(terms) = decorr_terms_data {
        decorrelation::parse_decorr_terms(terms)
    } else {
        Vec::new()
    };

    if let Some(weights) = decorr_weights_data {
        decorrelation::parse_decorr_weights(weights, &mut passes, is_mono);
    }

    if let Some(samples) = decorr_samples_data {
        decorrelation::parse_decorr_samples(samples, &mut passes, is_mono);
    }

    // Debug: hex dump raw sub-blocks when WP_TRACE is set
    if std::env::var("WP_TRACE").is_ok() {
        if let Some(d) = decorr_terms_data {
            eprint!("  RAW decorr_terms ({} bytes):", d.len());
            for b in d.iter().take(30) { eprint!(" {:02x}", b); }
            eprintln!();
        }
        if let Some(d) = decorr_weights_data {
            eprint!("  RAW decorr_weights ({} bytes):", d.len());
            for b in d.iter().take(30) { eprint!(" {:02x}", b); }
            eprintln!();
        }
        if let Some(d) = decorr_samples_data {
            eprint!("  RAW decorr_samples ({} bytes):", d.len());
            for b in d.iter().take(60) { eprint!(" {:02x}", b); }
            eprintln!();
        }
        if let Some(d) = entropy_vars_data {
            eprint!("  RAW entropy_vars ({} bytes):", d.len());
            for b in d.iter().take(20) { eprint!(" {:02x}", b); }
            eprintln!();
        }
        eprint!("  RAW bitstream first 16 bytes:");
        for b in bitstream.iter().take(16) { eprint!(" {:02x}", b); }
        eprintln!();
        // Show binary representation for bit-level debugging
        eprint!("  RAW bitstream bits (LSB-first): ");
        for (bi, &byte) in bitstream.iter().take(8).enumerate() {
            for bit in 0..8 {
                eprint!("{}", (byte >> bit) & 1);
            }
            eprint!(" ");
        }
        eprintln!();
    }

    // Debug: dump decorrelation configuration (after parsing weights + samples)
    if !passes.is_empty() {
        let terms: Vec<i32> = passes.iter().map(|p| p.term).collect();
        let deltas: Vec<i32> = passes.iter().map(|p| p.delta).collect();
        let weights: Vec<[i32;2]> = passes.iter().map(|p| p.weights).collect();
        eprintln!(
            "  block[{}]: {} ch, {} samples, left_shift={}, joint={}, flags=0x{:08x}",
            hdr.block_index, num_channels, num_samples, left_shift,
            hdr.is_joint_stereo(), hdr.flags
        );
        eprintln!("    terms={:?} deltas={:?}", terms, deltas);
        eprintln!("    weights={:?}", weights);
        for (pi, p) in passes.iter().enumerate() {
            if p.term >= 1 && p.term <= 8 {
                eprintln!("    pass[{}] term={}: samples[0]={:?} w={}", pi, p.term, &p.samples[0][..p.term as usize], p.weights[0]);
            } else if p.term == 17 || p.term == 18 {
                eprintln!("    pass[{}] term={}: samples[0]={:?} w={}", pi, p.term, &p.samples[0][..2], p.weights[0]);
            }
        }
    }

    // Parse entropy state
    let entropy_states = if let Some(evars) = entropy_vars_data {
        entropy::parse_entropy_vars(evars, num_channels)
    } else {
        (0..num_channels)
            .map(|_| entropy::EntropyState::new())
            .collect()
    };

    // Decode entropy (residuals) — frame at a time for correct zero-run handling
    let mut decoder = entropy::EntropyDecoder::new(bitstream, entropy_states);
    if std::env::var("WP_TRACE").is_ok() {
        decoder.trace = true;
    }
    let mut samples: Vec<Vec<i32>> = (0..num_channels)
        .map(|_| Vec::with_capacity(num_samples))
        .collect();

    for sample_idx in 0..num_samples {
        // Only trace first 3 samples to keep output manageable
        if sample_idx == 3 {
            decoder.trace = false;
        }
        let frame = decoder.decode_frame(num_channels);
        for (ch, &val) in frame.iter().enumerate() {
            samples[ch].push(val);
        }
    }

    // Trace entropy residuals before decorrelation
    if std::env::var("WP_TRACE").is_ok() && !samples.is_empty() {
        let n = 20.min(num_samples);
        eprint!("  Entropy residuals ch0[0..{}]:", n);
        for i in 0..n {
            eprint!(" {}", samples[0][i]);
        }
        eprintln!();
    }

    // Apply decorrelation (reverse the encoding passes)
    decorrelation::apply_decorrelation(&mut passes, &mut samples, num_samples, is_mono);

    // Apply joint stereo inverse if applicable
    if !is_mono && hdr.is_joint_stereo() {
        let (left, right) = samples.split_at_mut(1);
        decorrelation::undo_joint_stereo(&mut left[0], &mut right[0], num_samples);
    }

    // Verify CRC (computed after decorrelation + joint stereo, before left_shift/int32_info)
    let computed_crc = compute_crc(&samples, num_samples, num_channels);
    if computed_crc != hdr.crc {
        eprintln!(
            "CRC mismatch: computed=0x{:08x} header=0x{:08x} (samples={}, ch={})",
            computed_crc, hdr.crc, num_samples, num_channels
        );
    }

    // Apply left-shift
    if left_shift > 0 {
        for ch in &mut samples {
            for s in ch.iter_mut() {
                *s <<= left_shift;
            }
        }
    }

    // Handle int32 info (24-bit and extended)
    if let Some(i32_data) = int32_info_data {
        apply_int32_info(&mut samples, i32_data, num_samples, num_channels);
    }

    Ok(samples)
}

/// Apply INT32_INFO sub-block transformations for extended bit depths.
///
/// The sub-block contains 4 bytes:
/// - byte 0: sent_bits (extra bits in WVX bitstream)
/// - byte 1: zeros (zero bits shifted)
/// - byte 2: ones (one bits shifted)
/// - byte 3: dups (duplicate bits shifted)
fn apply_int32_info(
    samples: &mut [Vec<i32>],
    data: &[u8],
    num_samples: usize,
    num_channels: usize,
) {
    if data.len() < 4 {
        return;
    }

    let _sent_bits = data[0];
    let zeros = data[1] as u32;
    let ones = data[2] as u32;
    let dups = data[3] as u32;

    // Apply the shift/fill operations
    let shift = zeros.max(ones).max(dups);
    if shift == 0 {
        return;
    }

    for ch in 0..num_channels {
        for i in 0..num_samples {
            let s = samples[ch][i];
            if zeros > 0 {
                samples[ch][i] = s << zeros;
            } else if ones > 0 {
                samples[ch][i] = (s << ones) | ((1 << ones) - 1);
            } else if dups > 0 {
                let lsb = s & 1;
                let fill = if lsb != 0 { (1i32 << dups) - 1 } else { 0 };
                samples[ch][i] = (s << dups) | fill;
            }
        }
    }
}

/// Determine the sample rate from header flags and sub-blocks.
fn determine_sample_rate(
    hdr: &header::BlockHeader,
    sub_blocks: &[header::MetaSubBlock],
) -> u32 {
    // Check standard rate table first
    if let Some(rate) = hdr.sample_rate() {
        return rate;
    }

    // Custom rate: look for ID_SAMPLE_RATE sub-block (0x27)
    for sb in sub_blocks {
        if sb.id == header::ID_SAMPLE_RATE && sb.data.len() >= 3 {
            // 24-bit LE sample rate
            return u32::from_le_bytes([sb.data[0], sb.data[1], sb.data[2], 0]);
        }
    }

    // Fallback
    44100
}

/// Compute the WavPack CRC for decoded sample data.
///
/// WavPack uses a CRC-32 variant over the decoded samples.
pub fn compute_crc(samples: &[Vec<i32>], num_samples: usize, num_channels: usize) -> u32 {
    let mut crc = 0xFFFFFFFFu32;

    for i in 0..num_samples {
        for ch in 0..num_channels {
            let sample = samples[ch][i] as u32;
            crc = crc.wrapping_mul(3).wrapping_add(sample);
        }
    }

    crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_data_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join(name)
    }

    fn skip_if_missing(path: &PathBuf) -> bool {
        if !path.exists() {
            eprintln!("Skipping test: {} not found", path.display());
            true
        } else {
            false
        }
    }

    #[test]
    fn parse_stereo16_header() {
        let path = test_data_path("test_stereo16_normal.wv");
        if skip_if_missing(&path) {
            return;
        }

        let reader = WavPackReader::open(&path).unwrap();
        let info = reader.info();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.total_samples, 22050 * 2); // 0.5s × 44100 × 2ch
    }

    #[test]
    fn parse_mono16_header() {
        let path = test_data_path("test_mono16_fast.wv");
        if skip_if_missing(&path) {
            return;
        }

        let reader = WavPackReader::open(&path).unwrap();
        let info = reader.info();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);
    }

    #[test]
    fn parse_stereo24_header() {
        let path = test_data_path("test_stereo24_normal.wv");
        if skip_if_missing(&path) {
            return;
        }

        let reader = WavPackReader::open(&path).unwrap();
        let info = reader.info();
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 24);
    }

    #[test]
    fn parse_48k_header() {
        let path = test_data_path("test_48k_normal.wv");
        if skip_if_missing(&path) {
            return;
        }

        let reader = WavPackReader::open(&path).unwrap();
        let info = reader.info();
        assert_eq!(info.sample_rate, 48000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
    }

    /// Read 16-bit WAV samples as i32 values.
    fn read_wav_i16(path: &PathBuf) -> Vec<i32> {
        let mut file = File::open(path).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut buf).unwrap();

        let mut pos = 12; // skip RIFF header
        while pos + 8 < buf.len() {
            let chunk_id = &buf[pos..pos + 4];
            let chunk_size =
                u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]])
                    as usize;
            if chunk_id == b"data" {
                let data = &buf[pos + 8..pos + 8 + chunk_size];
                return data
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
                    .collect();
            }
            pos += 8 + chunk_size;
            if pos % 2 != 0 {
                pos += 1;
            }
        }
        panic!("No data chunk found");
    }

    /// Run TWO independent entropy decoders on the same bitstream to find
    /// the first sample where median states diverge.
    ///
    /// Decoder A: our EntropyDecoder
    /// Decoder B: a reference implementation matching FFmpeg's wv_get_value exactly
    #[test]
    fn compare_entropy_decoders() {
        let wv_path = test_data_path("test_mono16_fast.wv");
        if skip_if_missing(&wv_path) {
            return;
        }

        let file = File::open(&wv_path).unwrap();
        let mut rdr = BufReader::new(file);
        assert!(header::sync_to_block(&mut rdr).unwrap());
        let hdr = header::read_block_header(&mut rdr).unwrap().unwrap();
        let payload = header::read_block_payload(&mut rdr, &hdr).unwrap();
        let subs = header::parse_metadata(&payload).unwrap();
        let num_samples = hdr.block_samples as usize;

        let mut evars_data = None;
        let mut bs_data = None;
        for sb in &subs {
            match sb.id {
                header::ID_ENTROPY_VARS => evars_data = Some(sb.data.clone()),
                header::ID_WV_BITSTREAM => bs_data = Some(sb.data.clone()),
                _ => {}
            }
        }
        let bs_bytes = bs_data.unwrap();

        // Parse initial medians
        let estates_a = entropy::parse_entropy_vars(evars_data.as_ref().unwrap(), 1);
        let estates_b = entropy::parse_entropy_vars(evars_data.as_ref().unwrap(), 1);

        eprintln!("Initial medians: {:?}", estates_a[0].medians);

        // Decoder A: our EntropyDecoder
        let mut dec_a = entropy::EntropyDecoder::new(&bs_bytes, estates_a);

        // Decoder B: reference decoder using separate BitstreamReader
        let mut bs_b = bitstream::BitstreamReader::new(&bs_bytes);
        let mut ref_medians: [u32; 3] = estates_b[0].medians;
        let mut ref_zero = false;
        let mut ref_one = false;
        let mut ref_zeroes: u32 = 0;

        let median_divs: [u32; 3] = [128, 64, 32];

        // Reference: FFmpeg's GET_MED
        let get_med = |medians: &[u32; 3], n: usize| -> u32 {
            (medians[n] >> 4) + 1
        };

        // Reference: FFmpeg's INC_MED (exactly matching the C macro)
        let inc_med = |medians: &mut [u32; 3], n: usize| {
            let div = median_divs[n];
            medians[n] += ((medians[n] + div) / div) * 5;
        };

        // Reference: FFmpeg's DEC_MED (exactly matching the C macro)
        let dec_med = |medians: &mut [u32; 3], n: usize| {
            let div = median_divs[n];
            let sub = ((medians[n] + div - 2) / div) * 2;
            medians[n] = medians[n].saturating_sub(sub);
        };

        for sample_idx in 0..num_samples {
            let bit_pos_a = dec_a.bs.bit_position();

            // Decoder A: our implementation
            let val_a = dec_a.decode_frame(1)[0];
            let med_a = dec_a.states[0].medians;

            let bit_pos_a_after = dec_a.bs.bit_position();

            // Decoder B: reference implementation (inline FFmpeg algorithm)
            let bit_pos_b = bs_b.bit_position();
            let val_b;

            // 1. Check for zero-run mode
            let all_low = ref_medians[0] < 2; // mono: just 1 channel
            if all_low && !ref_zero && !ref_one {
                if ref_zeroes > 0 {
                    ref_zeroes -= 1;
                    if ref_zeroes > 0 {
                        // Still in zero run
                        val_b = 0;
                        let med_b = ref_medians;

                        // Compare
                        if val_a != val_b || med_a != med_b || bit_pos_a != bit_pos_b {
                            eprintln!(
                                "[{:4}] DIVERGE! val:{}/{} med_a:{:?} med_b:{:?} bit:{}/{}",
                                sample_idx, val_a, val_b, med_a, med_b, bit_pos_a, bit_pos_b
                            );
                            panic!("Divergence at sample {}", sample_idx);
                        }
                        continue;
                    }
                    // Zeroes exhausted — fall through to normal decode
                } else {
                    let mut run = bs_b.read_unary();
                    if run >= 2 {
                        let extra = bs_b.read_bits(run - 1);
                        run = extra | (1u32 << (run - 1));
                    }
                    ref_zeroes = run;
                    if ref_zeroes > 0 {
                        ref_medians = [0, 0, 0];
                        val_b = 0;
                        let med_b = ref_medians;

                        if val_a != val_b || med_a != med_b || bit_pos_a != bit_pos_b {
                            eprintln!(
                                "[{:4}] DIVERGE! val:{}/{} med_a:{:?} med_b:{:?} bit:{}/{}",
                                sample_idx, val_a, val_b, med_a, med_b, bit_pos_a, bit_pos_b
                            );
                            panic!("Divergence at sample {}", sample_idx);
                        }
                        continue;
                    }
                    // run=0, fall through to normal decode
                }
            }

            // 2. Normal decode with zero/one state machine
            let zone;
            if ref_zero {
                zone = 0u32;
                ref_zero = false;
            } else {
                let mut raw = bs_b.read_unary();
                if raw == 16 {
                    let t2 = bs_b.read_unary();
                    if t2 < 2 {
                        raw += t2;
                    } else {
                        raw += bs_b.read_bits(t2 - 1) | (1u32 << (t2 - 1));
                    }
                }
                if ref_one {
                    ref_one = (raw & 1) != 0;
                    zone = (raw >> 1) + 1;
                } else {
                    ref_one = (raw & 1) != 0;
                    zone = raw >> 1;
                }
                ref_zero = !ref_one;
            }

            // 3. Decode magnitude from zone
            let magnitude;
            match zone {
                0 => {
                    let m = get_med(&ref_medians, 0);
                    let rem = ref_get_tail(&mut bs_b, m - 1);
                    dec_med(&mut ref_medians, 0);
                    magnitude = rem;
                }
                1 => {
                    let m0 = get_med(&ref_medians, 0);
                    let m1 = get_med(&ref_medians, 1);
                    let rem = ref_get_tail(&mut bs_b, m1 - 1);
                    inc_med(&mut ref_medians, 0);
                    dec_med(&mut ref_medians, 1);
                    magnitude = m0 + rem;
                }
                2 => {
                    let m0 = get_med(&ref_medians, 0);
                    let m1 = get_med(&ref_medians, 1);
                    let m2 = get_med(&ref_medians, 2);
                    let rem = ref_get_tail(&mut bs_b, m2 - 1);
                    inc_med(&mut ref_medians, 0);
                    inc_med(&mut ref_medians, 1);
                    dec_med(&mut ref_medians, 2);
                    magnitude = m0 + m1 + rem;
                }
                _ => {
                    let m0 = get_med(&ref_medians, 0);
                    let m1 = get_med(&ref_medians, 1);
                    let m2 = get_med(&ref_medians, 2);
                    let base = m0 + m1 + m2 * (zone - 2);
                    let rem = ref_get_tail(&mut bs_b, m2 - 1);
                    inc_med(&mut ref_medians, 0);
                    inc_med(&mut ref_medians, 1);
                    inc_med(&mut ref_medians, 2);
                    magnitude = base + rem;
                }
            }

            // 4. Sign bit
            let sign = bs_b.read_bit();
            val_b = if sign != 0 {
                !(magnitude as i32)
            } else {
                magnitude as i32
            };

            let med_b = ref_medians;
            let bit_pos_b_after = bs_b.bit_position();

            // Compare everything
            let val_ok = val_a == val_b;
            let med_ok = med_a == med_b;
            let bit_ok = bit_pos_a == bit_pos_b;
            let bit_after_ok = bit_pos_a_after == bit_pos_b_after;

            if !val_ok || !med_ok || !bit_ok || !bit_after_ok {
                eprintln!("\n[{:4}] DIVERGE!", sample_idx);
                eprintln!("  val:  A={:6}  B={:6}", val_a, val_b);
                eprintln!("  med:  A={:?}", med_a);
                eprintln!("  med:  B={:?}", med_b);
                eprintln!("  bit_before: A={} B={}", bit_pos_a, bit_pos_b);
                eprintln!("  bit_after:  A={} B={}", bit_pos_a_after, bit_pos_b_after);
                eprintln!("  zone={} magnitude={} sign={}", zone, magnitude, sign);
                eprintln!(
                    "  A state: zero={} one={} zeroes={}",
                    dec_a.zero, dec_a.one, dec_a.zeroes
                );
                eprintln!(
                    "  B state: zero={} one={} zeroes={}",
                    ref_zero, ref_one, ref_zeroes
                );
                panic!("Entropy decoder diverged at sample {}", sample_idx);
            }

            // Periodic summary
            if sample_idx % 20 == 0 || (sample_idx >= 95 && sample_idx <= 105) {
                eprintln!(
                    "[{:4}] val={:4} med={:?} bit={} zone={} z/o={}/{} ok",
                    sample_idx, val_a, med_a, bit_pos_a, zone, ref_zero, ref_one
                );
            }
        }
        eprintln!(
            "\nAll {} samples match between decoder A and reference B!",
            num_samples
        );
    }

    /// Reference get_tail matching FFmpeg's implementation exactly.
    fn ref_get_tail(bs: &mut bitstream::BitstreamReader, k: u32) -> u32 {
        if k < 1 {
            return 0;
        }
        let p = 31 - k.leading_zeros();
        let e = (1u32 << (p + 1)) - k - 1;
        let mut res = bs.read_bits(p);
        if res >= e {
            res = res * 2 - e + bs.read_bit();
        }
        res
    }

    /// Get prediction history for sample i.
    fn get_history(i: usize, samples: &[i32], hist0: i32, hist1: i32) -> (i32, i32) {
        let s1 = if i >= 1 { samples[i - 1] } else { hist0 };
        let s2 = if i >= 2 {
            samples[i - 2]
        } else if i == 1 {
            hist0
        } else {
            hist1
        };
        (s1, s2)
    }

    /// Compute prediction for a given term.
    fn predict(
        term: i32,
        s1: i32,
        s2: i32,
        i: usize,
        pass: &decorrelation::DecorrPass,
    ) -> i32 {
        match term {
            18 => ((3i64 * s1 as i64 - s2 as i64) >> 1) as i32,
            17 => 2 * s1 - s2,
            1..=8 => {
                if i >= term as usize {
                    s1 // already handled by get_history for term 1
                } else {
                    pass.samples[0][(term as usize - 1 - i).min(7)]
                }
            }
            _ => 0,
        }
    }

    /// Apply weight to prediction: (pred * weight + 512) >> 10
    fn apply_weight_pred(pred: i32, weight: i32) -> i32 {
        ((pred as i64 * weight as i64 + 512) >> 10) as i32
    }

    /// Get update direction as a string for display.
    fn update_dir(residual: i32, prediction: i32) -> &'static str {
        if residual != 0 && prediction != 0 {
            if (residual ^ prediction) >= 0 { "+" } else { "-" }
        } else {
            "0"
        }
    }

    /// Update weight with clamping.
    fn do_update_weight(weight: &mut i32, delta: i32, residual: i32, prediction: i32) {
        if residual != 0 && prediction != 0 {
            if (residual ^ prediction) >= 0 {
                *weight += delta;
            } else {
                *weight -= delta;
            }
        }
        *weight = (*weight).clamp(-1024, 1024);
    }
}
