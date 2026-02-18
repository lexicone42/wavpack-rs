use std::path::PathBuf;
use std::process::Command;
use wavpack_rs::WavPackReader;

fn test_data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(name)
}

/// Read WAV samples as i32 values, auto-detecting bit depth from the fmt chunk.
fn read_wav_samples(path: &PathBuf) -> Vec<i32> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();

    // Parse fmt chunk for bits_per_sample
    let mut bits_per_sample = 16u16;
    let mut pos = 12;
    while pos + 8 < buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;
        if chunk_id == b"fmt " && chunk_size >= 16 {
            bits_per_sample = u16::from_le_bytes([buf[pos + 22], buf[pos + 23]]);
        }
        if chunk_id == b"data" {
            let data = &buf[pos + 8..pos + 8 + chunk_size];
            return match bits_per_sample {
                16 => data
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
                    .collect(),
                24 => data
                    .chunks_exact(3)
                    .map(|c| {
                        let val = (c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16);
                        if val & 0x800000 != 0 { val | !0xFFFFFF } else { val }
                    })
                    .collect(),
                _ => panic!("Unsupported bit depth: {}", bits_per_sample),
            };
        }
        pos += 8 + chunk_size;
        if pos % 2 != 0 {
            pos += 1;
        }
    }
    panic!("No data chunk found in {}", path.display());
}

/// Decode a .wv file with wvunpack and return WAV samples.
/// Returns None if wvunpack is not available.
fn decode_with_wvunpack(wv_path: &PathBuf) -> Option<Vec<i32>> {
    let tmp_wav = wv_path.with_extension("wvunpack_ref.wav");
    let output = Command::new("wvunpack")
        .arg("-y")
        .arg("-q")
        .arg(wv_path)
        .arg("-o")
        .arg(&tmp_wav)
        .output()
        .ok()?;
    if !output.status.success() {
        eprintln!("wvunpack failed: {}", String::from_utf8_lossy(&output.stderr));
        return None;
    }
    let samples = read_wav_samples(&tmp_wav);
    let _ = std::fs::remove_file(&tmp_wav);
    Some(samples)
}

/// Compare decoded samples against reference, panicking on mismatch.
fn assert_bit_exact(decoded: &[i32], reference: &[i32], label: &str) {
    assert_eq!(
        decoded.len(),
        reference.len(),
        "{}: sample count mismatch: decoded={} ref={}",
        label,
        decoded.len(),
        reference.len()
    );

    let mut mismatches = 0;
    let mut first_mismatch = usize::MAX;
    for (i, (d, r)) in decoded.iter().zip(reference).enumerate() {
        if d != r {
            if mismatches < 5 {
                eprintln!(
                    "{}: sample {} mismatch: decoded={} ref={} diff={}",
                    label,
                    i,
                    d,
                    r,
                    d - r
                );
            }
            if first_mismatch == usize::MAX {
                first_mismatch = i;
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "{}: {} sample mismatches out of {} (first at sample {})",
        label,
        mismatches,
        decoded.len(),
        first_mismatch
    );
}

/// Generic test: decode .wv with our decoder, compare to reference WAV.
fn test_decode(wv_name: &str, ref_name: &str) {
    let wv_path = test_data_path(wv_name);
    let ref_path = test_data_path(ref_name);
    if !wv_path.exists() || !ref_path.exists() {
        eprintln!("Skipping {}: test data not found", wv_name);
        return;
    }

    let mut reader = WavPackReader::open(&wv_path).unwrap();
    let decoded: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
    let reference = read_wav_samples(&ref_path);
    assert_bit_exact(&decoded, &reference, wv_name);
}

/// Generic test: decode .wv with our decoder AND wvunpack, compare both.
fn test_decode_vs_wvunpack(wv_name: &str, ref_name: &str) {
    let wv_path = test_data_path(wv_name);
    let ref_path = test_data_path(ref_name);
    if !wv_path.exists() || !ref_path.exists() {
        eprintln!("Skipping {}: test data not found", wv_name);
        return;
    }

    let mut reader = WavPackReader::open(&wv_path).unwrap();
    let decoded: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
    let reference = read_wav_samples(&ref_path);
    assert_bit_exact(&decoded, &reference, &format!("{} vs WAV", wv_name));

    // Also verify against wvunpack if available
    if let Some(wvunpack_ref) = decode_with_wvunpack(&wv_path) {
        assert_bit_exact(&decoded, &wvunpack_ref, &format!("{} vs wvunpack", wv_name));
        eprintln!("  {} verified against wvunpack ({} samples)", wv_name, decoded.len());
    }
}

// ── Synthetic tests: mono 16-bit ──

#[test]
fn decode_mono16_fast() {
    test_decode_vs_wvunpack("test_mono16_fast.wv", "ref_mono16.wav");
}

#[test]
fn decode_mono16_normal() {
    test_decode_vs_wvunpack("test_mono16_normal.wv", "ref_mono16.wav");
}

#[test]
fn decode_mono16_high() {
    test_decode_vs_wvunpack("test_mono16_high.wv", "ref_mono16.wav");
}

// ── Synthetic tests: stereo 16-bit ──

#[test]
fn decode_stereo16_fast() {
    test_decode_vs_wvunpack("test_stereo16_fast.wv", "ref_stereo16.wav");
}

#[test]
fn decode_stereo16_normal() {
    test_decode_vs_wvunpack("test_stereo16_normal.wv", "ref_stereo16.wav");
}

#[test]
fn decode_stereo16_high() {
    test_decode_vs_wvunpack("test_stereo16_high.wv", "ref_stereo16.wav");
}

#[test]
fn decode_stereo16_vhigh() {
    test_decode_vs_wvunpack("test_stereo16_vhigh.wv", "ref_stereo16.wav");
}

// ── Synthetic tests: stereo 24-bit ──

#[test]
fn decode_stereo24_fast() {
    test_decode_vs_wvunpack("test_stereo24_fast.wv", "ref_stereo24.wav");
}

#[test]
fn decode_stereo24_normal() {
    test_decode_vs_wvunpack("test_stereo24_normal.wv", "ref_stereo24.wav");
}

#[test]
fn decode_stereo24_high() {
    test_decode_vs_wvunpack("test_stereo24_high.wv", "ref_stereo24.wav");
}

// ── Edge cases ──

#[test]
fn decode_silence() {
    test_decode("test_silence.wv", "ref_silence.wav");
}

#[test]
fn decode_ramp() {
    test_decode("test_ramp16.wv", "test_ramp16.wav");
}

// ── Real-world music tests ──

#[test]
fn decode_realworld_gd_fast() {
    test_decode_vs_wvunpack("test_realworld_gd_fast.wv", "ref_realworld_gd.wav");
}

#[test]
fn decode_realworld_gd_normal() {
    test_decode_vs_wvunpack("test_realworld_gd_normal.wv", "ref_realworld_gd.wav");
}

#[test]
fn decode_realworld_gd_high() {
    test_decode_vs_wvunpack("test_realworld_gd_high.wv", "ref_realworld_gd.wav");
}

#[test]
fn decode_realworld_reich_normal() {
    test_decode_vs_wvunpack("test_realworld_reich_normal.wv", "ref_realworld_reich.wav");
}

#[test]
fn decode_realworld_reich_high() {
    test_decode_vs_wvunpack("test_realworld_reich_high.wv", "ref_realworld_reich.wav");
}

// ── Downloaded sample from filesamples.com ──

#[test]
fn decode_filesamples_wv() {
    let wv_path = PathBuf::from("/tmp/sample1.wv");
    if !wv_path.exists() {
        eprintln!("Skipping: /tmp/sample1.wv not found (download from filesamples.com)");
        return;
    }

    let mut reader = WavPackReader::open(&wv_path).unwrap();
    let info = reader.info();
    eprintln!(
        "filesamples.com sample: {}ch, {}Hz, {}bit, {} total samples",
        info.channels, info.sample_rate, info.bits_per_sample, info.total_samples
    );

    let decoded: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
    eprintln!("Decoded {} samples", decoded.len());

    // Verify against wvunpack
    if let Some(wvunpack_ref) = decode_with_wvunpack(&wv_path) {
        assert_bit_exact(&decoded, &wvunpack_ref, "filesamples sample1.wv vs wvunpack");
        eprintln!(
            "  filesamples sample1.wv verified against wvunpack ({} samples)",
            decoded.len()
        );
    } else {
        panic!("wvunpack not available for verification");
    }
}
