/// Quick verification tests for various test files.
use std::io::Read;
use std::path::PathBuf;
use wavpack_rs::WavPackReader;

fn test_data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data").join(name)
}

fn read_wav_i16(path: &PathBuf) -> Vec<i32> {
    let mut buf = Vec::new();
    std::fs::File::open(path).unwrap().read_to_end(&mut buf).unwrap();
    let mut pos = 12;
    while pos + 8 < buf.len() {
        let id = &buf[pos..pos + 4];
        let sz = u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]) as usize;
        if id == b"data" {
            let data = &buf[pos + 8..pos + 8 + sz];
            return data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i32).collect();
        }
        pos += 8 + sz;
        if pos % 2 != 0 { pos += 1; }
    }
    panic!("No data chunk");
}

#[test]
fn verify_silence() {
    let wv = test_data_path("test_silence.wv");
    let rf = test_data_path("ref_silence.wav");
    if !wv.exists() || !rf.exists() {
        eprintln!("Skipping: test data not found");
        return;
    }
    let mut reader = WavPackReader::open(&wv).unwrap();
    let info = reader.info();
    eprintln!("silence: {}ch, {}Hz, {}bit, {} total", info.channels, info.sample_rate, info.bits_per_sample, info.total_samples);
    let decoded: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
    let reference = read_wav_i16(&rf);
    eprintln!("decoded={} ref={}", decoded.len(), reference.len());
    assert_eq!(decoded.len(), reference.len(), "length mismatch");
    let mut mismatches = 0;
    for (i, (&d, &r)) in decoded.iter().zip(&reference).enumerate() {
        if d != r {
            if mismatches < 10 { eprintln!("  [{i}] decoded={d} ref={r}"); }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} sample mismatches");
}

#[test]
fn verify_ramp() {
    let wv = test_data_path("test_ramp16.wv");
    let rf = test_data_path("ref_ramp16.wav");
    if !wv.exists() || !rf.exists() {
        eprintln!("Skipping: test data not found");
        return;
    }
    let mut reader = WavPackReader::open(&wv).unwrap();
    let info = reader.info();
    eprintln!("ramp: {}ch, {}Hz, {}bit, {} total", info.channels, info.sample_rate, info.bits_per_sample, info.total_samples);
    let decoded: Vec<i32> = reader.samples().collect::<Result<_, _>>().unwrap();
    let reference = read_wav_i16(&rf);
    eprintln!("decoded={} ref={}", decoded.len(), reference.len());
    assert_eq!(decoded.len(), reference.len(), "length mismatch");
    for (i, (&d, &r)) in decoded.iter().zip(&reference).enumerate() {
        if d != r {
            eprintln!("  [{i}] decoded={d} ref={r}");
        }
        assert_eq!(d, r, "sample {i} mismatch");
    }
}
