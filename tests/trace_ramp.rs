/// Trace decode of 16-sample ramp [0,100,200,...,1500] to understand WavPack encoding.
use std::io::Read;
use std::path::PathBuf;

fn test_data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(name)
}

#[test]
fn trace_ramp16() {
    let wv_path = test_data_path("test_ramp16.wv");
    if !wv_path.exists() {
        eprintln!("Skipping: test data not found");
        return;
    }

    let mut file = std::fs::File::open(&wv_path).unwrap();
    let mut all = Vec::new();
    file.read_to_end(&mut all).unwrap();

    assert_eq!(&all[0..4], b"wvpk");
    let block_size = u32::from_le_bytes([all[4], all[5], all[6], all[7]]) as usize;
    let flags = u32::from_le_bytes([all[24], all[25], all[26], all[27]]);
    let block_samples = u32::from_le_bytes([all[20], all[21], all[22], all[23]]);
    let crc = u32::from_le_bytes([all[28], all[29], all[30], all[31]]);

    println!("=== BLOCK HEADER ===");
    println!("block_size={block_size}, block_samples={block_samples}");
    println!("flags=0x{flags:08x}, crc=0x{crc:08x}");
    println!("mono={}, joint={}, left_shift={}, bps={}",
        (flags >> 2) & 1, (flags >> 4) & 1,
        (flags >> 13) & 0x1f, ((flags & 3) + 1) * 8);

    let payload = &all[32..32 + block_size - 24];

    // Parse ALL sub-blocks with full detail
    println!("\n=== ALL SUB-BLOCKS ===");
    let mut pos = 0;
    let mut decorr_terms_data: &[u8] = &[];
    let mut decorr_weights_data: &[u8] = &[];
    let mut decorr_samples_data: &[u8] = &[];
    let mut entropy_vars_data: &[u8] = &[];
    let mut bitstream_data_slice: &[u8] = &[];

    while pos < payload.len() {
        let raw_id = payload[pos];
        pos += 1;
        let id = raw_id & 0x3f;
        let is_odd = raw_id & 0x40 != 0;
        let is_large = raw_id & 0x80 != 0;

        let size_words: u32 = if is_large {
            let w = u32::from_le_bytes([payload[pos], payload[pos+1], payload[pos+2], 0]);
            pos += 3;
            w
        } else {
            let w = payload[pos] as u32;
            pos += 1;
            w
        };

        let size_bytes = (size_words * 2) as usize;
        let data_len = if is_odd && size_bytes > 0 { size_bytes - 1 } else { size_bytes };
        let data = &payload[pos..pos + data_len];

        let name = match id {
            0x02 => "DECORR_TERMS",
            0x03 => "DECORR_WEIGHTS",
            0x04 => "DECORR_SAMPLES",
            0x05 => "ENTROPY_VARS",
            0x0a => "WV_BITSTREAM",
            0x21 => "RIFF_HEADER",
            0x25 => "CONFIG",
            0x2f => "CHECKSUM",
            _ => "other",
        };

        print!("  id=0x{id:02x} ({name}), raw=0x{raw_id:02x}, {data_len} bytes: ");
        if data_len <= 16 {
            for b in data { print!("{b:02x} "); }
        } else {
            for b in &data[..16] { print!("{b:02x} "); }
            print!("...");
        }
        println!();

        match id {
            0x02 => decorr_terms_data = data,
            0x03 => decorr_weights_data = data,
            0x04 => decorr_samples_data = data,
            0x05 => entropy_vars_data = data,
            0x0a => bitstream_data_slice = data,
            _ => {}
        }

        pos += size_bytes;
    }

    // Parse decorrelation
    println!("\n=== DECORRELATION ===");
    let mut terms: Vec<(i32, i32)> = Vec::new();
    for &b in decorr_terms_data {
        let term = (b & 0x1f) as i32 - 5;
        let delta = (b >> 5) as i32;
        terms.push((term, delta));
        println!("  term={term}, delta={delta}");
    }

    let mut weights: Vec<i32> = Vec::new();
    for &b in decorr_weights_data {
        let w = b as i8;
        let mut rw = (w as i32) << 3;
        if rw > 0 { rw += (rw + 64) >> 7; }
        else if rw < 0 { rw -= ((-rw) + 64) >> 7; }
        weights.push(rw);
        println!("  weight stored={w}, restored={rw}");
    }

    let mut history: Vec<i32> = Vec::new();
    {
        let mut sp = 0;
        while sp + 2 <= decorr_samples_data.len() {
            let packed = i16::from_le_bytes([decorr_samples_data[sp], decorr_samples_data[sp + 1]]);
            let restored = restore_sample_log2(packed);
            history.push(restored);
            println!("  history: packed=0x{:04x}({packed}) → {restored}", packed as u16);
            sp += 2;
        }
    }

    // Parse entropy vars
    println!("\n=== ENTROPY VARS ===");
    let mut medians = [0u32; 3];
    for i in 0..3 {
        let p = i * 2;
        if p + 2 <= entropy_vars_data.len() {
            let packed = u16::from_le_bytes([entropy_vars_data[p], entropy_vars_data[p + 1]]);
            medians[i] = restore_median_log2(packed);
            println!("  median[{i}]: packed=0x{packed:04x}, internal={}, effective={}",
                medians[i], (medians[i] >> 4) + 1);
        }
    }

    // Raw bitstream
    println!("\n=== RAW BITSTREAM ({} bytes) ===", bitstream_data_slice.len());
    for i in 0..bitstream_data_slice.len().min(32) {
        let b = bitstream_data_slice[i];
        if i % 8 == 0 { print!("  "); }
        print!("{b:02x} ");
        if i % 8 == 7 { println!(); }
    }
    println!();

    println!("  First 64 bits (LSB-first):");
    print!("  ");
    for i in 0..8.min(bitstream_data_slice.len()) {
        let b = bitstream_data_slice[i];
        for bit in 0..8 {
            print!("{}", (b >> bit) & 1);
        }
        print!(" ");
    }
    println!();

    // Try decoding with stored initial state
    println!("\n=== DECODE WITH STORED STATE ===");
    decode_and_print(bitstream_data_slice, &medians, &terms, &weights, &history, 16);

    // Try decoding with ZERO initial state
    println!("\n=== DECODE WITH ZERO STATE ===");
    let zero_medians = [0u32; 3];
    let zero_weights: Vec<i32> = vec![0; terms.len()];
    let zero_history: Vec<i32> = vec![0; history.len()];
    decode_and_print(bitstream_data_slice, &zero_medians, &terms, &zero_weights, &zero_history, 16);

    // Reference output
    let ref_path = test_data_path("ref_ramp16.wav");
    if ref_path.exists() {
        let ref_samples = read_wav_samples_i16(&ref_path);
        println!("\n=== REFERENCE ===");
        for (i, &s) in ref_samples.iter().enumerate() {
            println!("  ref[{i}] = {s}");
        }
    }
}

fn decode_and_print(
    bs_data: &[u8],
    initial_medians: &[u32; 3],
    terms: &[(i32, i32)],
    weights: &[i32],
    history: &[i32],
    num_samples: usize,
) {
    let mut medians = *initial_medians;
    let mut bs = TraceBitstream::new(bs_data);
    let mut zero_run = 0u32;
    let mut residuals = Vec::new();

    // Entropy decode
    for i in 0..num_samples {
        if zero_run > 0 {
            zero_run -= 1;
            residuals.push(0);
            println!("  residual[{i}] = 0 (zero-run)");
            continue;
        }

        let all_zero = medians.iter().all(|&m| m == 0);
        if all_zero {
            let run = bs.read_exp_golomb();
            if run > 0 {
                zero_run = run - 1;
                residuals.push(0);
                println!("  residual[{i}] = 0 (start zero-run len={run})");
                continue;
            }
            medians[0] = 1 << 4;
            println!("  (exit zero mode, median[0]←16)");
        }

        let mut zone = bs.read_unary();
        if zone >= 16 {
            let extra = bs.read_exp_golomb();
            println!("  ESCAPE: zone=16+{extra}");
            zone += extra;
        }

        let m0 = (medians[0] >> 4) + 1;
        let m1 = (medians[1] >> 4) + 1;
        let m2 = (medians[2] >> 4) + 1;

        let magnitude;
        match zone {
            0 => {
                let rem = bs.read_golomb_remainder(m0);
                magnitude = rem;
                let div = 128u32;
                let sub = ((medians[0] + div - 2) / div) * 2;
                medians[0] = medians[0].saturating_sub(sub);
            }
            1 => {
                let rem = bs.read_golomb_remainder(m1);
                magnitude = m0 + rem;
                medians[0] += ((medians[0] + 128) / 128) * 5;
                let sub = ((medians[1] + 30) / 32) * 2;
                medians[1] = medians[1].saturating_sub(sub);
            }
            _ => {
                let rem = bs.read_golomb_remainder(m2);
                let base = m0 + m1 + m2 * (zone - 2);
                magnitude = base + rem;
                medians[0] += ((medians[0] + 128) / 128) * 5;
                medians[1] += ((medians[1] + 32) / 32) * 5;
                if zone > 2 {
                    medians[2] += ((medians[2] + 128) / 128) * 5;
                } else {
                    let sub = ((medians[2] + 126) / 128) * 2;
                    medians[2] = medians[2].saturating_sub(sub);
                }
            }
        }

        let signed_val = if magnitude > 0 {
            let sign = bs.read_bit();
            if sign != 0 { -(magnitude as i32) - 1 } else { magnitude as i32 }
        } else {
            0
        };

        residuals.push(signed_val);
        println!("  residual[{i}] = {signed_val} (zone={zone}, mag={magnitude})");
    }

    // Apply decorrelation (reverse pass order)
    let mut decoded = residuals.clone();
    for pass_i in (0..terms.len()).rev() {
        let (term, delta) = terms[pass_i];
        let mut weight = if pass_i < weights.len() { weights[pass_i] } else { 0 };

        // Compute history offset
        let mut hist_off = 0;
        for pi in 0..pass_i {
            let (t, _) = terms[pi];
            if t > 8 { hist_off += 2; } else if t >= 1 { hist_off += t as usize; }
        }
        let (t, _) = terms[pass_i];
        let hcount = if t > 8 { 2 } else if t >= 1 { t as usize } else { 0 };
        let hist: Vec<i32> = (0..hcount)
            .map(|i| if hist_off + i < history.len() { history[hist_off + i] } else { 0 })
            .collect();

        for i in 0..decoded.len() {
            let prediction = match term {
                1..=8 => {
                    if i >= term as usize { decoded[i - term as usize] }
                    else if (term as usize - 1 - i) < hist.len() { hist[term as usize - 1 - i] }
                    else { 0 }
                }
                17 => {
                    let s1 = if i >= 1 { decoded[i-1] } else if !hist.is_empty() { hist[0] } else { 0 };
                    let s2 = if i >= 2 { decoded[i-2] } else if i == 1 { if !hist.is_empty() { hist[0] } else { 0 } } else { if hist.len() > 1 { hist[1] } else { 0 } };
                    2 * s1 - s2
                }
                18 => {
                    let s1 = if i >= 1 { decoded[i-1] } else if !hist.is_empty() { hist[0] } else { 0 };
                    let s2 = if i >= 2 { decoded[i-2] } else if i == 1 { if !hist.is_empty() { hist[0] } else { 0 } } else { if hist.len() > 1 { hist[1] } else { 0 } };
                    ((3i64 * s1 as i64 - s2 as i64) >> 1) as i32
                }
                _ => 0,
            };

            let residual = decoded[i];
            let applied = ((prediction as i64 * weight as i64 + 512) >> 10) as i32;
            decoded[i] = residual + applied;

            if residual != 0 && prediction != 0 {
                if (residual ^ prediction) >= 0 { weight += delta; } else { weight -= delta; }
            }
            weight = weight.clamp(-1024, 1024);
        }
    }

    println!("  Final decoded:");
    for (i, &d) in decoded.iter().enumerate() {
        println!("    [{i}] = {d}");
    }
}

struct TraceBitstream<'a> {
    data: &'a [u8],
    byte_pos: usize,
    accum: u64,
    bits_left: u32,
}

impl<'a> TraceBitstream<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, byte_pos: 0, accum: 0, bits_left: 0 }
    }

    fn refill(&mut self) {
        while self.bits_left <= 56 && self.byte_pos < self.data.len() {
            self.accum |= (self.data[self.byte_pos] as u64) << self.bits_left;
            self.byte_pos += 1;
            self.bits_left += 8;
        }
    }

    fn read_bits(&mut self, n: u32) -> u32 {
        if n == 0 { return 0; }
        self.refill();
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let val = (self.accum as u32) & mask;
        self.accum >>= n;
        self.bits_left = self.bits_left.saturating_sub(n);
        val
    }

    fn read_bit(&mut self) -> u32 { self.read_bits(1) }

    fn read_unary(&mut self) -> u32 {
        let mut count = 0u32;
        loop {
            self.refill();
            if self.bits_left == 0 { return count; }
            let bit = (self.accum & 1) as u32;
            if bit == 0 {
                self.accum >>= 1;
                self.bits_left = self.bits_left.saturating_sub(1);
                return count;
            }
            self.accum >>= 1;
            self.bits_left = self.bits_left.saturating_sub(1);
            count += 1;
            if count >= 16 { return count; }
        }
    }

    fn read_exp_golomb(&mut self) -> u32 {
        let mut prefix = 0u32;
        loop {
            self.refill();
            let bit = (self.accum & 1) as u32;
            self.accum >>= 1;
            self.bits_left = self.bits_left.saturating_sub(1);
            if bit == 0 { break; }
            prefix += 1;
        }
        if prefix < 2 { prefix }
        else {
            let suffix = self.read_bits(prefix - 1);
            (1 << (prefix - 1)) | suffix
        }
    }

    fn read_golomb_remainder(&mut self, m: u32) -> u32 {
        if m <= 1 { return 0; }
        let k = 31 - m.leading_zeros();
        let cutoff = (1u32 << (k + 1)) - m;
        let bits = self.read_bits(k);
        if bits < cutoff { bits }
        else {
            let extra = self.read_bit();
            (bits - cutoff) * 2 + extra + cutoff
        }
    }
}

fn restore_sample_log2(raw: i16) -> i32 {
    if raw == 0 { return 0; }
    if raw < 0 { return -restore_sample_log2_unsigned((-(raw as i32)) as u16); }
    restore_sample_log2_unsigned(raw as u16)
}

fn restore_sample_log2_unsigned(packed: u16) -> i32 {
    if packed == 0 { return 0; }
    let mantissa = (packed & 0xFF) as u32;
    let shift = (packed >> 8) as u32;
    let value = mantissa | 0x100;
    if shift <= 9 { (value >> (9 - shift)) as i32 }
    else { (value << (shift - 9)) as i32 }
}

fn restore_median_log2(packed: u16) -> u32 {
    if packed == 0 { return 0; }
    let mantissa = (packed & 0xFF) as u32;
    let shift = (packed >> 8) as u32;
    let value = mantissa | 0x100;
    let restored = if shift <= 9 { value >> (9 - shift) } else { value << (shift - 9) };
    restored << 4
}

fn read_wav_samples_i16(path: &PathBuf) -> Vec<i32> {
    let mut file = std::fs::File::open(path).unwrap();
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).unwrap();
    let mut pos = 12;
    while pos + 8 < buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([buf[pos+4], buf[pos+5], buf[pos+6], buf[pos+7]]) as usize;
        if chunk_id == b"data" {
            let data = &buf[pos + 8..pos + 8 + chunk_size];
            return data.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as i32).collect();
        }
        pos += 8 + chunk_size;
        if pos % 2 != 0 { pos += 1; }
    }
    panic!("No data chunk found");
}
