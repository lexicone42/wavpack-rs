/// Trace decode of first N samples through entropy → decorrelation → output.
use std::io::Read;
use std::path::PathBuf;

fn test_data_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(name)
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
        if bits < cutoff {
            bits
        } else {
            let extra = self.read_bit();
            (bits - cutoff) * 2 + extra + cutoff
        }
    }
}

fn restore_sample_log2(raw: i16) -> i32 {
    if raw == 0 { return 0; }
    if raw < 0 {
        return -restore_sample_log2_unsigned((-(raw as i32)) as u16);
    }
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

#[test]
fn trace_mono16_fast() {
    let wv_path = test_data_path("test_mono16_fast.wv");
    if !wv_path.exists() {
        eprintln!("Skipping: test data not found");
        return;
    }

    let mut file = std::fs::File::open(&wv_path).unwrap();
    let mut all = Vec::new();
    file.read_to_end(&mut all).unwrap();

    // Parse block header
    assert_eq!(&all[0..4], b"wvpk");
    let block_size = u32::from_le_bytes([all[4], all[5], all[6], all[7]]) as usize;
    let flags = u32::from_le_bytes([all[24], all[25], all[26], all[27]]);
    let block_samples = u32::from_le_bytes([all[20], all[21], all[22], all[23]]);
    let crc = u32::from_le_bytes([all[28], all[29], all[30], all[31]]);

    println!("=== BLOCK HEADER ===");
    println!("block_size={block_size}, block_samples={block_samples}, flags=0x{flags:08x}, crc=0x{crc:08x}");

    // Parse sub-blocks
    let payload = &all[32..32 + block_size - 24];

    // Raw hex dump of first 40 bytes of payload
    println!("\n=== RAW PAYLOAD (first 40 bytes) ===");
    for i in 0..40.min(payload.len()) {
        if i % 16 == 0 { print!("  {:04x}: ", i); }
        print!("{:02x} ", payload[i]);
        if i % 16 == 15 { println!(); }
    }
    println!();

    // Parse sub-blocks with offset tracking
    println!("\n=== SUB-BLOCK BOUNDARIES ===");
    {
        let mut p = 0;
        while p < payload.len() {
            let raw_id = payload[p];
            let id = raw_id & 0x3f;
            let is_odd = raw_id & 0x40 != 0;
            let is_large = raw_id & 0x80 != 0;
            p += 1;

            let size_words: u32 = if is_large {
                let w = u32::from_le_bytes([payload[p], payload[p+1], payload[p+2], 0]);
                p += 3;
                w
            } else {
                let w = payload[p] as u32;
                p += 1;
                w
            };
            let size_bytes = (size_words * 2) as usize;
            let data_len = if is_odd && size_bytes > 0 { size_bytes - 1 } else { size_bytes };

            let name = match id {
                0x02 => "DECORR_TERMS",
                0x03 => "DECORR_WEIGHTS",
                0x04 => "DECORR_SAMPLES",
                0x05 => "ENTROPY_VARS",
                0x0a => "WV_BITSTREAM",
                0x09 => "INT32_INFO",
                _ => "other",
            };
            println!("  offset={}, raw_id=0x{raw_id:02x}(id=0x{id:02x},odd={is_odd},large={is_large}), size_words={size_words}, size_bytes={size_bytes}, data_len={data_len}, data_at={p}  [{name}]", p - if is_large { 4 } else { 2 });

            p += size_bytes;
        }
    }

    let mut pos = 0;
    let mut entropy_data: &[u8] = &[];
    let mut bitstream_data: &[u8] = &[];
    let mut terms_data: &[u8] = &[];
    let mut weights_data: &[u8] = &[];
    let mut samples_data: &[u8] = &[];

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
        let data_start = pos;
        let data = &payload[data_start..data_start + data_len];

        match id {
            0x02 => terms_data = data,
            0x03 => weights_data = data,
            0x04 => samples_data = data,
            0x05 => entropy_data = data,
            0x0a => bitstream_data = data,
            _ => {}
        }
        pos += size_bytes;
    }

    // Parse decorrelation
    println!("\n=== DECORRELATION PASSES ===");
    let mut passes: Vec<(i32, i32)> = Vec::new(); // (term, delta)
    for (i, &b) in terms_data.iter().enumerate() {
        let term = (b & 0x1f) as i32 - 5;
        let delta = (b >> 5) as i32;
        passes.push((term, delta));
        println!("  pass[{i}]: term={term}, delta={delta}");
    }

    println!("\n  Weights:");
    let mut restored_weights: Vec<i32> = Vec::new();
    for (i, &b) in weights_data.iter().enumerate() {
        let w = b as i8;
        let mut rw = (w as i32) << 3;
        if rw > 0 { rw += (rw + 64) >> 7; }
        else if rw < 0 { rw -= ((-rw) + 64) >> 7; }
        restored_weights.push(rw);
        println!("    pass[{i}]: stored={w}, restored={rw}");
    }

    println!("\n  History samples:");
    let mut sp = 0;
    let mut sample_pairs: Vec<i32> = Vec::new();
    while sp + 2 <= samples_data.len() {
        let packed = i16::from_le_bytes([samples_data[sp], samples_data[sp + 1]]);
        let restored = restore_sample_log2(packed);
        sample_pairs.push(restored);
        println!("    [{sp}]: packed=0x{:04x} ({packed}) → {restored}", packed as u16);
        sp += 2;
    }

    // Parse entropy vars
    println!("\n=== ENTROPY VARS ===");
    let mut medians = [0u32; 3];
    for i in 0..3 {
        let p = i * 2;
        if p + 2 <= entropy_data.len() {
            let packed = u16::from_le_bytes([entropy_data[p], entropy_data[p + 1]]);
            medians[i] = restore_median_log2(packed);
            let eff = (medians[i] >> 4) + 1;
            println!("  median[{i}]: packed=0x{packed:04x}, internal={}, effective={eff}", medians[i]);
        }
    }

    // Compute expected residual for sample 0 (ref=0) by forward decorrelation
    println!("\n=== EXPECTED RESIDUALS (forward encoding of ref samples) ===");
    let ref_path2 = test_data_path("ref_mono16.wav");
    if ref_path2.exists() {
        let ref_samps = read_wav_samples_i16(&ref_path2);
        // Forward encoding: apply passes in forward order
        // Pass[0]: term=18, weight=976, history=[-128, -203]
        // Pass[1]: term=17, weight=1016, history=[0, 0]
        let mut encoded: Vec<i32> = Vec::new();
        // After pass[0]:
        let mut p0_weight = restored_weights[0];  // 976
        let mut p0_hist = [sample_pairs[0], sample_pairs[1]]; // [-128, -203]
        let mut after_p0 = Vec::new();
        for i in 0..10.min(ref_samps.len()) {
            let s1 = if i >= 1 { ref_samps[i-1] } else { p0_hist[0] };
            let s2 = if i >= 2 { ref_samps[i-2] } else if i == 1 { p0_hist[0] } else { p0_hist[1] };
            let pred = ((3i64 * s1 as i64 - s2 as i64) >> 1) as i32;
            let applied = ((pred as i64 * p0_weight as i64 + 512) >> 10) as i32;
            let residual = ref_samps[i] - applied;
            println!("  pass0[{i}]: sample={}, pred={pred}, weight={p0_weight}, applied={applied}, residual={residual}",
                ref_samps[i]);
            if residual != 0 && pred != 0 {
                if (residual ^ pred) >= 0 { p0_weight += 2; } else { p0_weight -= 2; }
            }
            p0_weight = p0_weight.clamp(-1024, 1024);
            after_p0.push(residual);
        }

        // After pass[1] (term=17):
        let mut p1_weight = restored_weights[1]; // 1016
        // History for pass[1]: what samples_data says... [0, 0] since we only have 2 samples total
        let p1_hist_offset = 2; // pass[0] consumed 2 samples
        let p1_hist = [
            if p1_hist_offset < sample_pairs.len() { sample_pairs[p1_hist_offset] } else { 0 },
            if p1_hist_offset + 1 < sample_pairs.len() { sample_pairs[p1_hist_offset + 1] } else { 0 },
        ];
        println!("  pass1 history: {:?}", p1_hist);
        for i in 0..10.min(after_p0.len()) {
            let s1 = if i >= 1 { after_p0[i-1] } else { p1_hist[0] };
            let s2 = if i >= 2 { after_p0[i-2] } else if i == 1 { p1_hist[0] } else { p1_hist[1] };
            let pred = 2 * s1 - s2;
            let applied = ((pred as i64 * p1_weight as i64 + 512) >> 10) as i32;
            let residual = after_p0[i] - applied;
            println!("  pass1[{i}]: input={}, pred={pred}, weight={p1_weight}, applied={applied}, residual={residual}",
                after_p0[i]);
            if residual != 0 && pred != 0 {
                if (residual ^ pred) >= 0 { p1_weight += 2; } else { p1_weight -= 2; }
            }
            p1_weight = p1_weight.clamp(-1024, 1024);
        }
    }

    // Raw bitstream bytes
    println!("\n=== RAW BITSTREAM (first 8 bytes, LSB-first binary) ===");
    for i in 0..8.min(bitstream_data.len()) {
        let b = bitstream_data[i];
        print!("  byte[{i}]=0x{b:02x}=");
        for bit in 0..8 {
            print!("{}", (b >> bit) & 1);
        }
        println!();
    }

    // Manual entropy decode
    println!("\n=== ENTROPY DECODE (first 20 residuals) ===");
    let mut bs = TraceBitstream::new(bitstream_data);
    let mut zero_run_pending = 0u32;
    let mut residuals = Vec::new();

    for sample_idx in 0..20 {
        if zero_run_pending > 0 {
            zero_run_pending -= 1;
            println!("  residual[{sample_idx}]: 0 (zero-run)");
            residuals.push(0i32);
            continue;
        }

        let all_zero = medians.iter().all(|&m| m == 0);
        if all_zero {
            let run = bs.read_exp_golomb();
            if run > 0 {
                zero_run_pending = run - 1;
                println!("  residual[{sample_idx}]: 0 (start zero-run len={run})");
                residuals.push(0);
                continue;
            }
            // Exit zero mode
            medians[0] = 1 << 4;
            println!("  (exit zero mode, median[0]←16)");
        }

        let mut zone = bs.read_unary();
        if zone >= 16 {
            let extra = bs.read_exp_golomb();
            println!("  residual[{sample_idx}]: zone=ESCAPE 16+{extra}={}", zone + extra);
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
                // dec median 0
                let div = 128u32;
                let sub = ((medians[0] + div - 2) / div) * 2;
                medians[0] = medians[0].saturating_sub(sub);
                if sample_idx < 20 {
                    println!("  residual[{sample_idx}]: zone=0, m0={m0}, rem={rem}, mag={magnitude}");
                }
            }
            1 => {
                let rem = bs.read_golomb_remainder(m1);
                magnitude = m0 + rem;
                medians[0] += ((medians[0] + 128) / 128) * 5;
                let sub = ((medians[1] + 30) / 32) * 2;
                medians[1] = medians[1].saturating_sub(sub);
                if sample_idx < 20 {
                    println!("  residual[{sample_idx}]: zone=1, m0={m0}, m1={m1}, rem={rem}, mag={magnitude}");
                }
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
                if sample_idx < 20 {
                    println!("  residual[{sample_idx}]: zone={zone}, m0={m0}, m1={m1}, m2={m2}, base={base}, rem={rem}, mag={magnitude}");
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
        println!("    → signed={signed_val}, medians=[{},{},{}] eff=[{},{},{}]",
            medians[0], medians[1], medians[2],
            (medians[0] >> 4) + 1, (medians[1] >> 4) + 1, (medians[2] >> 4) + 1);
    }

    // Now simulate decorrelation on those residuals
    println!("\n=== DECORRELATION (first 20 samples) ===");
    // We need to apply passes in reverse order
    let mut decoded = residuals.clone();

    // For each pass (reversed), apply prediction
    for pass_i in (0..passes.len()).rev() {
        let (term, delta) = passes[pass_i];
        let weight_idx = pass_i; // mono: one weight per pass
        let mut weight = if weight_idx < restored_weights.len() {
            restored_weights[weight_idx]
        } else {
            0
        };

        // Get history samples for this pass
        // For mono, samples are stored sequentially: term 17/18 → 2 samples, term 1-8 → term samples
        // Need to calculate offset into sample_pairs
        let mut hist_offset = 0;
        for pi in 0..pass_i {
            let (t, _) = passes[pi];
            if t > 8 { hist_offset += 2; }
            else if t >= 1 { hist_offset += t as usize; }
        }
        let (t, _) = passes[pass_i];
        let hist_count = if t > 8 { 2 } else if t >= 1 { t as usize } else { 0 };
        let hist: Vec<i32> = (0..hist_count)
            .map(|i| if hist_offset + i < sample_pairs.len() { sample_pairs[hist_offset + i] } else { 0 })
            .collect();

        println!("  Pass[{pass_i}] (reversed): term={term}, delta={delta}, weight={weight}, hist={hist:?}");

        for i in 0..decoded.len().min(20) {
            let prediction = match term {
                1..=8 => {
                    if i >= term as usize {
                        decoded[i - term as usize]
                    } else {
                        // History: samples[0] is most recent (s[-1]), [1] is s[-2]
                        if (term as usize - 1 - i) < hist.len() {
                            hist[term as usize - 1 - i]
                        } else { 0 }
                    }
                }
                17 => {
                    let s1 = if i >= 1 { decoded[i - 1] } else { hist[0] };
                    let s2 = if i >= 2 { decoded[i - 2] } else if i == 1 { hist[0] } else { hist[1] };
                    2 * s1 - s2
                }
                18 => {
                    let s1 = if i >= 1 { decoded[i - 1] } else { hist[0] };
                    let s2 = if i >= 2 { decoded[i - 2] } else if i == 1 { hist[0] } else { hist[1] };
                    ((3i64 * s1 as i64 - s2 as i64) >> 1) as i32
                }
                _ => 0,
            };

            let residual = decoded[i];
            let reconstructed = residual + ((prediction as i64 * weight as i64 + 512) >> 10) as i32;

            if i < 5 {
                println!("    [{i}]: residual={residual}, prediction={prediction}, weight={weight}, output={reconstructed}");
            }

            // Update weight
            if residual != 0 && prediction != 0 {
                if (residual ^ prediction) >= 0 { weight += delta; }
                else { weight -= delta; }
            }
            weight = weight.clamp(-1024, 1024);

            decoded[i] = reconstructed;
        }
    }

    println!("\n=== FINAL DECODED vs REFERENCE (first 20) ===");
    let ref_path = test_data_path("ref_mono16.wav");
    let ref_samples = if ref_path.exists() {
        read_wav_samples_i16(&ref_path)
    } else {
        vec![]
    };

    for i in 0..20.min(decoded.len()) {
        let r = if i < ref_samples.len() { ref_samples[i] } else { -99999 };
        let mark = if i < ref_samples.len() && decoded[i] == r { "OK" } else { "MISMATCH" };
        println!("  [{i:3}]: decoded={:6}, ref={:6}  {mark}", decoded[i], r);
    }
}
