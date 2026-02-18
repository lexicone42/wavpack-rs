use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

fn main() {
    let path = std::env::args().nth(1).expect("Usage: debug_dump <file.wv>");
    let file = File::open(&path).unwrap();
    let mut reader = BufReader::new(file);

    let mut block_num = 0;
    loop {
        let mut magic = [0u8; 4];
        if reader.read_exact(&mut magic).is_err() { break; }
        if &magic != b"wvpk" {
            eprintln!("Bad magic at block {block_num}");
            break;
        }

        let mut hdr_buf = [0u8; 28];
        reader.read_exact(&mut hdr_buf).unwrap();

        let block_size = u32::from_le_bytes([hdr_buf[0], hdr_buf[1], hdr_buf[2], hdr_buf[3]]);
        let version = u16::from_le_bytes([hdr_buf[4], hdr_buf[5]]);
        let block_index_u8 = hdr_buf[6];
        let total_samples_u8 = hdr_buf[7];
        let total_samples = u32::from_le_bytes([hdr_buf[8], hdr_buf[9], hdr_buf[10], hdr_buf[11]]);
        let block_index = u32::from_le_bytes([hdr_buf[12], hdr_buf[13], hdr_buf[14], hdr_buf[15]]);
        let block_samples = u32::from_le_bytes([hdr_buf[16], hdr_buf[17], hdr_buf[18], hdr_buf[19]]);
        let flags = u32::from_le_bytes([hdr_buf[20], hdr_buf[21], hdr_buf[22], hdr_buf[23]]);
        let crc = u32::from_le_bytes([hdr_buf[24], hdr_buf[25], hdr_buf[26], hdr_buf[27]]);

        let bps = (flags & 0x03) + 1;
        let mono = (flags >> 2) & 1;
        let hybrid = (flags >> 3) & 1;
        let joint = (flags >> 4) & 1;
        let cross = (flags >> 5) & 1;
        let left_shift = (flags >> 13) & 0x1f;
        let max_mag = (flags >> 18) & 0x1f;
        let sr_idx = (flags >> 23) & 0x0f;
        let false_stereo = (flags >> 30) & 1;

        println!("=== Block {block_num} ===");
        println!("  block_size={block_size}, version=0x{version:04x}");
        println!("  total_samples={total_samples} (u8={total_samples_u8}), block_index={block_index} (u8={block_index_u8})");
        println!("  block_samples={block_samples}");
        println!("  flags=0x{flags:08x}: bps={bps}({} bits), mono={mono}, hybrid={hybrid}, joint={joint}, cross={cross}", bps*8);
        println!("  left_shift={left_shift}, max_mag={max_mag}, sr_idx={sr_idx}, false_stereo={false_stereo}");
        println!("  crc=0x{crc:08x}");

        let payload_size = block_size - 24;
        let mut payload = vec![0u8; payload_size as usize];
        reader.read_exact(&mut payload).unwrap();

        // Parse sub-blocks
        let mut pos = 0usize;
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
            let data = &payload[pos..pos+data_len];

            let id_name = match id {
                0x00 => "DUMMY",
                0x02 => "DECORR_TERMS",
                0x03 => "DECORR_WEIGHTS",
                0x04 => "DECORR_SAMPLES",
                0x05 => "ENTROPY_VARS",
                0x09 => "INT32_INFO",
                0x0a => "WV_BITSTREAM",
                0x0d => "CHANNEL_INFO",
                0x21 => "RIFF_HEADER",
                0x22 => "RIFF_TRAILER",
                0x25 => "CONFIG_BLOCK",
                0x26 => "MD5_CHECKSUM",
                0x27 => "SAMPLE_RATE",
                0x2f => "BLOCK_CHECKSUM",
                _ => "UNKNOWN",
            };

            print!("  Sub-block 0x{id:02x} ({id_name}): {data_len} bytes");
            if data_len <= 32 {
                print!(" = ");
                for b in data { print!("{b:02x} "); }
            }
            println!();

            // Detailed parsing for key sub-blocks
            if id == 0x02 { // DECORR_TERMS
                print!("    terms: ");
                for &b in data {
                    let term = (b & 0x1f) as i32 - 5;
                    let delta = b >> 5;
                    print!("(term={term}, delta={delta}) ");
                }
                println!();
            }
            if id == 0x03 { // DECORR_WEIGHTS
                print!("    weights: ");
                for &b in data {
                    let stored = b as i8;
                    let mut w = (stored as i32) << 3;
                    if w > 0 { w += (w + 64) >> 7; }
                    else if w < 0 { w -= ((-w) + 64) >> 7; }
                    print!("{w} ");
                }
                println!();
            }
            if id == 0x05 { // ENTROPY_VARS
                print!("    entropy medians: ");
                for chunk in data.chunks_exact(4) {
                    let packed = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let mantissa = packed & 0xFF;
                    let exp = (packed >> 8) & 0x1F;
                    let restored = ((mantissa | 0x100) << exp) >> 9;
                    print!("{restored} (packed=0x{packed:08x}) ");
                }
                println!();
            }

            pos += size_bytes;
        }
        println!();

        block_num += 1;
        if block_num >= 4 { break; } // limit output
    }
}
