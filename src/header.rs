use std::io::{Read, Seek, SeekFrom};

use crate::error::WavPackError;

/// The 4-byte WavPack block magic: "wvpk".
pub const MAGIC: [u8; 4] = *b"wvpk";

/// Minimum valid WavPack version for decoding.
pub const MIN_VERSION: u16 = 0x0402;
/// Maximum valid WavPack version for decoding.
pub const MAX_VERSION: u16 = 0x0410;

/// Standard sample rates indexed by the 4-bit rate field in flags.
/// Index 15 = unknown/custom (read from ID_SAMPLE_RATE sub-block).
pub const SAMPLE_RATES: [u32; 15] = [
    6000, 8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 64000, 88200,
    96000, 192000,
];

// ── Flag bit positions ──────────────────────────────────────────────

pub const FLAG_BYTES_PER_SAMPLE_MASK: u32 = 0x03; // bits 0-1
pub const FLAG_MONO: u32 = 1 << 2;
pub const FLAG_HYBRID: u32 = 1 << 3;
pub const FLAG_JOINT_STEREO: u32 = 1 << 4;
pub const FLAG_CROSS_DECORR: u32 = 1 << 5;
pub const FLAG_HYBRID_SHAPE: u32 = 1 << 6;
pub const FLAG_FLOAT: u32 = 1 << 7;
pub const FLAG_INT32_INFO: u32 = 1 << 8;
pub const FLAG_HYBRID_BITRATE: u32 = 1 << 9;
pub const FLAG_HYBRID_BALANCED: u32 = 1 << 10;
pub const FLAG_INITIAL_BLOCK: u32 = 1 << 11;
pub const FLAG_FINAL_BLOCK: u32 = 1 << 12;
pub const FLAG_LEFT_SHIFT_MASK: u32 = 0x1f << 13; // bits 13-17
pub const FLAG_LEFT_SHIFT_SHIFT: u32 = 13;
pub const FLAG_MAX_MAGNITUDE_MASK: u32 = 0x1f << 18; // bits 18-22
pub const FLAG_MAX_MAGNITUDE_SHIFT: u32 = 18;
pub const FLAG_SAMPLE_RATE_MASK: u32 = 0x0f << 23; // bits 23-26
pub const FLAG_SAMPLE_RATE_SHIFT: u32 = 23;
pub const FLAG_HAS_CHECKSUM: u32 = 1 << 28;
pub const FLAG_FALSE_STEREO: u32 = 1 << 30;
pub const FLAG_DSD: u32 = 1 << 31;

// ── Metadata sub-block IDs ──────────────────────────────────────────

pub const ID_MASK: u8 = 0x3f;
pub const ID_FLAG_OPTIONAL: u8 = 0x20; // decoder can ignore
pub const ID_FLAG_ODD_SIZE: u8 = 0x40; // actual data is 1 byte less than stored
pub const ID_FLAG_LARGE: u8 = 0x80; // size is 3 bytes (24-bit) instead of 1

pub const ID_DUMMY: u8 = 0x00;
pub const ID_DECORR_TERMS: u8 = 0x02;
pub const ID_DECORR_WEIGHTS: u8 = 0x03;
pub const ID_DECORR_SAMPLES: u8 = 0x04;
pub const ID_ENTROPY_VARS: u8 = 0x05;
pub const ID_HYBRID_PROFILE: u8 = 0x06;
pub const ID_SHAPING_WEIGHTS: u8 = 0x07;
pub const ID_FLOAT_INFO: u8 = 0x08;
pub const ID_INT32_INFO: u8 = 0x09;
pub const ID_WV_BITSTREAM: u8 = 0x0a;
pub const ID_WVC_BITSTREAM: u8 = 0x0b;
pub const ID_WVX_BITSTREAM: u8 = 0x0c;
pub const ID_CHANNEL_INFO: u8 = 0x0d;
pub const ID_DSD_BLOCK: u8 = 0x0e;
pub const ID_RIFF_HEADER: u8 = 0x21;
pub const ID_RIFF_TRAILER: u8 = 0x22;
pub const ID_CONFIG_BLOCK: u8 = 0x25;
pub const ID_MD5_CHECKSUM: u8 = 0x26;
pub const ID_SAMPLE_RATE: u8 = 0x27;
pub const ID_ALT_HEADER: u8 = 0x23;
pub const ID_ALT_TRAILER: u8 = 0x24;
pub const ID_BLOCK_CHECKSUM: u8 = 0x2f;

/// Parsed 32-byte WavPack block header.
#[derive(Debug, Clone)]
pub struct BlockHeader {
    /// Size of the entire block minus 8 (magic + this field).
    pub block_size: u32,
    /// Format version (0x0402..=0x0410).
    pub version: u16,
    /// Total samples in file (40-bit, lower 32 + upper 8).
    /// 0xFFFFFFFFFF = unknown length.
    pub total_samples: u64,
    /// Index of the first sample in this block (40-bit).
    pub block_index: u64,
    /// Number of samples in this block (0 = metadata-only).
    pub block_samples: u32,
    /// Raw flags bitfield.
    pub flags: u32,
    /// CRC of decoded data.
    pub crc: u32,
}

impl BlockHeader {
    /// Bytes per sample (1-4), derived from flag bits 0-1.
    pub fn bytes_per_sample(&self) -> u32 {
        (self.flags & FLAG_BYTES_PER_SAMPLE_MASK) + 1
    }

    /// Bits per sample (8, 16, 24, or 32).
    pub fn bits_per_sample(&self) -> u32 {
        self.bytes_per_sample() * 8
    }

    /// True if mono output.
    pub fn is_mono(&self) -> bool {
        self.flags & FLAG_MONO != 0
    }

    /// True if hybrid (lossy) mode.
    pub fn is_hybrid(&self) -> bool {
        self.flags & FLAG_HYBRID != 0
    }

    /// True if joint stereo (mid/side) encoding.
    pub fn is_joint_stereo(&self) -> bool {
        self.flags & FLAG_JOINT_STEREO != 0
    }

    /// True if cross-channel decorrelation is used.
    pub fn is_cross_decorr(&self) -> bool {
        self.flags & FLAG_CROSS_DECORR != 0
    }

    /// True if floating-point data.
    pub fn is_float(&self) -> bool {
        self.flags & FLAG_FLOAT != 0
    }

    /// True if int32 info sub-block is present (>24-bit or shifted).
    pub fn has_int32_info(&self) -> bool {
        self.flags & FLAG_INT32_INFO != 0
    }

    /// True if this is the first block in a multichannel sequence.
    pub fn is_initial_block(&self) -> bool {
        self.flags & FLAG_INITIAL_BLOCK != 0
    }

    /// True if this is the last block in a multichannel sequence.
    pub fn is_final_block(&self) -> bool {
        self.flags & FLAG_FINAL_BLOCK != 0
    }

    /// Left-shift amount to apply after decode (bits 13-17).
    pub fn left_shift(&self) -> u32 {
        (self.flags & FLAG_LEFT_SHIFT_MASK) >> FLAG_LEFT_SHIFT_SHIFT
    }

    /// Maximum magnitude of decoded samples (bits 18-22).
    /// The actual number of significant bits is this value + 1.
    pub fn max_magnitude(&self) -> u32 {
        (self.flags & FLAG_MAX_MAGNITUDE_MASK) >> FLAG_MAX_MAGNITUDE_SHIFT
    }

    /// Sample rate from the standard table, or None if custom/unknown.
    pub fn sample_rate_index(&self) -> u32 {
        (self.flags & FLAG_SAMPLE_RATE_MASK) >> FLAG_SAMPLE_RATE_SHIFT
    }

    /// Sample rate from standard table. Returns None for index 15 (custom).
    pub fn sample_rate(&self) -> Option<u32> {
        let idx = self.sample_rate_index() as usize;
        SAMPLE_RATES.get(idx).copied()
    }

    /// True if false stereo (data is mono, output duplicated to both channels).
    pub fn is_false_stereo(&self) -> bool {
        self.flags & FLAG_FALSE_STEREO != 0
    }

    /// True if DSD audio (unsupported).
    pub fn is_dsd(&self) -> bool {
        self.flags & FLAG_DSD != 0
    }

    /// True if the block has a trailing checksum sub-block.
    pub fn has_checksum(&self) -> bool {
        self.flags & FLAG_HAS_CHECKSUM != 0
    }

    /// Number of channels in this block (1 or 2).
    pub fn block_channels(&self) -> usize {
        if self.is_mono() { 1 } else { 2 }
    }
}

/// A parsed metadata sub-block from within a WavPack block.
#[derive(Debug, Clone)]
pub struct MetaSubBlock {
    /// The function ID (lower 6 bits of the raw ID byte).
    pub id: u8,
    /// Raw ID byte including flag bits.
    pub raw_id: u8,
    /// The sub-block data payload.
    pub data: Vec<u8>,
}

impl MetaSubBlock {
    /// True if the decoder can ignore this sub-block.
    pub fn is_optional(&self) -> bool {
        self.raw_id & ID_FLAG_OPTIONAL != 0
    }
}

/// Read a 32-byte block header from the current position.
/// Returns None at EOF, or error if partial/corrupt.
pub fn read_block_header<R: Read>(reader: &mut R) -> Result<Option<BlockHeader>, WavPackError> {
    let mut buf = [0u8; 32];

    // Read magic — EOF here is normal (end of file)
    match reader.read_exact(&mut buf[..4]) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }

    if buf[..4] != MAGIC {
        return Err(WavPackError::BadMagic);
    }

    // Read remaining 28 bytes
    reader.read_exact(&mut buf[4..])?;

    let block_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let version = u16::from_le_bytes([buf[8], buf[9]]);
    let block_index_u8 = buf[10];
    let total_samples_u8 = buf[11];
    let total_samples_lo = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]);
    let block_index_lo = u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]);
    let block_samples = u32::from_le_bytes([buf[20], buf[21], buf[22], buf[23]]);
    let flags = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
    let crc = u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]);

    // Reconstruct 40-bit values
    let total_samples = (total_samples_lo as u64) | ((total_samples_u8 as u64) << 32);
    let block_index = (block_index_lo as u64) | ((block_index_u8 as u64) << 32);

    if version < MIN_VERSION || version > MAX_VERSION {
        return Err(WavPackError::UnsupportedVersion(version));
    }

    Ok(Some(BlockHeader {
        block_size,
        version,
        total_samples,
        block_index,
        block_samples,
        flags,
        crc,
    }))
}

/// Read all metadata sub-blocks from a block's payload.
///
/// `payload` is the raw bytes after the 32-byte header, with length
/// `block_size - 24` (since block_size counts from after magic+size, i.e.
/// the 32-byte header minus 8 = 24 bytes of header fields after size).
pub fn parse_metadata(payload: &[u8]) -> Result<Vec<MetaSubBlock>, WavPackError> {
    let mut subs = Vec::new();
    let mut pos = 0;

    while pos < payload.len() {
        if pos + 1 > payload.len() {
            break;
        }
        let raw_id = payload[pos];
        pos += 1;

        let id = raw_id & ID_MASK;
        let is_odd = raw_id & ID_FLAG_ODD_SIZE != 0;
        let is_large = raw_id & ID_FLAG_LARGE != 0;

        // Read size in words
        let size_words: u32 = if is_large {
            if pos + 3 > payload.len() {
                return Err(WavPackError::BadMetadata("truncated large size".into()));
            }
            let w = u32::from_le_bytes([payload[pos], payload[pos + 1], payload[pos + 2], 0]);
            pos += 3;
            w
        } else {
            if pos >= payload.len() {
                return Err(WavPackError::BadMetadata("truncated small size".into()));
            }
            let w = payload[pos] as u32;
            pos += 1;
            w
        };

        // Convert to bytes (words are 2 bytes each)
        let size_bytes = (size_words * 2) as usize;

        // Actual data length may be 1 less if odd flag is set
        let data_len = if is_odd && size_bytes > 0 {
            size_bytes - 1
        } else {
            size_bytes
        };

        if pos + size_bytes > payload.len() {
            return Err(WavPackError::BadMetadata(format!(
                "sub-block 0x{id:02x} claims {size_bytes} bytes but only {} remain",
                payload.len() - pos
            )));
        }

        let data = payload[pos..pos + data_len].to_vec();
        pos += size_bytes; // advance past padding byte too

        subs.push(MetaSubBlock { id, raw_id, data });
    }

    Ok(subs)
}

/// Scan forward looking for the next "wvpk" magic, skipping non-block data
/// (like APEv2 tags at the start of file). Returns Ok(true) if found,
/// Ok(false) at EOF.
pub fn sync_to_block<R: Read + Seek>(reader: &mut R) -> Result<bool, WavPackError> {
    let mut buf = [0u8; 1];
    let mut window: u32 = 0;

    loop {
        match reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
            Err(e) => return Err(e.into()),
        }

        window = (window << 8) | (buf[0] as u32);
        if window == u32::from_be_bytes(MAGIC) {
            // Seek back to before the magic so read_block_header can read it
            reader.seek(SeekFrom::Current(-4))?;
            return Ok(true);
        }
    }
}

/// Read the full payload of a block (everything after the 32-byte header).
/// The payload length is `block_size - 24`.
pub fn read_block_payload<R: Read>(
    reader: &mut R,
    header: &BlockHeader,
) -> Result<Vec<u8>, WavPackError> {
    let payload_size = header
        .block_size
        .checked_sub(24)
        .ok_or_else(|| WavPackError::BadHeader("block_size too small".into()))?
        as usize;

    let mut payload = vec![0u8; payload_size];
    reader.read_exact(&mut payload)?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_header_from_bytes() {
        // Hand-crafted minimal WavPack header
        let mut buf = Vec::new();
        buf.extend_from_slice(b"wvpk"); // magic
        buf.extend_from_slice(&100u32.to_le_bytes()); // block_size
        buf.extend_from_slice(&0x0410u16.to_le_bytes()); // version
        buf.push(0); // block_index_u8
        buf.push(0); // total_samples_u8
        buf.extend_from_slice(&44100u32.to_le_bytes()); // total_samples (lo)
        buf.extend_from_slice(&0u32.to_le_bytes()); // block_index (lo)
        buf.extend_from_slice(&22050u32.to_le_bytes()); // block_samples
        // flags: 16-bit stereo, lossless, joint, initial+final, sr_idx=9, checksum
        let flags: u32 = 0x01 // 2 bytes/sample
            | FLAG_JOINT_STEREO
            | FLAG_CROSS_DECORR
            | FLAG_INITIAL_BLOCK
            | FLAG_FINAL_BLOCK
            | (15 << FLAG_MAX_MAGNITUDE_SHIFT) // 15 = 16-bit magnitude
            | (9 << FLAG_SAMPLE_RATE_SHIFT) // 44100 Hz
            | FLAG_HAS_CHECKSUM;
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // crc

        let mut cursor = Cursor::new(buf);
        let hdr = read_block_header(&mut cursor).unwrap().unwrap();

        assert_eq!(hdr.version, 0x0410);
        assert_eq!(hdr.total_samples, 44100);
        assert_eq!(hdr.block_samples, 22050);
        assert_eq!(hdr.bits_per_sample(), 16);
        assert!(!hdr.is_mono());
        assert!(hdr.is_joint_stereo());
        assert!(hdr.is_initial_block());
        assert!(hdr.is_final_block());
        assert!(!hdr.is_hybrid());
        assert_eq!(hdr.sample_rate(), Some(44100));
        assert_eq!(hdr.max_magnitude(), 15);
        assert_eq!(hdr.left_shift(), 0);
        assert_eq!(hdr.crc, 0xDEADBEEF);
    }

    #[test]
    fn parse_metadata_sub_blocks() {
        // Small sub-block: ID=0x02, size=3 words (6 bytes), odd flag (actual 5 bytes)
        let mut payload = Vec::new();
        payload.push(0x02 | ID_FLAG_ODD_SIZE); // id with odd flag
        payload.push(3); // 3 words = 6 bytes stored, 5 bytes actual
        payload.extend_from_slice(&[1, 2, 3, 4, 5, 0]); // data + padding

        // Large sub-block: ID=0x0a with large flag, size=2 words
        payload.push(0x0a | ID_FLAG_LARGE);
        payload.extend_from_slice(&[2, 0, 0]); // 24-bit LE: 2 words = 4 bytes
        payload.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

        let subs = parse_metadata(&payload).unwrap();
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0].id, 0x02);
        assert_eq!(subs[0].data, &[1, 2, 3, 4, 5]);
        assert_eq!(subs[1].id, 0x0a);
        assert_eq!(subs[1].data, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn eof_returns_none() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        assert!(read_block_header(&mut cursor).unwrap().is_none());
    }

    #[test]
    fn bad_magic_is_error() {
        let mut cursor = Cursor::new(b"RIFFxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".to_vec());
        assert!(matches!(
            read_block_header(&mut cursor),
            Err(WavPackError::BadMagic)
        ));
    }

    #[test]
    fn flag_accessors() {
        let hdr = BlockHeader {
            block_size: 100,
            version: 0x0410,
            total_samples: 1000,
            block_index: 0,
            block_samples: 1000,
            flags: FLAG_MONO | FLAG_HYBRID | FLAG_FLOAT | FLAG_DSD | 0x02, // 3 bytes/sample
            crc: 0,
        };
        assert!(hdr.is_mono());
        assert!(hdr.is_hybrid());
        assert!(hdr.is_float());
        assert!(hdr.is_dsd());
        assert_eq!(hdr.bytes_per_sample(), 3);
        assert_eq!(hdr.bits_per_sample(), 24);
        assert_eq!(hdr.block_channels(), 1);
    }
}
