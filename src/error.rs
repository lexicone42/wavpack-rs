use std::fmt;
use std::io;

/// Errors that can occur during WavPack decoding.
#[derive(Debug)]
pub enum WavPackError {
    /// I/O error from the underlying reader.
    Io(io::Error),
    /// Invalid or missing "wvpk" magic bytes.
    BadMagic,
    /// Unsupported WavPack version.
    UnsupportedVersion(u16),
    /// Hybrid (lossy) mode is not supported.
    HybridNotSupported,
    /// Floating-point data is not supported.
    FloatNotSupported,
    /// DSD audio is not supported.
    DsdNotSupported,
    /// Corrupt or truncated block header.
    BadHeader(String),
    /// Corrupt or truncated metadata sub-block.
    BadMetadata(String),
    /// Error in entropy decoding (bitstream corruption).
    EntropyError(String),
    /// Error in decorrelation filter.
    DecorrelationError(String),
    /// CRC mismatch after decoding a block.
    CrcMismatch { expected: u32, actual: u32 },
    /// No audio blocks found in the file.
    NoAudioBlocks,
}

impl fmt::Display for WavPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::BadMagic => write!(f, "invalid WavPack magic (expected 'wvpk')"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported WavPack version 0x{v:04x}")
            }
            Self::HybridNotSupported => write!(f, "hybrid (lossy) mode not supported"),
            Self::FloatNotSupported => write!(f, "floating-point data not supported"),
            Self::DsdNotSupported => write!(f, "DSD audio not supported"),
            Self::BadHeader(msg) => write!(f, "bad block header: {msg}"),
            Self::BadMetadata(msg) => write!(f, "bad metadata sub-block: {msg}"),
            Self::EntropyError(msg) => write!(f, "entropy decode error: {msg}"),
            Self::DecorrelationError(msg) => write!(f, "decorrelation error: {msg}"),
            Self::CrcMismatch { expected, actual } => {
                write!(f, "CRC mismatch: expected 0x{expected:08x}, got 0x{actual:08x}")
            }
            Self::NoAudioBlocks => write!(f, "no audio blocks found in file"),
        }
    }
}

impl std::error::Error for WavPackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for WavPackError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
