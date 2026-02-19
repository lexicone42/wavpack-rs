# wavpack-rs

A pure Rust WavPack lossless audio decoder. Decodes `.wv` files to raw PCM samples with no external dependencies (no C libraries, no ffmpeg).

## Features

- Lossless decoding of WavPack files (version 4.x format)
- Mono and stereo, 16-bit and 24-bit
- All compression modes: fast, normal, high, very high
- Joint stereo with cross-channel decorrelation (terms -1, -2, -3)
- Bit-exact output verified against `wvunpack` reference decoder

## Usage

```rust
use wavpack_rs::WavPackReader;
use std::io::BufReader;
use std::fs::File;

let file = BufReader::new(File::open("audio.wv").unwrap());
let mut reader = WavPackReader::new(file).unwrap();

let info = reader.info();
println!("{}ch, {}Hz, {}bit", info.channels, info.sample_rate, info.bits_per_sample);

let samples = reader.decode_all().unwrap();
// samples[0] = left channel, samples[1] = right channel (if stereo)
```

## Test Suite

18 decode tests covering all compression modes and real-world files:

```
cargo test
```

Test data includes synthetic ramps (mono/stereo, 16/24-bit, all compression levels), real-world Grateful Dead (1970 SBD, high compression), and Steve Reich (24-bit, high compression).

## Architecture

- `lib.rs` — WavPackReader, block parsing, decode pipeline
- `header.rs` — WavPack block header and sub-block parsing
- `bitstream.rs` — LSB-first bitstream reader
- `entropy.rs` — Adaptive entropy decoder (3-median zone coding)
- `decorrelation.rs` — Multi-pass LMS decorrelation with cross-channel support

## Implementation notes

This is an original Rust implementation, not a translation of any existing decoder. The official WavPack source code (BSD-licensed, by David Bryant) was referenced during development for algorithm details, particularly the decorrelation and entropy coding. Output is verified bit-exact against `wvunpack 5.8.1`.

## License

MIT
