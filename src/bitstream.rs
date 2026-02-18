/// LSB-first bitstream reader over a byte slice.
///
/// WavPack packs bits LSB-first: the first bit read is bit 0 of the first
/// byte. This is the opposite of Shorten (which is MSB-first).
pub struct BitstreamReader<'a> {
    data: &'a [u8],
    /// Byte position in data.
    byte_pos: usize,
    /// Bit accumulator — we keep up to 64 bits cached.
    accum: u64,
    /// Number of valid bits in accum.
    bits_left: u32,
}

impl<'a> BitstreamReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            accum: 0,
            bits_left: 0,
        }
    }

    /// Refill the accumulator from the byte buffer.
    #[inline]
    fn refill(&mut self) {
        while self.bits_left <= 56 && self.byte_pos < self.data.len() {
            self.accum |= (self.data[self.byte_pos] as u64) << self.bits_left;
            self.byte_pos += 1;
            self.bits_left += 8;
        }
    }

    /// Read `n` bits (0..=32) and return them as a u32.
    #[inline]
    pub fn read_bits(&mut self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        self.refill();
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let val = (self.accum as u32) & mask;
        self.accum >>= n;
        self.bits_left = self.bits_left.saturating_sub(n);
        val
    }

    /// Read a single bit.
    #[inline]
    pub fn read_bit(&mut self) -> u32 {
        self.read_bits(1)
    }

    /// Read a unary code: count consecutive 1-bits, stop at 0-bit terminator.
    ///
    /// Matches ffmpeg's `get_unary_0_33`: counts 1-bits up to a cap of 33,
    /// ALWAYS consuming the 0-bit terminator (unlike the old version which
    /// returned at 16 without consuming it). The escape check (value >= 16)
    /// is handled by the caller in the entropy decoder.
    #[inline]
    pub fn read_unary(&mut self) -> u32 {
        let mut count = 0u32;
        loop {
            self.refill();
            if self.bits_left == 0 {
                return count; // data exhausted
            }
            let bit = (self.accum & 1) as u32;
            self.accum >>= 1;
            self.bits_left = self.bits_left.saturating_sub(1);
            if bit == 0 {
                return count; // 0-bit terminator consumed
            }
            count += 1;
            if count >= 33 {
                return count; // safety cap (like ffmpeg)
            }
        }
    }

    /// Read an Elias gamma code (used for escape values and zero-run lengths).
    ///
    /// Elias gamma: unary-coded exponent (number of 0-bits until first 1-bit),
    /// then that many bits of mantissa. Value = (1 << exp) + mantissa - 1.
    ///
    /// In WavPack's LSB-first encoding: count 0-bits, then read (exp+1) bits.
    /// Actually in WavPack the escape encoding is:
    /// - Read groups of 1-bit (continuation) + bit pairs
    /// - This builds up the value progressively
    ///
    /// WavPack uses a specific variable-length encoding for escape values:
    /// Read bits in pairs with a continuation bit.
    pub fn read_egc(&mut self) -> u32 {
        // WavPack's "extended Golomb code" for escape values:
        // Read 2 bits at a time, with a continuation bit before each pair.
        // ones_count = 0;
        // loop: read 1 bit; if 0, read 2 bits for value += bits << (2*ones_count), done
        //        if 1, read 2 bits for value += bits << (2*ones_count), ones_count++, continue
        let mut value = 1u32; // start at 1 (escape offset)
        let mut shift = 0u32;

        loop {
            let cont = self.read_bit();
            let bits = self.read_bits(1);
            value += (bits + 1) << shift;
            shift += 1;

            if cont == 0 {
                break;
            }
        }

        value
    }

    /// Read a unary code with no limit (for exp-Golomb prefix in escape/zero-run).
    /// Counts consecutive 1-bits until a 0-bit terminator, no cap.
    #[inline]
    pub fn read_unary_unlimited(&mut self) -> u32 {
        let mut count = 0u32;
        loop {
            self.refill();
            if self.bits_left == 0 {
                return count;
            }
            let bit = (self.accum & 1) as u32;
            self.accum >>= 1;
            self.bits_left = self.bits_left.saturating_sub(1);
            if bit == 0 {
                return count;
            }
            count += 1;
        }
    }

    /// Number of bits remaining in the buffer.
    pub fn bits_remaining(&self) -> usize {
        (self.data.len() - self.byte_pos) * 8 + self.bits_left as usize
    }

    /// True if the bitstream is exhausted.
    pub fn is_empty(&self) -> bool {
        self.bits_left == 0 && self.byte_pos >= self.data.len()
    }

    /// Total bits consumed so far.
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 - self.bits_left as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bits_basic() {
        // 0xA5 = 1010_0101 → LSB-first: bit0=1, bit1=0, bit2=1, bit3=0, bit4=0, bit5=1, ...
        let data = [0xA5u8];
        let mut br = BitstreamReader::new(&data);

        assert_eq!(br.read_bit(), 1); // bit 0
        assert_eq!(br.read_bit(), 0); // bit 1
        assert_eq!(br.read_bits(2), 0b01); // bits 2-3 (bit2=1, bit3=0 → 01 in LSB-first)
        assert_eq!(br.read_bits(4), 0b1010); // bits 4-7
    }

    #[test]
    fn read_bits_multi_byte() {
        // 0xFF, 0x00 = 8 ones, then 8 zeros
        let data = [0xFF, 0x00];
        let mut br = BitstreamReader::new(&data);

        assert_eq!(br.read_bits(8), 0xFF);
        assert_eq!(br.read_bits(8), 0x00);
    }

    #[test]
    fn read_unary_basic() {
        // Bit pattern LSB-first: 1,1,1,0,... → unary = 3
        // Byte: 0b_xxxx_0111 = 0x07 (plus whatever)
        let data = [0x07]; // bits: 1,1,1,0,0,0,0,0
        let mut br = BitstreamReader::new(&data);
        assert_eq!(br.read_unary(), 3);
    }

    #[test]
    fn read_unary_zero() {
        // First bit is 0 → unary = 0
        let data = [0x00];
        let mut br = BitstreamReader::new(&data);
        assert_eq!(br.read_unary(), 0);
    }

    #[test]
    fn read_unary_escape() {
        // 16 consecutive 1-bits → escape
        let data = [0xFF, 0xFF]; // 16 ones
        let mut br = BitstreamReader::new(&data);
        assert_eq!(br.read_unary(), 16);
    }

    #[test]
    fn read_bits_cross_byte() {
        // Read 12 bits spanning 2 bytes
        let data = [0xAB, 0xCD]; // 0xCDAB in 16-bit LE
        let mut br = BitstreamReader::new(&data);
        let val = br.read_bits(12);
        // LSB-first: bottom 12 bits of 0xCDAB = 0xDAB
        assert_eq!(val, 0xDAB);
    }

    #[test]
    fn bits_remaining() {
        let data = [0x00, 0x00, 0x00];
        let mut br = BitstreamReader::new(&data);
        assert_eq!(br.bits_remaining(), 24);
        br.read_bits(5);
        assert_eq!(br.bits_remaining(), 19);
    }
}
