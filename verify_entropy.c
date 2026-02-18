// Minimal entropy decoder verification using exact FFmpeg/WavPack algorithm.
// Compiles with: gcc -o verify_entropy verify_entropy.c -O2
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// ── Bitstream reader (LSB-first, matching WavPack) ──

typedef struct {
    const uint8_t *data;
    int byte_pos;
    uint64_t accum;
    int bits_left;
} BitReader;

void br_init(BitReader *br, const uint8_t *data) {
    br->data = data;
    br->byte_pos = 0;
    br->accum = 0;
    br->bits_left = 0;
}

static inline void br_refill(BitReader *br) {
    while (br->bits_left <= 56) {
        br->accum |= (uint64_t)br->data[br->byte_pos] << br->bits_left;
        br->byte_pos++;
        br->bits_left += 8;
    }
}

static inline uint32_t br_read_bits(BitReader *br, int n) {
    if (n == 0) return 0;
    br_refill(br);
    uint32_t mask = (n == 32) ? 0xFFFFFFFF : ((1u << n) - 1);
    uint32_t val = (uint32_t)(br->accum) & mask;
    br->accum >>= n;
    br->bits_left -= n;
    return val;
}

static inline uint32_t br_read_bit(BitReader *br) {
    return br_read_bits(br, 1);
}

// get_unary_0_33: count 1-bits until 0 or cap at 33
static inline uint32_t br_read_unary(BitReader *br) {
    uint32_t count = 0;
    while (count < 33) {
        br_refill(br);
        uint32_t bit = br->accum & 1;
        br->accum >>= 1;
        br->bits_left--;
        if (bit == 0) return count;
        count++;
    }
    return count;
}

int br_bit_position(BitReader *br) {
    return br->byte_pos * 8 - br->bits_left;
}

// ── Entropy decoder (exact FFmpeg wv_get_value algorithm) ──

#define GET_MED(n) ((medians[n] >> 4) + 1)
#define INC_MED(n) do { \
    uint32_t div = (128U >> (n)); \
    medians[n] += ((medians[n] + div) / div) * 5; \
} while(0)
#define DEC_MED(n) do { \
    uint32_t div = (128U >> (n)); \
    uint32_t sub = ((medians[n] + div - 2) / div) * 2; \
    if (sub > medians[n]) medians[n] = 0; else medians[n] -= sub; \
} while(0)

// get_tail (adjusted binary Golomb)
static uint32_t get_tail(BitReader *br, uint32_t k) {
    if (k < 1) return 0;
    int p = 31 - __builtin_clz(k);  // av_log2(k)
    uint32_t e = (1u << (p + 1)) - k - 1;
    uint32_t res = br_read_bits(br, p);
    if (res >= e) {
        res = res * 2 - e + br_read_bit(br);
    }
    return res;
}

int main(void) {
    // Bitstream bytes from the mono16_high test file
    uint8_t bs[] = {
        0xff, 0xff, 0x32, 0x7f, 0xe5, 0xff, 0xdf, 0x85,
        0x2f, 0xfe, 0x82, 0x5c, 0x19, 0x9c, 0x27, 0x2e,
        // need more bytes for safety, pad with zeros
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    BitReader br;
    br_init(&br, bs);

    // Initial medians from entropy_vars: [88, 76, 191]
    uint32_t medians[3] = {88, 76, 191};
    int zero = 0, one = 0;
    uint32_t zeroes = 0;

    printf("Initial medians: [%u, %u, %u]\n", medians[0], medians[1], medians[2]);
    printf("GET_MED: [%u, %u, %u]\n\n", GET_MED(0), GET_MED(1), GET_MED(2));

    // Decode 5 samples
    for (int s = 0; s < 5; s++) {
        int bit_before = br_bit_position(&br);
        printf("Sample %d (bit=%d):\n", s, bit_before);
        printf("  medians=[%u, %u, %u] GET_MED=[%u, %u, %u]\n",
               medians[0], medians[1], medians[2],
               GET_MED(0), GET_MED(1), GET_MED(2));
        printf("  zero=%d one=%d zeroes=%u\n", zero, one, zeroes);

        // 1. Zero-run check (mono: just 1 channel)
        if (medians[0] < 2 && !zero && !one) {
            if (zeroes > 0) {
                zeroes--;
                if (zeroes > 0) {
                    printf("  zero-run: value=0\n\n");
                    continue;
                }
                // fall through
            } else {
                uint32_t run = br_read_unary(&br);
                if (run >= 2) {
                    uint32_t extra = br_read_bits(&br, run - 1);
                    run = extra | (1u << (run - 1));
                }
                zeroes = run;
                if (zeroes > 0) {
                    medians[0] = medians[1] = medians[2] = 0;
                    printf("  zero-run=%u, value=0\n\n", zeroes);
                    continue;
                }
            }
        }

        // 2. Normal decode
        uint32_t zone;
        if (zero) {
            zone = 0;
            zero = 0;
        } else {
            uint32_t raw = br_read_unary(&br);
            printf("  read_unary=%u (bit_after=%d)\n", raw, br_bit_position(&br));

            if (raw == 16) {
                uint32_t t2 = br_read_unary(&br);
                if (t2 < 2) {
                    raw += t2;
                } else {
                    raw += br_read_bits(&br, t2 - 1) | (1u << (t2 - 1));
                }
                printf("  escape: raw=%u\n", raw);
            }

            if (one) {
                one = raw & 1;
                zone = (raw >> 1) + 1;
            } else {
                one = raw & 1;
                zone = raw >> 1;
            }
            zero = !one;
        }
        printf("  zone=%u (one=%d zero=%d)\n", zone, one, zero);

        // 3. Decode magnitude from zone
        uint32_t base, add, magnitude;
        if (zone == 0) {
            base = 0;
            add = GET_MED(0) - 1;
            DEC_MED(0);
        } else if (zone == 1) {
            base = GET_MED(0);
            add = GET_MED(1) - 1;
            INC_MED(0);
            DEC_MED(1);
        } else if (zone == 2) {
            base = GET_MED(0) + GET_MED(1);
            add = GET_MED(2) - 1;
            INC_MED(0);
            INC_MED(1);
            DEC_MED(2);
        } else {
            base = GET_MED(0) + GET_MED(1) + GET_MED(2) * (zone - 2);
            add = GET_MED(2) - 1;
            INC_MED(0);
            INC_MED(1);
            INC_MED(2);
        }

        int tail_bit = br_bit_position(&br);
        uint32_t rem = get_tail(&br, add);
        magnitude = base + rem;
        printf("  base=%u add=%u rem=%u magnitude=%u (tail bits=%d-%d)\n",
               base, add, rem, magnitude, tail_bit, br_bit_position(&br));

        // 4. Sign bit
        uint32_t sign = br_read_bit(&br);
        int32_t result = sign ? ~(int32_t)magnitude : (int32_t)magnitude;
        printf("  sign=%u result=%d (total bits=%d)\n", sign, result, br_bit_position(&br) - bit_before);
        printf("  medians_after=[%u, %u, %u]\n\n", medians[0], medians[1], medians[2]);
    }

    return 0;
}
