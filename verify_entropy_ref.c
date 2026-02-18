// Standalone WavPack reference entropy decoder verification.
// Uses EXACT code from WavPack reference (bitstream, entropy logic).
// Compiles with: gcc -o verify_entropy_ref verify_entropy_ref.c -O2
#include <stdio.h>
#include <stdint.h>
#include <string.h>

// ═══════════════════════════════════════════════════════
// EXACT WavPack Bitstream struct and macros
// ═══════════════════════════════════════════════════════

typedef struct bs {
    unsigned char *buf, *end, *ptr;
    void (*wrap)(struct bs *bs);
    int error, bc;
    uint32_t sr;
} Bitstream;

// Wrap function that sets error (shouldn't happen in our test)
static void bs_wrap(Bitstream *bs) {
    fprintf(stderr, "ERROR: bitstream wrap called!\n");
    bs->error = 1;
}

#define getbit(bs) ( \
    (((bs)->bc) ? \
        ((bs)->bc--, (bs)->sr & 1) : \
            (((++((bs)->ptr) != (bs)->end) ? (void) 0 : (bs)->wrap (bs)), (bs)->bc = sizeof (*((bs)->ptr)) * 8 - 1, ((bs)->sr = *((bs)->ptr)) & 1) \
    ) ? \
        ((bs)->sr >>= 1, 1) : \
        ((bs)->sr >>= 1, 0) \
)

#define getbits(value, nbits, bs) do { \
    while ((nbits) > (bs)->bc) { \
        if (++((bs)->ptr) == (bs)->end) (bs)->wrap (bs); \
        (bs)->sr |= (uint32_t)*((bs)->ptr) << (bs)->bc; \
        (bs)->bc += sizeof (*((bs)->ptr)) * 8; \
    } \
    *(value) = (bs)->sr; \
    if ((bs)->bc > 32) { \
        (bs)->bc -= (nbits); \
        (bs)->sr = *((bs)->ptr) >> (sizeof (*((bs)->ptr)) * 8 - (bs)->bc); \
    } \
    else { \
        (bs)->bc -= (nbits); \
        (bs)->sr >>= (nbits); \
    } \
} while (0)

#define count_bits(av) ((av) ? 32 - __builtin_clz (av) : 0)

// ═══════════════════════════════════════════════════════
// EXACT WavPack entropy constants and macros
// ═══════════════════════════════════════════════════════

#define LIMIT_ONES 16
#define DIV0 128
#define DIV1 64
#define DIV2 32
#define CLEARA(destin) memset (destin, 0, sizeof (destin))

struct entropy_data {
    uint32_t median[3], slow_level, error_limit;
};

// Using pointer-to-struct for the macros (matches WavPack's c-> usage)
#define GET_MED(med) (((c->median [med]) >> 4) + 1)
#define INC_MED0() (c->median [0] += ((c->median [0] + DIV0) / DIV0) * 5)
#define DEC_MED0() (c->median [0] -= ((c->median [0] + (DIV0-2)) / DIV0) * 2)
#define INC_MED1() (c->median [1] += ((c->median [1] + DIV1) / DIV1) * 5)
#define DEC_MED1() (c->median [1] -= ((c->median [1] + (DIV1-2)) / DIV1) * 2)
#define INC_MED2() (c->median [2] += ((c->median [2] + DIV2) / DIV2) * 5)
#define DEC_MED2() (c->median [2] -= ((c->median [2] + (DIV2-2)) / DIV2) * 2)

// ═══════════════════════════════════════════════════════
// EXACT WavPack read_code function
// ═══════════════════════════════════════════════════════

static uint32_t read_code(Bitstream *bs, uint32_t maxcode)
{
    unsigned long local_sr;
    uint32_t extras, code;
    int bitcount;

    if (maxcode < 2)
        return maxcode ? getbit(bs) : 0;

    bitcount = count_bits(maxcode);
    extras = (1 << bitcount) - maxcode - 1;

    local_sr = bs->sr;

    while (bs->bc < bitcount) {
        if (++(bs->ptr) == bs->end)
            bs->wrap(bs);
        local_sr |= (long)*(bs->ptr) << bs->bc;
        bs->bc += sizeof(*(bs->ptr)) * 8;
    }

    if ((code = local_sr & ((1 << (bitcount - 1)) - 1)) >= extras)
        code = (code << 1) - extras + ((local_sr >> (bitcount - 1)) & 1);
    else
        bitcount--;

    if (sizeof(local_sr) < 8 && bs->bc > sizeof(local_sr) * 8) {
        bs->bc -= bitcount;
        bs->sr = *(bs->ptr) >> (sizeof(*(bs->ptr)) * 8 - bs->bc);
    }
    else {
        bs->bc -= bitcount;
        bs->sr = local_sr >> bitcount;
    }

    return code;
}

// ═══════════════════════════════════════════════════════
// EXACT get_words_lossless from WavPack (mono only, simplified)
// ═══════════════════════════════════════════════════════

int main(void)
{
    // Bitstream data from mono16_high test file
    unsigned char bs_data[] = {
        0xff, 0xff, 0x32, 0x7f, 0xe5, 0xff, 0xdf, 0x85,
        0x2f, 0xfe, 0x82, 0x5c, 0x19, 0x9c, 0x27, 0x2e,
        // padding
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    // Initialize Bitstream exactly as WavPack does:
    // ptr = data - 1 (getbit/getbits increment ptr before reading)
    Bitstream bs;
    bs.buf = bs_data;
    bs.end = bs_data + sizeof(bs_data);
    bs.ptr = bs_data - 1;    // KEY: one before data start
    bs.bc = 0;
    bs.sr = 0;
    bs.error = 0;
    bs.wrap = bs_wrap;

    // Two entropy channels (c[0] and c[1]), even for mono
    struct entropy_data w_c[2];
    memset(&w_c, 0, sizeof(w_c));

    // Initial medians for channel 0 (from test file entropy_vars)
    w_c[0].median[0] = 88;
    w_c[0].median[1] = 76;
    w_c[0].median[2] = 191;
    // Channel 1 medians stay 0 (mono)

    uint32_t holding_one = 0;
    int holding_zero = 0;
    uint32_t zeros_acc = 0;

    printf("Initial medians: [%u, %u, %u]\n", w_c[0].median[0], w_c[0].median[1], w_c[0].median[2]);
    printf("GET_MED: ");
    { struct entropy_data *c = &w_c[0]; printf("[%u, %u, %u]\n\n", GET_MED(0), GET_MED(1), GET_MED(2)); }

    int32_t buffer[20];
    int nsamples = 20;

    // EXACT get_words_lossless logic for MONO
    int32_t csamples;
    for (csamples = 0; csamples < nsamples; ++csamples) {
        struct entropy_data *c = &w_c[0];  // always channel 0 for mono
        uint32_t ones_count, low, high;

        int bit_before = bs.ptr - (bs_data - 1);  // approximate byte position

        // Handle holding_zero (forced zone-0 from previous iteration)
        if (holding_zero) {
            holding_zero = 0;
            low = read_code(&bs, GET_MED(0) - 1);
            DEC_MED0();
            int sign = getbit(&bs);
            buffer[csamples] = sign ? ~(int32_t)low : (int32_t)low;
            printf("Sample %d: holding_zero → low=%u sign=%d → %d  medians=[%u,%u,%u]\n",
                   csamples, low, sign, buffer[csamples],
                   w_c[0].median[0], w_c[0].median[1], w_c[0].median[2]);

            if (++csamples == nsamples)
                break;
            c = &w_c[0];
        }

        // Zero-run check
        if (w_c[0].median[0] < 2 && !holding_one && w_c[1].median[0] < 2) {
            uint32_t mask;
            int cbits;

            if (zeros_acc) {
                if (--zeros_acc) {
                    buffer[csamples] = 0;
                    printf("Sample %d: zeros_acc → 0\n", csamples);
                    continue;
                }
            }
            else {
                for (cbits = 0; cbits < 33 && getbit(&bs); ++cbits);

                if (cbits == 33)
                    break;

                if (cbits < 2)
                    zeros_acc = cbits;
                else {
                    for (mask = 1, zeros_acc = 0; --cbits; mask <<= 1)
                        if (getbit(&bs))
                            zeros_acc |= mask;
                    zeros_acc |= mask;
                }

                if (zeros_acc) {
                    CLEARA(w_c[0].median);
                    CLEARA(w_c[1].median);
                    buffer[csamples] = 0;
                    printf("Sample %d: zero-run=%u → 0\n", csamples, zeros_acc);
                    continue;
                }
            }
        }

        // Count ones (unary code) — simple version (no CTZ optimization)
        for (ones_count = 0; ones_count < (LIMIT_ONES + 1) && getbit(&bs); ++ones_count);

        if (ones_count >= LIMIT_ONES) {
            uint32_t mask;
            int cbits;

            if (ones_count == (LIMIT_ONES + 1))
                break;

            for (cbits = 0; cbits < 33 && getbit(&bs); ++cbits);

            if (cbits == 33)
                break;

            if (cbits < 2)
                ones_count = cbits;
            else {
                for (mask = 1, ones_count = 0; --cbits; mask <<= 1)
                    if (getbit(&bs))
                        ones_count |= mask;
                ones_count |= mask;
            }

            ones_count += LIMIT_ONES;
        }

        // Zone calculation (EXACT WavPack get_words_lossless logic)
        low = holding_one;
        holding_one = ones_count & 1;
        holding_zero = ~ones_count & 1;
        ones_count = (ones_count >> 1) + low;

        printf("Sample %d: raw_unary_related → ones_count=%u holding_one=%u holding_zero=%d\n",
               csamples, ones_count, holding_one, holding_zero);

        // Decode magnitude from zone
        if (ones_count == 0) {
            low = 0;
            high = GET_MED(0) - 1;
            DEC_MED0();
        }
        else {
            low = GET_MED(0);
            INC_MED0();

            if (ones_count == 1) {
                high = low + GET_MED(1) - 1;
                DEC_MED1();
            }
            else {
                low += GET_MED(1);
                INC_MED1();

                if (ones_count == 2) {
                    high = low + GET_MED(2) - 1;
                    DEC_MED2();
                }
                else {
                    low += (ones_count - 2) * GET_MED(2);
                    high = low + GET_MED(2) - 1;
                    INC_MED2();
                }
            }
        }

        printf("  zone=%u low=%u high=%u code_range=%u\n", ones_count, low, high, high - low);

        // read_code for lossless (error_limit == 0)
        low += read_code(&bs, high - low);

        // Sign bit
        int sign = getbit(&bs);
        buffer[csamples] = sign ? ~(int32_t)low : (int32_t)low;

        printf("  magnitude=%u sign=%d → result=%d  medians=[%u,%u,%u]\n",
               low, sign, buffer[csamples],
               w_c[0].median[0], w_c[0].median[1], w_c[0].median[2]);
    }

    printf("\n=== Residuals ===\n");
    for (int i = 0; i < csamples; i++)
        printf("  [%d] = %d\n", i, buffer[i]);

    printf("\nExpected (from our decoder): 91 48 -176 -16 52 4 10 14 -1 -14 ...\n");

    return 0;
}
