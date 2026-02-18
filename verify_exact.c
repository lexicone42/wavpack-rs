// Exact WavPack reference decoder verification.
// Uses macros copied verbatim from wavpack_local.h
// Compiles with: gcc -o verify_exact verify_exact.c -O2
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define MAX_TERM 8

// ═══════════════════════════════════════════════════════
// EXACT WavPack reference macros from wavpack_local.h
// ═══════════════════════════════════════════════════════

#define apply_weight_i(weight, sample) ((weight * sample + 512) >> 10)

#define apply_weight_f(weight, sample) (((((sample & 0xffff) * weight) >> 9) + \
    (((sample & ~0xffff) >> 9) * weight) + 1) >> 1)

#define apply_weight(weight, sample) (sample != (int16_t) sample ? \
    apply_weight_f (weight, sample) : apply_weight_i (weight, sample))

#define update_weight(weight, delta, source, result) \
    if (source && result) { int32_t s = (int32_t) (source ^ result) >> 31; weight = (delta ^ s) + (weight - s); }

// ═══════════════════════════════════════════════════════
// EXACT decorr_mono_pass from WavPack unpack.c
// ═══════════════════════════════════════════════════════

struct decorr_pass {
    int32_t term, delta, weight_A;
    int32_t samples_A[MAX_TERM];
};

static void decorr_mono_pass(struct decorr_pass *dpp, int32_t *buffer, int32_t sample_count)
{
    int32_t delta = dpp->delta, weight_A = dpp->weight_A;
    int32_t *bptr, *eptr = buffer + sample_count, sam_A;
    int m, k;

    switch (dpp->term) {
        case 17:
            for (bptr = buffer; bptr < eptr; bptr++) {
                sam_A = 2 * dpp->samples_A [0] - dpp->samples_A [1];
                dpp->samples_A [1] = dpp->samples_A [0];
                dpp->samples_A [0] = apply_weight (weight_A, sam_A) + bptr [0];
                update_weight (weight_A, delta, sam_A, bptr [0]);
                bptr [0] = dpp->samples_A [0];
            }
            break;

        case 18:
            for (bptr = buffer; bptr < eptr; bptr++) {
                sam_A = (3 * dpp->samples_A [0] - dpp->samples_A [1]) >> 1;
                dpp->samples_A [1] = dpp->samples_A [0];
                dpp->samples_A [0] = apply_weight (weight_A, sam_A) + bptr [0];
                update_weight (weight_A, delta, sam_A, bptr [0]);
                bptr [0] = dpp->samples_A [0];
            }
            break;

        default:
            for (m = 0, k = dpp->term & (MAX_TERM - 1), bptr = buffer; bptr < eptr; bptr++) {
                sam_A = dpp->samples_A [m];
                dpp->samples_A [k] = apply_weight (weight_A, sam_A) + bptr [0];
                update_weight (weight_A, delta, sam_A, bptr [0]);
                bptr [0] = dpp->samples_A [k];
                m = (m + 1) & (MAX_TERM - 1);
                k = (k + 1) & (MAX_TERM - 1);
            }

            if (m) {
                int32_t temp_samples [MAX_TERM];
                memcpy (temp_samples, dpp->samples_A, sizeof (dpp->samples_A));
                for (k = 0; k < MAX_TERM; k++, m++)
                    dpp->samples_A [k] = temp_samples [m & (MAX_TERM - 1)];
            }
            break;
    }

    dpp->weight_A = weight_A;
}

// ═══════════════════════════════════════════════════════
// Exact wp_exp2s from WavPack entropy_utils.c
// ═══════════════════════════════════════════════════════

static const unsigned char exp2_table [] = {
    0x00, 0x01, 0x01, 0x02, 0x03, 0x03, 0x04, 0x05, 0x06, 0x06, 0x07, 0x08, 0x08, 0x09, 0x0a, 0x0b,
    0x0b, 0x0c, 0x0d, 0x0e, 0x0e, 0x0f, 0x10, 0x10, 0x11, 0x12, 0x13, 0x13, 0x14, 0x15, 0x16, 0x16,
    0x17, 0x18, 0x19, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1d, 0x1e, 0x1f, 0x20, 0x20, 0x21, 0x22, 0x23,
    0x24, 0x24, 0x25, 0x26, 0x27, 0x28, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2c, 0x2d, 0x2e, 0x2f, 0x30,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3a, 0x3b, 0x3c, 0x3d,
    0x3e, 0x3f, 0x40, 0x41, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x48, 0x49, 0x4a, 0x4b,
    0x4c, 0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a,
    0x5b, 0x5c, 0x5d, 0x5e, 0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x87, 0x88, 0x89, 0x8a,
    0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b,
    0x9c, 0x9d, 0x9f, 0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad,
    0xaf, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc8, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf, 0xd0, 0xd2, 0xd3, 0xd4,
    0xd6, 0xd7, 0xd8, 0xd9, 0xdb, 0xdc, 0xdd, 0xde, 0xe0, 0xe1, 0xe2, 0xe4, 0xe5, 0xe6, 0xe8, 0xe9,
    0xea, 0xec, 0xed, 0xee, 0xf0, 0xf1, 0xf2, 0xf4, 0xf5, 0xf6, 0xf8, 0xf9, 0xfa, 0xfc, 0xfd, 0xff
};

int32_t wp_exp2s(int log)
{
    uint32_t value;
    if (log < 0)
        return ~((uint32_t) wp_exp2s(-log) - 1);
    value = exp2_table[log & 0xff] | 0x100;
    if ((log >>= 8) <= 9)
        return value >> (9 - log);
    else
        return value << ((log - 9) & 0x1f);
}

// Exact restore_weight from WavPack
int32_t restore_weight(signed char stored)
{
    int weight = (int) stored << 3;
    if (weight > 0)
        weight += (weight + 64) >> 7;
    else if (weight < 0)
        weight -= (-weight + 64) >> 7;
    return weight;
}

// ═══════════════════════════════════════════════════════
// Parse sub-blocks exactly as the WavPack reference does
// (terms/weights/samples stored in REVERSED order)
// ═══════════════════════════════════════════════════════

int main(void)
{
    // Raw sub-block data from the mono16_high test file trace
    unsigned char raw_terms[10] = {0x57, 0x57, 0x57, 0x46, 0x47, 0x48, 0x4a, 0x46, 0x56, 0x49};
    unsigned char raw_weights[10] = {0x79, 0x7d, 0x49, 0xdf, 0x43, 0x36, 0x20, 0xd1, 0xfa, 0x20};
    unsigned char raw_samples[4] = {0x00, 0xf8, 0x6a, 0xf7};

    int num_terms = 10;
    struct decorr_pass passes[16];
    memset(passes, 0, sizeof(passes));

    // Parse terms in REVERSE order (exactly like WavPack read_decorr_terms)
    printf("=== Parsing terms (WavPack reversed order) ===\n");
    for (int i = 0; i < num_terms; i++) {
        struct decorr_pass *dpp = &passes[num_terms - 1 - i];
        dpp->term = (int)(raw_terms[i] & 0x1f) - 5;
        dpp->delta = (raw_terms[i] >> 5) & 0x7;
        printf("  byte[%d] -> passes[%d]: term=%d delta=%d\n", i, num_terms-1-i, dpp->term, dpp->delta);
    }

    // Parse weights in REVERSE order (exactly like WavPack read_decorr_weights)
    // First zero all weights
    for (int i = 0; i < num_terms; i++)
        passes[i].weight_A = 0;

    // Then read from end to beginning
    printf("\n=== Parsing weights (WavPack reversed order) ===\n");
    int widx = 0;
    for (int i = num_terms - 1; i >= 0 && widx < 10; i--) {
        passes[i].weight_A = restore_weight((signed char)raw_weights[widx]);
        printf("  byte[%d] (0x%02x=%d) -> passes[%d].w=%d\n",
               widx, raw_weights[widx], (signed char)raw_weights[widx], i, passes[i].weight_A);
        widx++;
    }

    // Parse samples in REVERSE order (exactly like WavPack read_decorr_samples)
    printf("\n=== Parsing samples (WavPack reversed order) ===\n");
    unsigned char *byteptr = raw_samples;
    unsigned char *endptr = raw_samples + sizeof(raw_samples);
    // Start from passes[num_terms-1] going down
    for (int i = num_terms - 1; i >= 0 && byteptr < endptr; i--) {
        struct decorr_pass *dpp = &passes[i];
        if (dpp->term > MAX_TERM) {
            // Terms 17/18: read 2 samples
            if (byteptr + 4 <= endptr) {
                dpp->samples_A[0] = wp_exp2s((int16_t)(byteptr[0] + (byteptr[1] << 8)));
                dpp->samples_A[1] = wp_exp2s((int16_t)(byteptr[2] + (byteptr[3] << 8)));
                printf("  passes[%d] term=%d: samples=[%d, %d]\n",
                       i, dpp->term, dpp->samples_A[0], dpp->samples_A[1]);
                byteptr += 4;
            }
        } else if (dpp->term < 0) {
            // skip for mono
        } else {
            // Terms 1-8: read term samples
            int cnt = dpp->term;
            int m = 0;
            while (cnt-- && byteptr + 2 <= endptr) {
                dpp->samples_A[m] = wp_exp2s((int16_t)(byteptr[0] + (byteptr[1] << 8)));
                byteptr += 2;
                m++;
            }
            if (m > 0)
                printf("  passes[%d] term=%d: %d samples read\n", i, dpp->term, m);
        }
    }

    // Print parsed state
    printf("\n=== Parsed decorr_passes (WavPack reference order) ===\n");
    for (int i = 0; i < num_terms; i++) {
        printf("  passes[%d]: term=%d delta=%d w=%d samples=[%d, %d]\n",
               i, passes[i].term, passes[i].delta, passes[i].weight_A,
               passes[i].samples_A[0], passes[i].samples_A[1]);
    }

    // Entropy residuals from our trace
    int32_t buffer[10] = {91, 48, -176, -16, 52, 4, 10, 14, -1, -14};
    int nsamp = 10;

    printf("\n=== Applying decorrelation (forward through passes) ===\n");
    printf("Input residuals: ");
    for (int i = 0; i < nsamp; i++) printf("%d ", buffer[i]);
    printf("\n\n");

    // Apply passes FORWARD (same as WavPack reference: passes[0] first)
    for (int p = 0; p < num_terms; p++) {
        int32_t pre[3] = {buffer[0], buffer[1], buffer[2]};
        decorr_mono_pass(&passes[p], buffer, nsamp);
        printf("Pass[%d] term=%2d w_init=%-5d: [%d, %d, %d] -> [%d, %d, %d] w_after=%d\n",
               p, passes[p].term, 0 /* already updated */, pre[0], pre[1], pre[2],
               buffer[0], buffer[1], buffer[2], passes[p].weight_A);
    }

    printf("\n=== Final output ===\n");
    printf("Decoded: ");
    for (int i = 0; i < nsamp; i++) printf("%d ", buffer[i]);
    printf("\n");
    printf("Expected: 0 256 511 765 1016 1262 ...\n");

    // Check first few diffs
    int expected[] = {0, 256, 511, 765, 1016, 1262, 1504, 1740, 1968, 2189};
    printf("\nDiffs: ");
    for (int i = 0; i < nsamp; i++) printf("%d ", buffer[i] - expected[i]);
    printf("\n");

    return 0;
}
