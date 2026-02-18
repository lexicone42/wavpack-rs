// Minimal standalone decorrelation verification program.
// Compiles with: gcc -o verify_decorr verify_decorr.c -O2
// Uses the exact same algorithm as the official WavPack decoder.
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define MAX_TERM 8
#define NPASS 10
#define NSAMP 10  // just trace first 10 samples

struct decorr_pass {
    int32_t term, delta, weight_A;
    int32_t samples_A[MAX_TERM];
};

// Exact WavPack apply_weight macro (i16 branching)
#define apply_weight_i(weight, sample) ((weight * (int32_t)(sample) + 512) >> 10)
#define apply_weight_f(weight, sample) ((((((sample) & 0xffff) * weight) >> 9) + \
    (((sample) >> 16) * weight) + 1) >> 1)
#define apply_weight(weight, sample) ((sample) != (short)(sample) ? \
    apply_weight_f(weight, sample) : apply_weight_i(weight, sample))

// Exact WavPack update_weight macro
#define update_weight(weight, delta, source, result) \
    if ((source) && (result)) (weight) -= ((((source ^ result) >> 30) & 2) - 1) * (delta);

static void decorr_mono_pass(struct decorr_pass *dpp, int32_t *buffer, int32_t sample_count)
{
    int32_t delta = dpp->delta, weight_A = dpp->weight_A;
    int32_t *bptr, *eptr = buffer + sample_count, sam_A;
    int m, k;

    switch (dpp->term) {
        case 17:
            for (bptr = buffer; bptr < eptr; bptr++) {
                sam_A = 2 * dpp->samples_A[0] - dpp->samples_A[1];
                dpp->samples_A[1] = dpp->samples_A[0];
                dpp->samples_A[0] = apply_weight(weight_A, sam_A) + bptr[0];
                update_weight(weight_A, delta, sam_A, bptr[0]);
                bptr[0] = dpp->samples_A[0];
            }
            break;

        case 18:
            for (bptr = buffer; bptr < eptr; bptr++) {
                sam_A = (3 * dpp->samples_A[0] - dpp->samples_A[1]) >> 1;
                dpp->samples_A[1] = dpp->samples_A[0];
                dpp->samples_A[0] = apply_weight(weight_A, sam_A) + bptr[0];
                update_weight(weight_A, delta, sam_A, bptr[0]);
                bptr[0] = dpp->samples_A[0];
            }
            break;

        default:
            for (m = 0, k = dpp->term & (MAX_TERM - 1), bptr = buffer;
                 bptr < eptr; bptr++) {
                sam_A = dpp->samples_A[m];
                dpp->samples_A[k] = apply_weight(weight_A, sam_A) + bptr[0];
                update_weight(weight_A, delta, sam_A, bptr[0]);
                bptr[0] = dpp->samples_A[k];
                m = (m + 1) & (MAX_TERM - 1);
                k = (k + 1) & (MAX_TERM - 1);
            }

            if (m) {
                int32_t temp_samples[MAX_TERM];
                memcpy(temp_samples, dpp->samples_A, sizeof(dpp->samples_A));
                for (k = 0; k < MAX_TERM; k++, m++)
                    dpp->samples_A[k] = temp_samples[m & (MAX_TERM - 1)];
            }
            break;
    }

    dpp->weight_A = weight_A;
}

// ENCODE direction: given original audio, produce residuals
// Apply passes in REVERSE of decode order (i.e., forward: pass[9] first, pass[0] last)
static void encode_mono_pass(struct decorr_pass *dpp, int32_t *buffer, int32_t sample_count)
{
    int32_t delta = dpp->delta, weight_A = dpp->weight_A;
    int32_t *bptr, *eptr = buffer + sample_count, sam_A;
    int m, k;

    switch (dpp->term) {
        case 17:
            for (bptr = buffer; bptr < eptr; bptr++) {
                sam_A = 2 * dpp->samples_A[0] - dpp->samples_A[1];
                dpp->samples_A[1] = dpp->samples_A[0];
                dpp->samples_A[0] = bptr[0];  // history = original sample
                bptr[0] -= apply_weight(weight_A, sam_A);  // residual = original - weighted
                update_weight(weight_A, delta, sam_A, bptr[0]);  // update with (pred, residual)
            }
            break;

        case 18:
            for (bptr = buffer; bptr < eptr; bptr++) {
                sam_A = (3 * dpp->samples_A[0] - dpp->samples_A[1]) >> 1;
                dpp->samples_A[1] = dpp->samples_A[0];
                dpp->samples_A[0] = bptr[0];
                bptr[0] -= apply_weight(weight_A, sam_A);
                update_weight(weight_A, delta, sam_A, bptr[0]);
            }
            break;

        default:
            for (m = 0, k = dpp->term & (MAX_TERM - 1), bptr = buffer;
                 bptr < eptr; bptr++) {
                sam_A = dpp->samples_A[m];
                dpp->samples_A[k] = bptr[0];
                bptr[0] -= apply_weight(weight_A, sam_A);
                update_weight(weight_A, delta, sam_A, bptr[0]);
                m = (m + 1) & (MAX_TERM - 1);
                k = (k + 1) & (MAX_TERM - 1);
            }
            if (m) {
                int32_t temp_samples[MAX_TERM];
                memcpy(temp_samples, dpp->samples_A, sizeof(dpp->samples_A));
                for (k = 0; k < MAX_TERM; k++, m++)
                    dpp->samples_A[k] = temp_samples[m & (MAX_TERM - 1)];
            }
            break;
    }

    dpp->weight_A = weight_A;
}

int main(void)
{
    // Exact values from the trace of test_mono16_high.wv
    // In byte-stream order: passes[0..9]
    // terms=[18, 18, 18, 1, 2, 3, 5, 1, 17, 4]
    // weights=[976, 1008, 589, -266, 540, 435, 258, -379, -48, 258]
    // Only pass[0] has non-zero initial samples: [-128, -192]

    // The WavPack reference reverses during parsing, so:
    // decorr_passes[0] = byte[9] (term=4, w=258)
    // decorr_passes[1] = byte[8] (term=17, w=-48)
    // ...
    // decorr_passes[9] = byte[0] (term=18, w=976, samples=[-128,-192])

    struct decorr_pass passes[NPASS];
    memset(passes, 0, sizeof(passes));

    // Set up in the WavPack REVERSED order (decorr_passes[0] = last byte stream term)
    // byte stream: [18, 18, 18, 1, 2, 3, 5, 1, 17, 4]
    // reversed:    [4, 17, 1, 5, 3, 2, 1, 18, 18, 18]
    int terms[]   = {4, 17, 1, 5, 3, 2, 1, 18, 18, 18};
    int weights[] = {258, -48, -379, 258, 435, 540, -266, 589, 1008, 976};
    int deltas[]  = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    for (int i = 0; i < NPASS; i++) {
        passes[i].term = terms[i];
        passes[i].delta = deltas[i];
        passes[i].weight_A = weights[i];
    }
    // Only pass[9] (byte[0], term=18) has non-zero samples
    passes[9].samples_A[0] = -128;
    passes[9].samples_A[1] = -192;

    // Entropy residuals (first NSAMP samples) from the test
    // We only know the first 3 from the trace: [91, 48, -176]
    // Let's use those and see if the first 3 match [0, 256, 511]
    int32_t buffer[NSAMP] = {91, 48, -176, 0, 0, 0, 0, 0, 0, 0};
    // We only check first 3

    printf("Input (entropy residuals): ");
    for (int i = 0; i < 3; i++) printf("%d ", buffer[i]);
    printf("\n\n");

    // Apply passes FORWARD (same as WavPack reference)
    for (int p = 0; p < NPASS; p++) {
        int32_t before[3] = {buffer[0], buffer[1], buffer[2]};
        decorr_mono_pass(&passes[p], buffer, 3);
        printf("Pass[%d] term=%d w=%d: [%d, %d, %d] -> [%d, %d, %d] w_after=%d\n",
               p, terms[p], weights[p],
               before[0], before[1], before[2],
               buffer[0], buffer[1], buffer[2],
               passes[p].weight_A);
    }

    printf("\nFinal output: [%d, %d, %d]\n", buffer[0], buffer[1], buffer[2]);
    printf("Expected:     [0, 256, 511]\n");

    // ---- ENCODE: reverse-engineer the correct residuals from expected output ----
    printf("\n=== ENCODING: expected output -> residuals ===\n");

    // Re-initialize passes (same parameters)
    memset(passes, 0, sizeof(passes));
    for (int i = 0; i < NPASS; i++) {
        passes[i].term = terms[i];
        passes[i].delta = deltas[i];
        passes[i].weight_A = weights[i];
    }
    passes[9].samples_A[0] = -128;
    passes[9].samples_A[1] = -192;

    int32_t enc_buffer[NSAMP] = {0, 256, 511, 765, 1016, 1262, 0, 0, 0, 0};
    // (first 6 expected values from the reference WAV for a sine wave)

    printf("Expected output: [%d, %d, %d, %d, %d, %d]\n",
           enc_buffer[0], enc_buffer[1], enc_buffer[2],
           enc_buffer[3], enc_buffer[4], enc_buffer[5]);

    // Encode: apply passes in REVERSE of decode order
    // Decode applies: pass[0], pass[1], ..., pass[9]
    // Encode applies: pass[9], pass[8], ..., pass[0]
    for (int p = NPASS - 1; p >= 0; p--) {
        int32_t before[6];
        memcpy(before, enc_buffer, sizeof(before));
        encode_mono_pass(&passes[p], enc_buffer, 6);
        printf("Enc pass[%d] term=%d: [%d, %d, %d] -> [%d, %d, %d]\n",
               p, terms[p],
               before[0], before[1], before[2],
               enc_buffer[0], enc_buffer[1], enc_buffer[2]);
    }

    printf("\nExpected residuals: [%d, %d, %d, %d, %d, %d]\n",
           enc_buffer[0], enc_buffer[1], enc_buffer[2],
           enc_buffer[3], enc_buffer[4], enc_buffer[5]);
    printf("Actual residuals:   [91, 48, -176, ...]\n");

    return 0;
}
