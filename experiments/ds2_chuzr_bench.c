/* Cost + selection harness for the DS2 CHUZR component.
 *
 * PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * Loaded by experiments/ds2_chuzr_validate.py through ctypes. It brackets
 * every candidate CHUZR implementation with rdtsc so the comparison is in
 * cycles, which are load-invariant on this box (docs/DS2-REWRITE.md,
 * "Measurement discipline"): wall time drifts 4-19% here and is unusable.
 *
 * The reference scans are transcribed from the SHIPPED dual simplex
 * (src/linprogx/_csparse.c:14566-14606 scalar, :12647-12746 AVX2) so that
 * "cost versus the existing dense scan" means the actual existing scan, not
 * a strawman. The shipped file itself is not modified or linked.
 *
 * Order bias: on each call the variants are run in an order that alternates
 * with the pivot index, so cold-cache advantage does not accrue to one side.
 */

#include "../src/linprogx/_ds2_chuzr.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#define DS2_X86 1
#include <x86intrin.h>
#else
#define DS2_X86 0
#endif

static inline uint64_t ds2_cycles(void)
{
#if DS2_X86
    unsigned aux;
    _mm_lfence();
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
#else
    return 0;
#endif
}

/* ---- reference scans (shipped behaviour) ------------------------------ */

/* Scalar dense scan, arithmetic identical to _csparse.c:14566-14606.
 * rule 0 = Dantzig (max violation), rule 1 = DSE (violation^2 / weight). */
static DS2Leaving ref_scalar(
    const int32_t *basis, const double *x_B, const double *lo_ext,
    const double *hi_ext, const double *weights, int32_t m, double tol,
    int rule, double floor_w)
{
    DS2Leaving out = {-1, 0, 0.0};
    double best = 0.0;
    for (int32_t k = 0; k < m; k++) {
        int32_t j = basis[k];
        double viol = 0.0;
        int sigma = 0;
        if (isfinite(lo_ext[j]) && x_B[k] < lo_ext[j] - tol) {
            viol = lo_ext[j] - x_B[k];
            sigma = 1;
        }
        if (isfinite(hi_ext[j]) && x_B[k] > hi_ext[j] + tol) {
            double v2 = x_B[k] - hi_ext[j];
            if (v2 > viol) { viol = v2; sigma = -1; }
        }
        if (viol > 0.0) {
            double score;
            if (rule == DS2_RULE_DANTZIG) {
                score = viol;
            } else {
                double w = weights[k];
                if (w < floor_w) w = floor_w;
                score = (viol * viol) / w;
            }
            if (score > best) {
                best = score;
                out.basis_pos = k;
                out.sigma = sigma;
                out.violation = viol;
            }
        }
    }
    return out;
}

#if DS2_X86
/* AVX2 dense scan, transcribed from _csparse.c ds_price_avx2 (:12647-12746),
 * including its lowest-index lane tie-break. */
__attribute__((target("avx2")))
static DS2Leaving ref_avx2(
    const int32_t *basis, const double *x_B, const double *lo_ext,
    const double *hi_ext, const double *weights, int32_t m, double tol,
    int rule, double floor_w)
{
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);
    const __m256d tol_v = _mm256_set1_pd(tol);
    const __m256d wfloor = _mm256_set1_pd(floor_w);
    const __m256d no_index = _mm256_set1_pd(-1.0);
    __m256d lane_best_score = zero;
    __m256d lane_best_index = no_index;
    __m256d lane_best_sigma = zero;
    __m256d lane_best_viol = zero;
    int32_t k = 0;

    for (; k + 4 <= m; k += 4) {
        __m128i bidx = _mm_loadu_si128((const __m128i *)(basis + k));
        __m256d x = _mm256_loadu_pd(x_B + k);
        __m256d lo = _mm256_i32gather_pd(lo_ext, bidx, 8);
        __m256d hi = _mm256_i32gather_pd(hi_ext, bidx, 8);
        __m256d lo_mask =
            _mm256_cmp_pd(x, _mm256_sub_pd(lo, tol_v), _CMP_LT_OQ);
        __m256d hi_mask =
            _mm256_cmp_pd(x, _mm256_add_pd(hi, tol_v), _CMP_GT_OQ);
        __m256d lo_viol = _mm256_sub_pd(lo, x);
        __m256d hi_viol = _mm256_sub_pd(x, hi);
        __m256d viol = _mm256_blendv_pd(zero, lo_viol, lo_mask);
        __m256d sigma = _mm256_blendv_pd(zero, one, lo_mask);
        __m256d take_hi = _mm256_and_pd(
            hi_mask, _mm256_cmp_pd(hi_viol, viol, _CMP_GT_OQ));
        viol = _mm256_blendv_pd(viol, hi_viol, take_hi);
        sigma = _mm256_blendv_pd(sigma, neg_one, take_hi);

        __m256d score;
        if (rule == DS2_RULE_DANTZIG) {
            score = viol;
        } else {
            __m256d w = _mm256_loadu_pd(weights + k);
            w = _mm256_max_pd(w, wfloor);
            score = _mm256_div_pd(_mm256_mul_pd(viol, viol), w);
        }
        __m256d better = _mm256_cmp_pd(score, lane_best_score, _CMP_GT_OQ);
        __m256d index = _mm256_setr_pd((double)k, (double)(k + 1),
                                       (double)(k + 2), (double)(k + 3));
        lane_best_score = _mm256_blendv_pd(lane_best_score, score, better);
        lane_best_index = _mm256_blendv_pd(lane_best_index, index, better);
        lane_best_sigma = _mm256_blendv_pd(lane_best_sigma, sigma, better);
        lane_best_viol = _mm256_blendv_pd(lane_best_viol, viol, better);
    }

    double scores[4], indices[4], sigmas[4], viols[4];
    _mm256_storeu_pd(scores, lane_best_score);
    _mm256_storeu_pd(indices, lane_best_index);
    _mm256_storeu_pd(sigmas, lane_best_sigma);
    _mm256_storeu_pd(viols, lane_best_viol);
    DS2Leaving out = {-1, 0, 0.0};
    double best = 0.0;
    for (int lane = 0; lane < 4; lane++) {
        int32_t index = (int32_t)indices[lane];
        if (index < 0) continue;
        if (scores[lane] > best ||
            (scores[lane] == best && (out.basis_pos < 0 || index < out.basis_pos))) {
            best = scores[lane];
            out.basis_pos = index;
            out.sigma = (int)sigmas[lane];
            out.violation = viols[lane];
        }
    }
    for (; k < m; k++) {
        int32_t j = basis[k];
        double viol = 0.0;
        int sigma = 0;
        if (isfinite(lo_ext[j]) && x_B[k] < lo_ext[j] - tol) {
            viol = lo_ext[j] - x_B[k];
            sigma = 1;
        }
        if (isfinite(hi_ext[j]) && x_B[k] > hi_ext[j] + tol) {
            double v2 = x_B[k] - hi_ext[j];
            if (v2 > viol) { viol = v2; sigma = -1; }
        }
        if (viol > 0.0) {
            double score;
            if (rule == DS2_RULE_DANTZIG) {
                score = viol;
            } else {
                double w = weights[k];
                if (w < floor_w) w = floor_w;
                score = (viol * viol) / w;
            }
            if (score > best) {
                best = score;
                out.basis_pos = k;
                out.sigma = sigma;
                out.violation = viol;
            }
        }
    }
    return out;
}
#endif

/* Measured inside the same rotation as the real variants, so the report can
 * show the in-situ floor of one rdtscp bracket. Under load, and with the
 * caches cold from the surrounding solver work, that floor is not the ~90
 * ticks a tight calibration loop reports. Nothing is subtracted
 * automatically; the floor is reported alongside so it can be judged. */
__attribute__((noinline)) static int ds2_noop(int x) { return x + 1; }

int ds2_bench_has_avx2(void)
{
#if DS2_X86
    return __builtin_cpu_supports("avx2") != 0;
#else
    return 0;
#endif
}

/* ---- harness state ----------------------------------------------------- */

typedef struct {
    DS2Pricing *pricing;
    int32_t m;
    int     rule;
    double  floor_w;

    uint64_t cyc_ds2;         /* ds2_chuzr                                */
    uint64_t cyc_ds2_update;  /* ds2_chuzr_rows_changed                   */
    uint64_t cyc_scalar;      /* shipped scalar dense scan                */
    uint64_t cyc_avx2;        /* shipped AVX2 dense scan                  */
    long long n_calls;
    long long n_updates;

    long long agree_scalar;   /* ds2 picked the same basis position       */
    long long agree_merit;    /* ds2's merit matched the dense argmax      */
    long long disagree_merit; /* ds2's merit was strictly worse           */
    long long both_none;
    long long one_none;

    double *w_ref;            /* private weight copy for the reference     */

    /* Per-call samples. This box is heavily loaded: a measured region that
     * spans a context switch reads tens of thousands of extra TSC ticks, and
     * a handful of those dominate any SUM. Medians over the samples are what
     * the report quotes; the sums are kept only as a cross-check. */
    double *s_ds2, *s_upd, *s_sca, *s_avx, *s_nop;
    int32_t n_samp, cap_samp;
} DS2Bench;

#define DS2_SAMPLE_CAP 200000

void ds2_bench_free(DS2Bench *b);

DS2Bench *ds2_bench_new(int32_t m, int rule, double floor_w, uint64_t seed,
                        int random_start)
{
    DS2Bench *b = (DS2Bench *)calloc(1, sizeof(DS2Bench));
    if (b == NULL) return NULL;
    b->pricing = ds2_pricing_new(m, rule);
    b->w_ref = (double *)malloc((size_t)m * sizeof(double));
    b->cap_samp = DS2_SAMPLE_CAP;
    b->s_ds2 = (double *)malloc((size_t)b->cap_samp * sizeof(double));
    b->s_upd = (double *)malloc((size_t)b->cap_samp * sizeof(double));
    b->s_sca = (double *)malloc((size_t)b->cap_samp * sizeof(double));
    b->s_avx = (double *)malloc((size_t)b->cap_samp * sizeof(double));
    b->s_nop = (double *)malloc((size_t)b->cap_samp * sizeof(double));
    if (b->pricing == NULL || b->w_ref == NULL || b->s_ds2 == NULL ||
        b->s_upd == NULL || b->s_sca == NULL || b->s_avx == NULL ||
        b->s_nop == NULL) {
        ds2_bench_free(b);
        return NULL;
    }
    b->m = m;
    b->rule = rule;
    b->floor_w = floor_w;
    ds2_pricing_set_weight_floor(b->pricing, floor_w);
    ds2_pricing_set_seed(b->pricing, seed);
    ds2_pricing_set_random_start(b->pricing, random_start);
    return b;
}

void ds2_bench_free(DS2Bench *b)
{
    if (b == NULL) return;
    if (b->pricing != NULL) ds2_pricing_free(b->pricing);
    free(b->w_ref);
    free(b->s_ds2);
    free(b->s_upd);
    free(b->s_sca);
    free(b->s_avx);
    free(b->s_nop);
    free(b);
}

/* Copy one sample series out for the Python side. which: 0 ds2, 1 update,
 * 2 scalar reference, 3 AVX2 reference. Returns the number written. */
int32_t ds2_bench_samples(DS2Bench *b, int which, double *out, int32_t max_n)
{
    const double *src = (which == 0)   ? b->s_ds2
                        : (which == 1) ? b->s_upd
                        : (which == 2) ? b->s_sca
                        : (which == 3) ? b->s_avx
                                       : b->s_nop;
    int32_t n = b->n_samp < max_n ? b->n_samp : max_n;
    memcpy(out, src, (size_t)n * sizeof(double));
    return n;
}

void ds2_bench_set_paranoid(DS2Bench *b, int on)
{
    ds2_pricing_set_paranoid(b->pricing, on);
}

void ds2_bench_set_column_density(DS2Bench *b, double d)
{
    ds2_pricing_set_column_density(b->pricing, d);
}

void ds2_bench_set_list_mode(DS2Bench *b, int mode)
{
    ds2_pricing_set_list_mode(b->pricing, mode);
}

void ds2_bench_set_cutoff_enabled(DS2Bench *b, int on)
{
    ds2_pricing_set_cutoff_enabled(b->pricing, on);
}

int32_t ds2_bench_audit(DS2Bench *b, const int32_t *basis, const double *x_B,
                        const double *lo, const double *hi, double tol,
                        int32_t *first_bad)
{
    return ds2_chuzr_audit(b->pricing, basis, x_B, lo, hi, tol, 1e-12,
                           first_bad);
}

void ds2_bench_invalidate(DS2Bench *b) { ds2_chuzr_invalidate(b->pricing); }

void ds2_bench_rows_changed(DS2Bench *b, const int32_t *rows, int32_t n_rows,
                            const int32_t *basis, const double *x_B,
                            const double *lo, const double *hi, double tol)
{
    uint64_t t0 = ds2_cycles();
    ds2_chuzr_rows_changed(b->pricing, rows, n_rows, basis, x_B, lo, hi, tol);
    uint64_t t1 = ds2_cycles();
    b->cyc_ds2_update += t1 - t0;
    b->n_updates++;
    /* Charge the update to the call it followed. */
    if (b->n_samp > 0) b->s_upd[b->n_samp - 1] = (double)(t1 - t0);
}

/* One CHUZR at one real state. Runs the DS2 component and both reference
 * dense scans, alternating which goes first with the pivot parity.
 * out[0..2] = ds2 (basis_pos, sigma, violation)
 * out[3..5] = scalar reference
 * out[6..8] = avx2 reference (or the scalar result if AVX2 is unavailable) */
void ds2_bench_call(DS2Bench *b, int64_t pivot,
                    const int32_t *basis, const double *x_B,
                    const double *lo, const double *hi, double *weights,
                    double tol, double *out)
{
    const int32_t m = b->m;
    memcpy(b->w_ref, weights, (size_t)m * sizeof(double));

    DS2Leaving r_ds2 = {-1, 0, 0.0};
    DS2Leaving r_sca = {-1, 0, 0.0};
    DS2Leaving r_avx = {-1, 0, 0.0};
    uint64_t t0, t1;

    /* CHUZR runs right after BTRAN/FTRAN have thrashed the cache, so in the
     * real solver the first thing to touch these arrays pays the misses.
     * Whichever variant runs first here inherits that cost, so the three
     * variants take turns at going first, one third of the calls each. */
    const int have_avx = ds2_bench_has_avx2();
    const int lead = (int)(pivot % 4);
    uint64_t c_ds2 = 0, c_sca = 0, c_avx = 0, c_nop = 0;
    volatile int nop_sink = 0;
    for (int step = 0; step < 4; step++) {
        switch ((lead + step) % 4) {
        case 0:
            t0 = ds2_cycles();
            r_ds2 = ds2_chuzr(basis, x_B, lo, hi, weights, m, tol, b->pricing);
            t1 = ds2_cycles();
            c_ds2 = t1 - t0;
            break;
        case 1:
            t0 = ds2_cycles();
            r_sca = ref_scalar(basis, x_B, lo, hi, b->w_ref, m, tol, b->rule,
                               b->floor_w);
            t1 = ds2_cycles();
            c_sca = t1 - t0;
            break;
        case 2:
#if DS2_X86
            if (have_avx) {
                t0 = ds2_cycles();
                r_avx = ref_avx2(basis, x_B, lo, hi, b->w_ref, m, tol,
                                 b->rule, b->floor_w);
                t1 = ds2_cycles();
                c_avx = t1 - t0;
            }
#endif
            break;
        default:
            t0 = ds2_cycles();
            nop_sink = ds2_noop(nop_sink);
            t1 = ds2_cycles();
            c_nop = t1 - t0;
            break;
        }
    }
    if (!have_avx) r_avx = r_sca;
    b->cyc_ds2 += c_ds2;
    b->cyc_scalar += c_sca;
    b->cyc_avx2 += c_avx;
    if (b->n_samp < b->cap_samp) {
        int32_t i = b->n_samp++;
        b->s_ds2[i] = (double)c_ds2;
        b->s_sca[i] = (double)c_sca;
        b->s_avx[i] = (double)c_avx;
        b->s_nop[i] = (double)c_nop;
        b->s_upd[i] = 0.0;
    }

    b->n_calls++;
    if (r_ds2.basis_pos < 0 && r_sca.basis_pos < 0) {
        b->both_none++;
    } else if (r_ds2.basis_pos < 0 || r_sca.basis_pos < 0) {
        b->one_none++;
    } else {
        if (r_ds2.basis_pos == r_sca.basis_pos) b->agree_scalar++;
        double mr, md;
        if (b->rule == DS2_RULE_DANTZIG) {
            mr = r_sca.violation;
            md = r_ds2.violation;
        } else {
            double wr = b->w_ref[r_sca.basis_pos];
            double wd = b->w_ref[r_ds2.basis_pos];
            if (wr < b->floor_w) wr = b->floor_w;
            if (wd < b->floor_w) wd = b->floor_w;
            mr = r_sca.violation * r_sca.violation / wr;
            md = r_ds2.violation * r_ds2.violation / wd;
        }
        if (md >= mr * (1.0 - 1e-12)) b->agree_merit++;
        else b->disagree_merit++;
    }

    out[0] = (double)r_ds2.basis_pos;
    out[1] = (double)r_ds2.sigma;
    out[2] = r_ds2.violation;
    out[3] = (double)r_sca.basis_pos;
    out[4] = (double)r_sca.sigma;
    out[5] = r_sca.violation;
    out[6] = (double)r_avx.basis_pos;
    out[7] = (double)r_avx.sigma;
    out[8] = r_avx.violation;
}

/* Report: cycles and counters, packed for ctypes. */
void ds2_bench_report(DS2Bench *b, double *out)
{
    const DS2ChuzrStats *s = ds2_chuzr_stats(b->pricing);
    out[0] = (double)b->cyc_ds2;
    out[1] = (double)b->cyc_scalar;
    out[2] = (double)b->cyc_avx2;
    out[3] = (double)b->n_calls;
    out[4] = (double)b->agree_scalar;
    out[5] = (double)b->agree_merit;
    out[6] = (double)b->disagree_merit;
    out[7] = (double)b->both_none;
    out[8] = (double)b->one_none;
    out[9] = (double)s->rebuilds;
    out[10] = (double)s->scanned;
    out[11] = (double)s->dense_scanned;
    out[12] = (double)s->dense_calls;
    out[13] = (double)s->cutoff_misses;
    out[14] = (double)s->list_len_sum;
    out[15] = (double)s->paranoid_mismatch;
    out[16] = (double)b->cyc_ds2_update;
    out[17] = (double)b->n_updates;
    out[18] = (double)s->changed_rows;
    out[19] = (double)b->m;
    out[20] = (double)s->recomputes;
    out[21] = (double)s->cutoff_installed;
    out[22] = (double)s->infeas_sum;
}
#define DS2_BENCH_REPORT_LEN 23

/* Straight-line cycle cost of a fixed number of empty rdtsc brackets, so the
 * Python side can subtract the measurement overhead from short calls. */
double ds2_bench_timer_overhead(int64_t reps)
{
    uint64_t total = 0;
    for (int64_t i = 0; i < reps; i++) {
        uint64_t t0 = ds2_cycles();
        uint64_t t1 = ds2_cycles();
        total += t1 - t0;
    }
    return (double)total / (double)(reps > 0 ? reps : 1);
}

/* TSC ticks elapsed, for frequency calibration against a Python wall clock. */
double ds2_bench_tsc_now(void) { return (double)ds2_cycles(); }
