/* PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * DS2 component A -- CHUZC: the dual ratio test with bound flipping.
 *
 * This is an independent reimplementation of the *algorithm* described in
 * experiments/highs_study_S3_per_pivot_machinery_2026_07_25.md, produced
 * under the 2026-07-25 owner authorisation recorded in docs/PROVENANCE.md.
 * No HiGHS code was copied; the structure (geometric pre-filter, sweep-based
 * group partition, backward large-alpha group scan, batched flips) is the
 * understood algorithm, written from scratch against linprogx's own data
 * layout and sign conventions.
 *
 * The component is deliberately free of any dependency on _csparse.c, on
 * CPython, and on the LU: it consumes a pivot row and the nonbasic state and
 * returns a decision.  ds2_core owns everything else.
 */
#ifndef LINPROGX_DS2_CHUZC_H
#define LINPROGX_DS2_CHUZC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bound-status codes.  These MUST agree with DS_BOUND_* in _csparse.c so
 * ds2_core can hand its own bound_status array straight through. */
#define DS2_BOUND_LO 0    /* nonbasic at lower bound   */
#define DS2_BOUND_HI 1    /* nonbasic at upper bound   */
#define DS2_BOUND_FREE 2  /* nonbasic free at zero     */
#define DS2_BOUND_FIXED 3 /* lo == hi                  */
#define DS2_BOUND_BASIC 4 /* in the basis              */

/* ---- the fixed contract (docs/DS2-REWRITE.md) ------------------------- */

typedef struct {
    int32_t entering;         /* entering column, -1 if dual unbounded      */
    double theta_dual;        /* dual step                                  */
    double alpha_pivot;       /* pivot element                              */
    int32_t n_flip;           /* number of bound flips to apply             */
    const int32_t *flip_cols; /* columns to flip (owned by callee)          */
} DS2Entering;

/* ---- component-A state ------------------------------------------------
 *
 * `ratio_state` in the fixed signature is a DS2ChuzcState *.  The contract
 * declares it opaque and component-A-owned, which is where the two inputs
 * the signature does not carry live: the leaving row's primal infeasibility
 * (`delta`, which the bound-flipping test needs to know when to stop
 * walking breakpoints) and the LU age (`update_count`, which sets the
 * minimum-pivot admission threshold).  ds2_core writes these three fields
 * before each call and reads the statistics afterwards.
 */
typedef struct {
    /* ---- set once, at construction ---- */
    int32_t n_total;  /* structural + logical columns                     */
    int32_t capacity; /* candidate capacity; >= n_total                   */

    /* ---- written by ds2_core before every call ---- */
    double delta;         /* |primal infeasibility| of the leaving row    */
    int32_t update_count; /* LU updates since the last refactorisation    */
    double expand_tau;    /* EXPAND numerator relaxation (baseline only)  */

    /* ---- configuration (global; no per-problem tuning) ---- */
    /* Optional mask, length n_total.  no_flip[j] != 0 means column j must
     * never be flipped even though both its bounds are finite -- linprogx's
     * big-M artificial boxes are finite but meaningless as flip targets.
     * Such a column is given an infinite range, which is what makes the
     * breakpoint walk terminate on it instead of stepping over it. */
    const uint8_t *no_flip;
    double harris_delta; /* baseline Harris band width (default 1e-7)     */

    /* Cached u_j - l_j, +inf where the column must not flip.  Bounds change
     * only at a phase switch, so ds2_core builds this once per phase with
     * ds2_chuzc_build_range() and the ratio test then does ONE sequential
     * gather per candidate instead of two random loads from lo_ext/hi_ext
     * plus two isfinite tests.  NULL means "compute inline"; the decision
     * is identical either way. */
    double *range;
    int range_valid;

    /* ---- statistics, accumulated across calls ---- */
    int64_t n_call;         /* calls                                      */
    int64_t n_admitted;     /* candidates admitted in stage 0             */
    int64_t n_prefilter;    /* candidates surviving stage 1               */
    int64_t n_sweep_visits; /* per-candidate visits in stages 1+2         */
    int64_t n_group;        /* groups built                               */
    int64_t n_flip_total;   /* flips emitted                              */
    int64_t n_no_group;     /* calls that fell back to a global argmax    */

    /* ---- last-call census (why the flip set came out the size it did) --
     * Off by default: it is an extra O(candidates) pass and would pollute
     * the cycle measurement.  Set `census` only on untimed diagnostic runs. */
    int census;
    int32_t last_n_cand;      /* admitted candidates                      */
    int32_t last_n_flippable; /* of those, with a finite range            */
    double last_absorb;       /* sum alpha_j * range_j over those         */
    double last_delta;        /* the infeasibility they had to cover      */
    /* These four are free (set once per call, outside every loop). */
    int32_t last_stage1_take; /* candidates surviving the pre-filter      */
    double last_total_change; /* absorption accumulated by the partition  */
    int8_t last_exhausted;    /* partition ended with nothing left, not   */
                              /* because absorption covered delta         */
    int8_t last_degenerate;   /* the chosen step was zero                 */

    /* ---- owned scratch (do not touch) ---- */
    void *cand;         /* DS2Cand[capacity]                              */
    int32_t *group;     /* group boundaries                               */
    int32_t group_cap;  /* capacity of `group`                            */
    int32_t n_group_cur;/* boundaries used by the last call               */
    int32_t *flip_cols; /* int32_t[capacity]                              */
} DS2ChuzcState;

DS2ChuzcState *ds2_chuzc_state_new(int32_t n_total);
void ds2_chuzc_state_free(DS2ChuzcState *state);
void ds2_chuzc_state_reset_stats(DS2ChuzcState *state);

/* Build state->range from the current bounds.  Call once per solve and again
 * whenever the bounds change (a Phase 1 -> Phase 2 bound swap).  Optional:
 * without it the ratio test computes ranges inline. */
void ds2_chuzc_build_range(DS2ChuzcState *state, const double *lo_ext,
                           const double *hi_ext);
void ds2_chuzc_invalidate_range(DS2ChuzcState *state);

/* ---- the fixed entry point ------------------------------------------- */

/* Bound-flipping (longest-step) dual ratio test.
 *
 * alpha_row      dense, indexed by column: alpha_j = rho^T a_j.  Only the
 *                positions listed in alpha_pattern are read.
 * alpha_pattern  the nonzero positions of alpha_row, any order.
 * r_ext          reduced costs, indexed by column.
 * bound_status   DS2_BOUND_* per column.
 * lo_ext/hi_ext  bounds, indexed by column (+-inf allowed).
 * leaving_sigma  +1 if the leaving basic is below its lower bound,
 *                -1 if above its upper bound.
 * dual_tol       dual feasibility tolerance (HiGHS's Td).
 * ratio_state    DS2ChuzcState *, with delta/update_count set.
 */
DS2Entering ds2_chuzc(const double *alpha_row, const int32_t *alpha_pattern,
                      int32_t alpha_nnz, const double *r_ext,
                      const int8_t *bound_status, const double *lo_ext,
                      const double *hi_ext, int leaving_sigma, double dual_tol,
                      void *ratio_state);

/* ---- A/B controls ----------------------------------------------------
 *
 * Two reimplementations of the shipped Harris two-pass test, sharing the
 * fixed signature so a harness can call all three on identical inputs.
 *
 *   _harris_dense    scans columns 0..n_total-1, exactly as the shipped
 *                    dual simplex does (alpha_pattern is ignored).  This is
 *                    the faithful cost model of the incumbent.
 *   _harris_pattern  scans only alpha_pattern.  Isolates the algorithmic
 *                    difference from the scan-shape difference.
 *
 * Neither ever flips: n_flip is always 0.  Both honour state->expand_tau
 * and state->harris_delta so they reproduce the shipped decision.
 */
DS2Entering ds2_chuzc_harris_dense(const double *alpha_row,
                                   const int32_t *alpha_pattern,
                                   int32_t alpha_nnz, const double *r_ext,
                                   const int8_t *bound_status,
                                   const double *lo_ext, const double *hi_ext,
                                   int leaving_sigma, double dual_tol,
                                   void *ratio_state);

DS2Entering ds2_chuzc_harris_pattern(const double *alpha_row,
                                     const int32_t *alpha_pattern,
                                     int32_t alpha_nnz, const double *r_ext,
                                     const int8_t *bound_status,
                                     const double *lo_ext, const double *hi_ext,
                                     int leaving_sigma, double dual_tol,
                                     void *ratio_state);

#ifdef __cplusplus
}
#endif

#endif /* LINPROGX_DS2_CHUZC_H */
