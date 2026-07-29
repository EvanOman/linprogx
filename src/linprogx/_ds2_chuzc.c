/* PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * DS2 component A -- CHUZC: the dual ratio test with bound flipping.
 *
 * WHAT THIS IS
 * ------------
 * The dual objective, as a function of the dual step theta along the leaving
 * row's direction, is piecewise linear and concave.  Its breakpoints are the
 * ratios r_j / alpha_j of the admissible nonbasic columns; at each breakpoint
 * one column's reduced cost changes sign, and if that column is BOXED it can
 * simply be flipped to its opposite bound, which absorbs |alpha_j| * (u_j-l_j)
 * of the leaving row's primal infeasibility and lets the walk continue.  The
 * longest-step (bound-flipping) test walks breakpoints until the accumulated
 * absorption covers the infeasibility, flips everything it stepped over, and
 * pivots on the last one.  One dual iteration then does the work of several.
 *
 * WHY THIS IMPLEMENTATION AND NOT THE EXISTING ONE
 * ------------------------------------------------
 * linprogx already has a bound-flipping ratio test (`bfrt=1`).  Measured, it
 * reduces pivots (25fv47 -10.2%, sierra -11.0%, greenbea -2.3%) and loses
 * wall on every instance, because it qsorts every admissible candidate every
 * pivot to get them in ratio order.  HiGHS's version never sorts.  It gets
 * the breakpoints in *approximate* order by two sweep-based partitions:
 *
 *   Stage 1  a geometric pre-filter that repeatedly multiplies a threshold by
 *            10 and partitions the candidates against it, stopping as soon as
 *            the collected prefix can absorb the whole infeasibility.  This
 *            bounds the work before any breakpoint structure exists.
 *   Stage 2  a group partition: repeatedly partition the survivors against a
 *            threshold, and set the next threshold to the smallest ratio
 *            among the *rejected* ones.  Each pass emits one group.  Groups
 *            are ordered by ratio; members within a group are not.  That is
 *            exactly as much order as the algorithm needs, and it costs
 *            O(passes x candidates) with no comparison sort.
 *   Stage 3  choose the group: scan groups from the LAST backwards, take the
 *            largest |alpha| in each, accept the first group whose best pivot
 *            exceeds 10% of the largest pivot anywhere.  Longest step first,
 *            with a *relative* stability guard -- the incumbent instead stops
 *            at the FIRST breakpoint that kills the slope.
 *   Stage 4  everything in a group strictly before the chosen one flips.
 *
 * DIFFERENCES FROM HiGHS, DELIBERATE
 * ----------------------------------
 *  - Candidates are held in a 32-byte struct-of-everything (alpha, tight,
 *    range, col, move) built once at admission, so the sweeps in stages 1
 *    and 2 touch one contiguous array and never chase workDual/workRange
 *    indirections.  HiGHS re-reads workMove[iCol] * workDual[iCol] and
 *    workRange[iCol] on every pass.  Same arithmetic, one cache stream.
 *  - Ties on |alpha| in stage 3 resolve to the lowest column index.  HiGHS
 *    uses a stored random permutation.  Determinism is a linprogx contract.
 *  - Columns carrying artificial (big-M) bounds are given an *infinite*
 *    range rather than being excluded from the candidate list.  An infinite
 *    range terminates the breakpoint walk at that column, which is the
 *    correct semantics: you may pivot on it, you may not step over it.
 *  - Both sweep loops carry a hard pass cap.  HiGHS's stage-1 loop has no
 *    termination proof when the initial threshold is non-positive (which a
 *    dual-infeasible column can cause); the cap makes the routine total.
 *
 * SIGN CONVENTIONS
 * ----------------
 * linprogx: bound_status is LO/HI/FREE; the leaving row's direction is
 * `leaving_sigma` (+1 below lower, -1 above upper); a column is admissible
 * when (LO and sigma*alpha < 0) or (HI and sigma*alpha > 0).
 * HiGHS: `move_j` is +1 at lower and -1 at upper, `move_out` is the sign of
 * the primal infeasibility, and admissibility is alpha_j*move_out*move_j > 0.
 * The two agree under move_out = -leaving_sigma, which is what this file
 * uses.  The oriented pivot `ap = alpha_j * move_out * move_j` is then
 * positive for every admitted column and equals |alpha_j|.
 *
 * theta_dual is returned in *linprogx's* convention,
 *     theta = -r_q / (sigma * alpha_q),
 * not HiGHS's r_q / alpha_q, so ds2_core can use it directly.  The zero-step
 * rule is convention-independent: theta is set to 0 exactly when the entering
 * column is already dual infeasible (move_q * r_q <= 0), and a zero step
 * invalidates the flip set, which is then discarded.
 */

#include "_ds2_chuzc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* HiGHS's sentinels, by value: an initial total-change floor that keeps the
 * partition from terminating on an exactly-zero accumulation, an initial
 * "smallest rejected ratio" that no real ratio can beat, and a ceiling past
 * which a threshold is treated as unbounded. */
#define DS2_INITIAL_TOTAL_CHANGE 1e-12
#define DS2_INITIAL_REMAIN_THETA 1e100
#define DS2_MAX_SELECT_THETA 1e18

/* Pass caps.  Stage 1 multiplies its threshold by 10 each pass, so 64 passes
 * span every representable double; stage 2 emits one group per pass and a
 * pivot row cannot have more distinct breakpoint groups than candidates. */
#define DS2_STAGE1_MAX_PASS 64

/* 32 bytes: two candidates per cache line, and a swap is one 32-byte move. */
typedef struct {
    double alpha; /* oriented pivot, always > 0 for an admitted column     */
    double tight; /* move_j * r_j -- the breakpoint numerator              */
    double range; /* u_j - l_j, +inf when the column must not flip         */
    int32_t col;
    int32_t move; /* +1 at lower bound, -1 at upper bound                  */
} DS2Cand;

static uint64_t ds2_chuzc_tie_rank(const DS2ChuzcState *st, int32_t col) {
    uint64_t x = (uint64_t)(uint32_t)col +
                 st->tie_seed * UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

DS2ChuzcState *ds2_chuzc_state_new(int32_t n_total) {
    if (n_total < 1) n_total = 1;
    DS2ChuzcState *st = (DS2ChuzcState *)calloc(1, sizeof(DS2ChuzcState));
    if (st == NULL) return NULL;
    st->n_total = n_total;
    st->capacity = n_total;
    st->harris_delta = 1e-7;
    st->expand_tau = 5e-10;
    st->group_cap = n_total + 2;
    st->cand = calloc((size_t)n_total, sizeof(DS2Cand));
    st->group = (int32_t *)calloc((size_t)st->group_cap, sizeof(int32_t));
    st->flip_cols = (int32_t *)calloc((size_t)n_total, sizeof(int32_t));
    if (st->cand == NULL || st->group == NULL || st->flip_cols == NULL) {
        ds2_chuzc_state_free(st);
        return NULL;
    }
    return st;
}

void ds2_chuzc_state_free(DS2ChuzcState *st) {
    if (st == NULL) return;
    free(st->cand);
    free(st->group);
    free(st->flip_cols);
    free(st->range);
    free(st);
}

#ifdef LINPROGX_DS2_IFACE_H
void *ds2_ratio_state_new(int32_t m, int32_t n_total) {
    (void)m;
    DS2ChuzcState *st = ds2_chuzc_state_new(n_total);
    const char *seed = getenv("LINPROGX_DS2_BFRT_SEED");
    if (st != NULL && seed != NULL && *seed != '\0')
        st->tie_seed = strtoull(seed, NULL, 10);
    return st;
}

void ds2_ratio_state_free(void *state) {
    ds2_chuzc_state_free((DS2ChuzcState *)state);
}

void ds2_ratio_prepare(void *state, double delta, int32_t update_count) {
    DS2ChuzcState *st = (DS2ChuzcState *)state;
    if (st == NULL) return;
    st->delta = delta;
    st->update_count = update_count;
}

void ds2_ratio_bounds_changed(void *state, const double *lo_ext,
                              const double *hi_ext) {
    ds2_chuzc_build_range((DS2ChuzcState *)state, lo_ext, hi_ext);
}
#endif

void ds2_chuzc_invalidate_range(DS2ChuzcState *st) {
    if (st != NULL) st->range_valid = 0;
}

void ds2_chuzc_build_range(DS2ChuzcState *st, const double *lo_ext,
                           const double *hi_ext) {
    if (st == NULL) return;
    if (st->range == NULL) {
        st->range = (double *)malloc((size_t)st->n_total * sizeof(double));
        if (st->range == NULL) return;
    }
    const uint8_t *no_flip = st->no_flip;
    for (int32_t j = 0; j < st->n_total; j++) {
        const double lo = lo_ext[j];
        const double hi = hi_ext[j];
        double range = hi - lo;
        if (!isfinite(lo) || !isfinite(hi) || !(range < INFINITY) ||
            (no_flip != NULL && no_flip[j])) {
            range = INFINITY;
        }
        st->range[j] = range;
    }
    st->range_valid = 1;
}

void ds2_chuzc_state_reset_stats(DS2ChuzcState *st) {
    if (st == NULL) return;
    st->n_call = 0;
    st->n_admitted = 0;
    st->n_prefilter = 0;
    st->n_sweep_visits = 0;
    st->n_group = 0;
    st->n_flip_total = 0;
    st->n_no_group = 0;
}

/* Minimum acceptable oriented pivot, ramped with LU age: a pivot that would
 * be fine on a fresh factorisation is not trustworthy after twenty updates. */
static inline double ds2_min_pivot(int32_t update_count) {
    if (update_count < 10) return 1e-9;
    if (update_count < 20) return 3e-8;
    return 1e-6;
}

DS2Entering ds2_chuzc(const double *alpha_row, const int32_t *alpha_pattern,
                      int32_t alpha_nnz, const double *r_ext,
                      const int8_t *bound_status, const double *lo_ext,
                      const double *hi_ext, int leaving_sigma, double dual_tol,
                      void *ratio_state) {
    DS2Entering out;
    out.entering = -1;
    out.theta_dual = 0.0;
    out.alpha_pivot = 0.0;
    out.n_flip = 0;
    out.flip_cols = NULL;

    DS2ChuzcState *st = (DS2ChuzcState *)ratio_state;
    DS2Cand *cand = (DS2Cand *)st->cand;
    out.flip_cols = st->flip_cols;
    st->n_call++;
    st->n_group_cur = 0;

    const double move_out = (leaving_sigma < 0) ? 1.0 : -1.0;
    const double Ta = ds2_min_pivot(st->update_count);
    const double Td = dual_tol;
    const uint8_t *no_flip = st->no_flip;

    /* ---- stage 0: admission -------------------------------------------
     * Also computes work_theta = min_j (tight_j + Td) / alpha_j, the
     * smallest relaxed breakpoint.  The relaxation is applied to the
     * NUMERATOR, as a dual-infeasibility allowance of Td -- so it scales
     * with the reduced costs, unlike a fixed relaxation of the ratio. */
    int32_t cnt = 0;
    double work_theta = DS2_INITIAL_REMAIN_THETA;
    const double *range_tab = st->range_valid ? st->range : NULL;
    for (int32_t i = 0; i < alpha_nnz; i++) {
        const int32_t j = alpha_pattern[i];
        const int8_t bs = bound_status[j];
        double move;
        if (bs == DS2_BOUND_LO) {
            move = 1.0;
        } else if (bs == DS2_BOUND_HI) {
            move = -1.0;
        } else if (bs == DS2_BOUND_FREE) {
            /* A free nonbasic has no natural direction, so give it the one
             * the pivot row asks for; it is then admissible by construction
             * and, having no finite bounds, can never be flipped. */
            move = (alpha_row[j] * move_out > 0.0) ? 1.0 : -1.0;
        } else {
            continue; /* basic or fixed */
        }
        const double a = alpha_row[j];
        const double ap = a * move_out * move;
        if (!(ap > Ta)) continue;

        double range;
        if (range_tab != NULL) {
            range = range_tab[j];
        } else if (bs == DS2_BOUND_FREE || (no_flip != NULL && no_flip[j])) {
            range = INFINITY;
        } else {
            const double lo = lo_ext[j];
            const double hi = hi_ext[j];
            range = hi - lo;
            if (!(range < INFINITY) || !isfinite(lo) || !isfinite(hi)) {
                range = INFINITY;
            }
        }

        DS2Cand *cd = &cand[cnt++];
        cd->alpha = ap;
        cd->tight = r_ext[j] * move;
        cd->range = range;
        cd->col = j;
        cd->move = (int32_t)move;

        const double relax = cd->tight + Td;
        if (work_theta * ap > relax) work_theta = relax / ap;
    }
    st->n_admitted += cnt;
    /* Census: how much of the leaving row's infeasibility could POSSIBLY be
     * absorbed by flips, given the formulation's finite ranges.  When this
     * is small next to delta, no ratio test can flip usefully and the
     * mechanism is inert for reasons that live outside this component. */
    if (st->census) {
        int32_t nfl = 0;
        double absorb = 0.0;
        for (int32_t i = 0; i < cnt; i++) {
            if (isfinite(cand[i].range)) {
                nfl++;
                absorb += cand[i].alpha * cand[i].range;
            }
        }
        st->last_n_cand = cnt;
        st->last_n_flippable = nfl;
        st->last_absorb = absorb;
        st->last_delta = fabs(st->delta);
    }
    if (cnt == 0) return out; /* dual unbounded on this row */

    const double total_delta = fabs(st->delta);

    /* ---- stage 1: geometric pre-filter --------------------------------
     * Partition the candidates against a threshold that starts just above
     * the smallest breakpoint and grows by 10x per pass, accumulating the
     * absorption of everything collected.  Stop as soon as the prefix can
     * absorb the whole infeasibility.  No sort, and on a typical row two or
     * three passes over a shrinking suffix. */
    int32_t full = cnt;
    int32_t take = 0;
    double total_change = 0.0;
    double select_theta = 10.0 * work_theta + 1e-7;
    if (!(select_theta > 0.0)) {
        /* work_theta <= 0 means an already dual-infeasible column is in the
         * row.  Multiplying a negative threshold by 10 diverges instead of
         * converging, so restart from a small positive one. */
        select_theta = 1e-7;
    }
    for (int32_t pass = 0; pass < DS2_STAGE1_MAX_PASS; pass++) {
        st->n_sweep_visits += full - take;
        for (int32_t i = take; i < full; i++) {
            if (cand[i].alpha * select_theta >= cand[i].tight) {
                const DS2Cand tmp = cand[take];
                cand[take] = cand[i];
                cand[i] = tmp;
                total_change += cand[take].range * cand[take].alpha;
                take++;
            }
        }
        select_theta *= 10.0;
        if (total_change >= total_delta || take == full) break;
    }
    if (take == 0) {
        /* Nothing cleared the pre-filter (possible only in the pathological
         * threshold case above).  Fall back to the whole candidate list. */
        take = cnt;
    }
    st->n_prefilter += take;

    /* ---- stage 2: group partition -------------------------------------
     * The breakpoints of the piecewise-linear dual objective, bucketed.
     * Each pass collects everything at or below the current threshold and
     * records the smallest rejected (relaxed) ratio as the next threshold.
     * Groups come out in ascending ratio order; members inside a group are
     * unordered, which is all stage 3 needs. */
    full = take;
    take = 0;
    total_change = DS2_INITIAL_TOTAL_CHANGE;
    select_theta = work_theta;
    int32_t ngroup = 0;
    st->group[ngroup++] = 0;

    int32_t prev_take = -1;
    double prev_select = DS2_INITIAL_REMAIN_THETA;
    while (select_theta < DS2_MAX_SELECT_THETA && ngroup < st->group_cap) {
        double remain = DS2_INITIAL_REMAIN_THETA;
        st->n_sweep_visits += full - take;
        for (int32_t i = take; i < full; i++) {
            const double a = cand[i].alpha;
            const double tight = cand[i].tight;
            if (tight <= select_theta * a) {
                const DS2Cand tmp = cand[take];
                cand[take] = cand[i];
                cand[i] = tmp;
                total_change += cand[take].alpha * cand[take].range;
                take++;
            } else if (tight + Td < remain * a) {
                remain = (tight + Td) / a;
            }
        }
        st->group[ngroup++] = take;
        select_theta = remain;
        /* No-progress guard: an unchanged (count, threshold) pair would
         * repeat forever. */
        if (take == prev_take && select_theta == prev_select) break;
        prev_take = take;
        prev_select = select_theta;
        if (total_change >= total_delta || take == full) break;
    }
    st->n_group += ngroup - 1;
    st->n_group_cur = ngroup;
    st->last_stage1_take = full;
    st->last_total_change = total_change;
    st->last_exhausted = (int8_t)(total_change < total_delta);
    st->last_degenerate = 0;

    /* ---- stage 3: choose the group ------------------------------------
     * Longest step first: scan groups backwards and take the first whose
     * best pivot is at least 10% of the best pivot anywhere in the
     * collected prefix (capped at 1.0, so a row of huge entries does not
     * demand a huge pivot).  This is the load-bearing policy difference
     * from the incumbent, which stops at the first slope-killing
     * breakpoint and therefore always takes the SHORTEST admissible step. */
    double final_compare = 0.0;
    for (int32_t i = 0; i < take; i++) {
        if (cand[i].alpha > final_compare) final_compare = cand[i].alpha;
    }
    final_compare *= 0.1;
    if (final_compare > 1.0) final_compare = 1.0;

    int32_t break_index = -1;
    int32_t break_group = -1;
    for (int32_t g = ngroup - 2; g >= 0; g--) {
        double best = 0.0;
        int32_t bi = -1;
        for (int32_t i = st->group[g]; i < st->group[g + 1]; i++) {
            if (cand[i].alpha > best) {
                best = cand[i].alpha;
                bi = i;
            } else if (bi >= 0 && cand[i].alpha == best &&
                       ((st->tie_seed == 0 && cand[i].col < cand[bi].col) ||
                        (st->tie_seed != 0 &&
                         ds2_chuzc_tie_rank(st, cand[i].col) <
                             ds2_chuzc_tie_rank(st, cand[bi].col)))) {
                bi = i;
            }
        }
        if (bi >= 0 && best > final_compare) {
            break_index = bi;
            break_group = g;
            break;
        }
    }
    if (break_index < 0) {
        /* Every group's best pivot failed the relative-size guard (only
         * reachable when the collected prefix is empty or uniformly tiny).
         * Take the globally largest pivot and do not flip anything. */
        st->n_no_group++;
        double best = 0.0;
        for (int32_t i = 0; i < cnt; i++) {
            if (cand[i].alpha > best) {
                best = cand[i].alpha;
                break_index = i;
            } else if (break_index >= 0 && cand[i].alpha == best &&
                       ((st->tie_seed == 0 &&
                         cand[i].col < cand[break_index].col) ||
                        (st->tie_seed != 0 &&
                         ds2_chuzc_tie_rank(st, cand[i].col) <
                             ds2_chuzc_tie_rank(
                                 st, cand[break_index].col)))) {
                break_index = i;
            }
        }
        if (break_index < 0) return out;
        break_group = -1;
    }

    /* ---- the decision -------------------------------------------------- */
    const int32_t q = cand[break_index].col;
    const double alpha_q = alpha_row[q];
    out.entering = q;
    out.alpha_pivot = alpha_q;

    const double d_q = r_ext[q];
    const double move_q = (double)cand[break_index].move;
    if (d_q * move_q > 0.0) {
        out.theta_dual = -d_q / ((double)leaving_sigma * alpha_q);
    } else {
        /* The entering column is already dual infeasible: the step is zero,
         * which means no breakpoint was actually crossed and the flip set is
         * not legitimate.  Discard it. */
        out.theta_dual = 0.0;
        st->last_degenerate = 1;
        return out;
    }

    /* ---- stage 4: flips ------------------------------------------------
     * Everything in a group strictly before the chosen one has a breakpoint
     * the dual step passes, so it flips to its opposite bound.  ds2_core
     * accumulates sum(delta_j * a_j) over these columns into ONE vector,
     * pays ONE FTRAN, and applies one combined primal update -- the flips
     * are free relative to the pivot they ride along with. */
    int32_t nf = 0;
    if (break_group > 0) {
        const int32_t stop = st->group[break_group];
        for (int32_t i = 0; i < stop; i++) {
            if (!isfinite(cand[i].range)) continue; /* never step over these */
            st->flip_cols[nf++] = cand[i].col;
        }
    }
    out.n_flip = nf;
    st->n_flip_total += nf;
    return out;
}

/* ---------------------------------------------------------------------- */
/* A/B controls: the incumbent Harris two-pass test, reimplemented.        */
/* ---------------------------------------------------------------------- */

/* Shared pass-2 body.  `cand` holds (col, alpha) pairs in ascending column
 * order, which is what makes the |alpha| tie-break resolve to the lowest
 * column index -- the incumbent's behaviour, and load-bearing on +-1
 * matrices where ties are pervasive. */
static DS2Entering ds2_harris_finish(DS2Cand *cand, int32_t cnt,
                                     const double *r_ext, double theta_min,
                                     double harris_delta, int leaving_sigma,
                                     DS2ChuzcState *st) {
    DS2Entering out;
    out.entering = -1;
    out.theta_dual = 0.0;
    out.alpha_pivot = 0.0;
    out.n_flip = 0;
    out.flip_cols = st->flip_cols;
    if (cnt == 0) return out;

    const double theta_max = theta_min + harris_delta;
    double best = 0.0;
    for (int32_t i = 0; i < cnt; i++) {
        const double a = cand[i].alpha; /* raw alpha here, not oriented */
        const double aa = fabs(a);
        const double ratio = fabs(r_ext[cand[i].col]) / aa;
        if (ratio > theta_max) continue;
        if (aa > best) {
            best = aa;
            out.entering = cand[i].col;
            out.alpha_pivot = a;
        }
    }
    if (out.entering < 0) return out;
    out.theta_dual =
        -r_ext[out.entering] / ((double)leaving_sigma * out.alpha_pivot);
    return out;
}

static inline int ds2_harris_admissible(int8_t bs, int leaving_sigma,
                                        double alpha_j) {
    if (bs == DS2_BOUND_LO) return (double)leaving_sigma * alpha_j < 0.0;
    if (bs == DS2_BOUND_HI) return (double)leaving_sigma * alpha_j > 0.0;
    return bs == DS2_BOUND_FREE;
}

DS2Entering ds2_chuzc_harris_dense(const double *alpha_row,
                                   const int32_t *alpha_pattern,
                                   int32_t alpha_nnz, const double *r_ext,
                                   const int8_t *bound_status,
                                   const double *lo_ext, const double *hi_ext,
                                   int leaving_sigma, double dual_tol,
                                   void *ratio_state) {
    (void)alpha_pattern;
    (void)alpha_nnz;
    (void)lo_ext;
    (void)hi_ext;
    (void)dual_tol;
    DS2ChuzcState *st = (DS2ChuzcState *)ratio_state;
    DS2Cand *cand = (DS2Cand *)st->cand;
    st->n_call++;

    double theta_min = 1e300;
    int32_t cnt = 0;
    for (int32_t j = 0; j < st->n_total; j++) {
        const int8_t bs = bound_status[j];
        if (bs == DS2_BOUND_BASIC || bs == DS2_BOUND_FIXED) continue;
        const double a = alpha_row[j];
        if (fabs(a) < 1e-9) continue;
        if (!ds2_harris_admissible(bs, leaving_sigma, a)) continue;
        const double ratio = (fabs(r_ext[j]) + st->expand_tau) / fabs(a);
        if (ratio < theta_min) theta_min = ratio;
        cand[cnt].col = j;
        cand[cnt].alpha = a;
        cnt++;
    }
    st->n_admitted += cnt;
    return ds2_harris_finish(cand, cnt, r_ext, theta_min, st->harris_delta,
                             leaving_sigma, st);
}

DS2Entering ds2_chuzc_harris_pattern(const double *alpha_row,
                                     const int32_t *alpha_pattern,
                                     int32_t alpha_nnz, const double *r_ext,
                                     const int8_t *bound_status,
                                     const double *lo_ext, const double *hi_ext,
                                     int leaving_sigma, double dual_tol,
                                     void *ratio_state) {
    (void)lo_ext;
    (void)hi_ext;
    (void)dual_tol;
    DS2ChuzcState *st = (DS2ChuzcState *)ratio_state;
    DS2Cand *cand = (DS2Cand *)st->cand;
    st->n_call++;

    double theta_min = 1e300;
    int32_t cnt = 0;
    for (int32_t i = 0; i < alpha_nnz; i++) {
        const int32_t j = alpha_pattern[i];
        const int8_t bs = bound_status[j];
        if (bs == DS2_BOUND_BASIC || bs == DS2_BOUND_FIXED) continue;
        const double a = alpha_row[j];
        if (fabs(a) < 1e-9) continue;
        if (!ds2_harris_admissible(bs, leaving_sigma, a)) continue;
        const double ratio = (fabs(r_ext[j]) + st->expand_tau) / fabs(a);
        if (ratio < theta_min) theta_min = ratio;
        cand[cnt].col = j;
        cand[cnt].alpha = a;
        cnt++;
    }
    st->n_admitted += cnt;
    return ds2_harris_finish(cand, cnt, r_ext, theta_min, st->harris_delta,
                             leaving_sigma, st);
}

#ifdef LINPROGX_DS2_IFACE_H
static DS2Entering ds2_chuzc_core_harris(
    const double *alpha_row,
    const int32_t *alpha_pattern, int32_t alpha_nnz,
    const double *r_ext, const int8_t *bound_status,
    int leaving_sigma, void *ratio_state) {
    DS2ChuzcState *st = (DS2ChuzcState *)ratio_state;
    DS2Entering out = {-1, 0.0, 0.0, 0, st->flip_cols};
    const double sigma = (double)leaving_sigma;
    const double tau = st->expand_tau;
    const double dtau = 5e-11;
    st->expand_tau += dtau;
    if (st->expand_tau > 1e-8) st->expand_tau = 5e-10;
    st->n_call++;

    double theta_max = HUGE_VAL;
    for (int32_t t = 0; t < alpha_nnz; t++) {
        const int32_t j = alpha_pattern[t];
        const int8_t bs = bound_status[j];
        if (bs == DS2_BOUND_BASIC || bs == DS2_BOUND_FIXED) continue;
        const double a = sigma * alpha_row[j];
        if (bs == DS2_BOUND_LO) {
            if (a > -1e-9) continue;
        } else if (bs == DS2_BOUND_HI) {
            if (a < 1e-9) continue;
        } else if (fabs(a) < 1e-9) {
            continue;
        }
        const double bound = (fabs(r_ext[j]) + tau) / fabs(a);
        if (bound < theta_max) theta_max = bound;
    }
    if (theta_max == HUGE_VAL) return out;

    double best_alpha = 0.0;
    double best_ratio = 0.0;
    for (int32_t t = 0; t < alpha_nnz; t++) {
        const int32_t j = alpha_pattern[t];
        const int8_t bs = bound_status[j];
        if (bs == DS2_BOUND_BASIC || bs == DS2_BOUND_FIXED) continue;
        const double a = sigma * alpha_row[j];
        if (bs == DS2_BOUND_LO) {
            if (a > -1e-9) continue;
        } else if (bs == DS2_BOUND_HI) {
            if (a < 1e-9) continue;
        } else if (fabs(a) < 1e-9) {
            continue;
        }
        double ratio = -r_ext[j] / a;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > theta_max) continue;
        const double abs_a = fabs(a);
        if (abs_a > best_alpha ||
            (abs_a == best_alpha && out.entering >= 0 && j < out.entering)) {
            best_alpha = abs_a;
            best_ratio = ratio;
            out.entering = j;
        }
    }
    if (out.entering < 0) return out;

    const double floor_step = dtau / best_alpha;
    if (best_ratio < floor_step) best_ratio = floor_step;
    out.theta_dual = -sigma * best_ratio;
    out.alpha_pivot = alpha_row[out.entering];
    return out;
}

DS2Entering ds2_chuzc_core(
    const double *alpha_row,
    const int32_t *alpha_pattern, int32_t alpha_nnz,
    const double *r_ext, const int8_t *bound_status,
    const double *lo_ext, const double *hi_ext,
    int leaving_sigma, double dual_tol,
    void *ratio_state) {
    const char *gate = getenv("LINPROGX_DS2_BFRT");
    if (gate == NULL || atoi(gate) != 0) {
        DS2Entering out =
            ds2_chuzc(alpha_row, alpha_pattern, alpha_nnz, r_ext,
                      bound_status, lo_ext, hi_ext, leaving_sigma, dual_tol,
                      ratio_state);
        /*
         * Component A reports the nonnegative oriented breakpoint step
         * -r/(sigma*alpha).  The core contract carries the signed reduced-cost
         * update r/alpha, so convert at this integration boundary.
         */
        out.theta_dual *= -(double)leaving_sigma;
        return out;
    }
    (void)lo_ext;
    (void)hi_ext;
    (void)dual_tol;
    return ds2_chuzc_core_harris(alpha_row, alpha_pattern, alpha_nnz, r_ext,
                                 bound_status, leaving_sigma, ratio_state);
}
#endif
