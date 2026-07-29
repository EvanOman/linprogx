/* DS2 CHUZR -- dual simplex leaving-row selection with dual steepest edge.
 *
 * PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * Written after reading HiGHS's HEkkDualRHS.cpp (chooseNormal :37-121,
 * createInfeasList :432-513, updateInfeasList :374-410, updatePrimal
 * :310-350, updatePivots :352-372) under the 2026-07-25 owner authorisation
 * in docs/PROVENANCE.md. No code was copied. What was taken is the shape of
 * the data structure:
 *
 *   - every row's primal infeasibility is MAINTAINED in a dense array, not
 *     recomputed from x_B and the bounds every pivot;
 *   - it is updated only over the FTRAN pattern of the entering column, which
 *     is the only place x_B can have moved;
 *   - the infeasible rows are additionally kept in a candidate list, so CHUZR
 *     touches |infeasible| entries rather than m;
 *   - the list is rebuilt on demand and abandoned for a plain scan when too
 *     large a fraction of rows are infeasible, because a short list is the
 *     only thing that makes the indirection pay;
 *   - a merit cutoff can shrink a very long list further, but only when the
 *     pivot column is sparse enough that few rows come back before the next
 *     rebuild;
 *   - the scan may start at a random offset and wrap, so merit ties are not
 *     systematically resolved to the lowest row index.
 *
 * The shipped linprogx CHUZR instead rescans all m rows every pivot, and for
 * each gathers lo_ext[basis[k]] / hi_ext[basis[k]] before it can even tell
 * whether the row is infeasible (src/linprogx/_csparse.c:14566-14606, AVX2
 * variant :12647-12746).
 *
 * Self-contained C99. No Python.h, no linprogx headers, no dependency on the
 * shipped solver.
 */

#include "_ds2_chuzr.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* A merit cutoff is only built for lists longer than this fraction of m, and
 * only when the pivot column is sparser than DS2_CUTOFF_MAX_DENSITY.
 *
 * DELIBERATE DEVIATION. HiGHS's thresholds here are absolute: build a cutoff
 * above 500 candidates, keep max(0.001*count, 500) of them, and give up on the
 * list above 0.2*m. At m ~ 2400 -- the whole simplex-routed class in this
 * campaign -- those cannot all hold at once: 500 candidates is already above
 * 0.2*2392 = 478, so any list long enough to earn a cutoff is immediately
 * abandoned for a dense scan. The cutoff branch is structurally unreachable at
 * our sizes. Expressed as fractions of m the same mechanism is reachable, so
 * it can be measured rather than assumed. These are global constants; nothing
 * here is fitted to an instance. */
#define DS2_CUTOFF_MIN_FRAC    0.05
#define DS2_CUTOFF_MAX_DENSITY 0.05
#define DS2_CUTOFF_KEEP_FRAC   0.02
#define DS2_CUTOFF_KEEP_MIN    32
/* Above this fraction of infeasible rows the list costs more than it saves. */
#define DS2_DENSE_FRAC         0.2

struct DS2Pricing {
    int32_t  m;
    int      rule;
    double   weight_floor;
    int      random_start;
    int      paranoid;
    int      list_mode;
    int      cutoff_enabled;
    double   column_density;
    double   feas_tol;      /* tolerance the maintained violations were made
                             * with; a change forces a recompute            */
    int      have_tol;

    double  *viol;          /* maintained violation per basis position      */
    int8_t  *sig;           /* +1 below lower, -1 above upper, 0 feasible   */
    int32_t  n_infeas;

    int8_t  *mark;          /* mark[k] != 0  <=>  k is in the candidate list */
    int32_t *list;
    int32_t  count;         /* candidate-list length                        */
    int      dense_mode;    /* 1 => scan the maintained array, not the list */
    double   cutoff;

    int8_t  *ban;
    int32_t  n_ban;
    int      valid;         /* 0 => recompute everything on next ds2_chuzr  */

    double  *scratch;       /* merits, for the cutoff selection             */

    /* Core integration: exact Forrest-Goldfarb dual steepest-edge state. */
    int      core_direct;
    int      have_weights;
    int      edge_update;
    int      edge_period;
    int64_t  edge_count;
    double  *w_keep;
    double  *tau;
    int32_t *pat;
    int32_t  n_total;
    int32_t *enter_count;
    int      churn_on;
    double   churn_alpha;
    int32_t  churn_cap;
    int32_t  churn_dead;

    uint64_t rng;
    DS2ChuzrStats stats;
};

/* xorshift64*: deterministic, seedable, and cheap enough that a random scan
 * start is free relative to the scan itself. */
static uint64_t ds2_rand(DS2Pricing *st)
{
    uint64_t x = st->rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    st->rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static int32_t ds2_rand_below(DS2Pricing *st, int32_t n)
{
    if (n <= 1) return 0;
    return (int32_t)(ds2_rand(st) % (uint64_t)n);
}

/* Violation of basis position k. Identical arithmetic to the shipped scan
 * (_csparse.c:14570-14586) so selection can be compared row for row. */
static inline double ds2_violation_at(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    int32_t k, double feas_tol, int *sigma_out)
{
    int32_t j = basis[k];
    double viol = 0.0;
    int sigma = 0;
    double lo = lo_ext[j];
    double hi = hi_ext[j];
    if (isfinite(lo) && x_B[k] < lo - feas_tol) {
        viol = lo - x_B[k];
        sigma = 1;
    }
    if (isfinite(hi) && x_B[k] > hi + feas_tol) {
        double v2 = x_B[k] - hi;
        if (v2 > viol) {
            viol = v2;
            sigma = -1;
        }
    }
    *sigma_out = sigma;
    return viol;
}

/* One merit for both rules: for viol >= 0, argmax viol == argmax viol^2, so
 * Dantzig is DSE with every weight equal to 1. */
static inline double ds2_merit_of(const DS2Pricing *st, double viol,
                                  const double *weights, int32_t k)
{
    if (st->rule == DS2_RULE_DANTZIG || weights == NULL) return viol * viol;
    double w = weights[k];
    if (w < st->weight_floor) w = st->weight_floor;
    return (viol * viol) / w;
}

/* ---- construction ------------------------------------------------------ */

DS2Pricing *ds2_pricing_new(int32_t m, int rule)
{
    if (m <= 0) return NULL;
    DS2Pricing *st = (DS2Pricing *)calloc(1, sizeof(DS2Pricing));
    if (st == NULL) return NULL;
    st->m = m;
    st->rule = rule;
    st->weight_floor = 1e-4;     /* HiGHS kMinDualSteepestEdgeWeight        */
    st->random_start = 0;
    st->list_mode = DS2_LIST_ON;
    st->cutoff_enabled = 1;
    st->column_density = 1.0;
    st->rng = 0x9E3779B97F4A7C15ULL;
    st->viol = (double *)calloc((size_t)m, sizeof(double));
    st->sig = (int8_t *)calloc((size_t)m, sizeof(int8_t));
    st->mark = (int8_t *)calloc((size_t)m, sizeof(int8_t));
    st->ban = (int8_t *)calloc((size_t)m, sizeof(int8_t));
    st->list = (int32_t *)malloc((size_t)m * sizeof(int32_t));
    st->scratch = (double *)malloc((size_t)m * sizeof(double));
    if (st->viol == NULL || st->sig == NULL || st->mark == NULL ||
        st->ban == NULL || st->list == NULL || st->scratch == NULL) {
        ds2_pricing_free(st);
        return NULL;
    }
    return st;
}

void ds2_pricing_free(DS2Pricing *st)
{
    if (st == NULL) return;
    free(st->viol);
    free(st->sig);
    free(st->mark);
    free(st->ban);
    free(st->list);
    free(st->scratch);
    free(st->w_keep);
    free(st->tau);
    free(st->pat);
    free(st->enter_count);
    free(st);
}

void ds2_pricing_set_weight_floor(DS2Pricing *st, double floor)
{
    if (st != NULL && floor > 0.0) st->weight_floor = floor;
}

void ds2_pricing_set_random_start(DS2Pricing *st, int on)
{
    if (st != NULL) st->random_start = on ? 1 : 0;
}

void ds2_pricing_set_seed(DS2Pricing *st, uint64_t seed)
{
    if (st != NULL) st->rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

void ds2_pricing_set_rule(DS2Pricing *st, int rule)
{
    if (st != NULL && st->rule != rule) {
        st->rule = rule;
        st->valid = 0;   /* the cutoff and the list depend on the merit */
    }
}

void ds2_pricing_set_column_density(DS2Pricing *st, double density)
{
    if (st != NULL) st->column_density = density;
}

void ds2_pricing_set_list_mode(DS2Pricing *st, int mode)
{
    if (st != NULL && st->list_mode != mode) {
        st->list_mode = mode;
        st->valid = 0;
    }
}

void ds2_pricing_set_cutoff_enabled(DS2Pricing *st, int on)
{
    if (st != NULL && st->cutoff_enabled != (on ? 1 : 0)) {
        st->cutoff_enabled = on ? 1 : 0;
        st->valid = 0;
    }
}

void ds2_pricing_set_paranoid(DS2Pricing *st, int on)
{
    if (st != NULL) st->paranoid = on ? 1 : 0;
}

const DS2ChuzrStats *ds2_chuzr_stats(const DS2Pricing *st)
{
    return (st == NULL) ? NULL : &st->stats;
}

void ds2_chuzr_stats_reset(DS2Pricing *st)
{
    if (st != NULL) memset(&st->stats, 0, sizeof(st->stats));
}

int32_t ds2_chuzr_list_len(const DS2Pricing *st)
{
    if (st == NULL) return 0;
    return st->dense_mode ? -st->m : st->count;
}

int32_t ds2_chuzr_num_infeasible(const DS2Pricing *st)
{
    return (st == NULL) ? 0 : st->n_infeas;
}

void ds2_chuzr_invalidate(DS2Pricing *st)
{
    if (st != NULL) st->valid = 0;
}

void ds2_chuzr_ban(DS2Pricing *st, int32_t basis_pos)
{
    if (st == NULL || basis_pos < 0 || basis_pos >= st->m) return;
    if (!st->ban[basis_pos]) {
        st->ban[basis_pos] = 1;
        st->n_ban++;
    }
}

void ds2_chuzr_clear_bans(DS2Pricing *st)
{
    if (st == NULL || st->n_ban == 0) return;
    memset(st->ban, 0, (size_t)st->m * sizeof(int8_t));
    st->n_ban = 0;
}

/* ---- incremental maintenance ------------------------------------------ */

void ds2_chuzr_rows_changed(
    DS2Pricing *st,
    const int32_t *rows, int32_t n_rows,
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext, double feas_tol)
{
    if (st == NULL || !st->valid) return;
    if (st->have_tol && feas_tol != st->feas_tol) {
        st->valid = 0;   /* the maintained violations are for another tol */
        return;
    }
    st->stats.changed_rows += n_rows;
    for (int32_t i = 0; i < n_rows; i++) {
        int32_t k = rows[i];
        if (k < 0 || k >= st->m) continue;
        int sigma;
        double v =
            ds2_violation_at(basis, x_B, lo_ext, hi_ext, k, feas_tol, &sigma);
        int was = st->viol[k] > 0.0;
        int now = v > 0.0;
        st->viol[k] = v;
        st->sig[k] = (int8_t)sigma;
        if (was != now) st->n_infeas += now ? 1 : -1;
        if (now && !st->dense_mode && !st->mark[k]) {
            st->mark[k] = 1;
            st->list[st->count++] = k;
        }
    }
}

/* Quickselect: leaves the (k+1)-th largest merit at position k. Rebuild path
 * only, and only for lists long enough to be worth cutting. */
static double ds2_select_kth_largest(double *a, int32_t n, int32_t k)
{
    int32_t lo = 0, hi = n - 1;
    while (lo < hi) {
        double pivot = a[lo + (hi - lo) / 2];
        int32_t i = lo, j = hi;
        while (i <= j) {
            while (a[i] > pivot) i++;
            while (a[j] < pivot) j--;
            if (i <= j) {
                double t = a[i]; a[i] = a[j]; a[j] = t;
                i++; j--;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return a[k];
}

/* Recompute every row's violation from the caller's arrays. */
static void ds2_recompute_all(
    DS2Pricing *st,
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext, double feas_tol)
{
    st->stats.recomputes++;
    st->n_infeas = 0;
    for (int32_t k = 0; k < st->m; k++) {
        int sigma;
        double v =
            ds2_violation_at(basis, x_B, lo_ext, hi_ext, k, feas_tol, &sigma);
        st->viol[k] = v;
        st->sig[k] = (int8_t)sigma;
        if (v > 0.0) st->n_infeas++;
    }
    st->feas_tol = feas_tol;
    st->have_tol = 1;
}

/* (Re)build the candidate list from the maintained violations. */
static void ds2_build_list(DS2Pricing *st, const double *weights,
                           int allow_cutoff)
{
    const int32_t m = st->m;
    st->stats.rebuilds++;
    st->stats.infeas_sum += st->n_infeas;
    st->cutoff = 0.0;
    st->dense_mode = 0;
    st->count = 0;

    if (st->list_mode == DS2_LIST_OFF) {
        st->dense_mode = 1;
        st->valid = 1;
        return;
    }

    memset(st->mark, 0, (size_t)m * sizeof(int8_t));
    double max_merit = 0.0;
    for (int32_t k = 0; k < m; k++) {
        if (st->viol[k] <= 0.0) continue;
        double merit = ds2_merit_of(st, st->viol[k], weights, k);
        if (merit > max_merit) max_merit = merit;
        st->scratch[st->count] = merit;
        st->mark[k] = 1;
        st->list[st->count++] = k;
    }

    const double min_for_cutoff = (double)m * DS2_CUTOFF_MIN_FRAC;
    if (allow_cutoff && st->cutoff_enabled &&
        (double)st->count > min_for_cutoff &&
        st->column_density < DS2_CUTOFF_MAX_DENSITY) {
        int32_t keep = (int32_t)((double)m * DS2_CUTOFF_KEEP_FRAC);
        if (keep < DS2_CUTOFF_KEEP_MIN) keep = DS2_CUTOFF_KEEP_MIN;
        if (keep < st->count) {
            double cut = ds2_select_kth_largest(st->scratch, st->count, keep);
            st->cutoff = max_merit * 0.99999;
            if (cut * 1.00001 < st->cutoff) st->cutoff = cut * 1.00001;
            st->stats.cutoff_installed++;
            int32_t put = 0;
            for (int32_t i = 0; i < st->count; i++) {
                int32_t k = st->list[i];
                if (ds2_merit_of(st, st->viol[k], weights, k) >= st->cutoff) {
                    st->list[put++] = k;
                } else {
                    st->mark[k] = 0;
                }
            }
            st->count = put;
        }
    }

    if (st->list_mode != DS2_LIST_ALWAYS &&
        (double)st->count > DS2_DENSE_FRAC * (double)m) {
        st->dense_mode = 1;
        st->cutoff = 0.0;
        st->count = 0;
        memset(st->mark, 0, (size_t)m * sizeof(int8_t));
    }
    st->valid = 1;
}

/* ---- the scan ---------------------------------------------------------- */

/* Scan the maintained array (dense mode) or the candidate list. Feasible
 * entries are tombstoned out of the list as they are met; they come back
 * through ds2_chuzr_rows_changed if they go infeasible again. */
static DS2Leaving ds2_choose(DS2Pricing *st, double *weights,
                             double *best_merit_out)
{
    DS2Leaving out = {-1, 0, 0.0};
    double best = 0.0;
    const int use_w = (weights != NULL && st->rule != DS2_RULE_DANTZIG);
    const double wfloor = st->weight_floor;

    if (st->dense_mode) {
        const int32_t m = st->m;
        st->stats.dense_calls++;
        st->stats.dense_scanned += m;
        int32_t start = st->random_start ? ds2_rand_below(st, m) : 0;
        for (int32_t pass = 0; pass < 2; pass++) {
            int32_t from = (pass == 0) ? start : 0;
            int32_t to = (pass == 0) ? m : start;
            for (int32_t k = from; k < to; k++) {
                double v = st->viol[k];
                if (v <= 0.0 || st->ban[k]) continue;
                double merit = v * v;
                if (use_w) {
                    /* The weight floor is CHUZR's one in-place obligation,
                     * and it is paid only on rows actually examined. */
                    double w = weights[k];
                    if (w < wfloor) { w = wfloor; weights[k] = w; }
                    merit /= w;
                }
                if (merit > best) {
                    best = merit;
                    out.basis_pos = k;
                    out.sigma = st->sig[k];
                    out.violation = v;
                }
            }
        }
        /* Recover to list mode once well under the threshold; the hysteresis
         * keeps a solve hovering at the boundary from rebuilding every
         * pivot. */
        if (st->list_mode != DS2_LIST_OFF &&
            (double)st->n_infeas < 0.5 * DS2_DENSE_FRAC * (double)m)
            st->valid = 0;
        *best_merit_out = best;
        return out;
    }

    const int32_t n = st->count;
    st->stats.list_len_sum += n;
    st->stats.scanned += n;
    int32_t start = st->random_start ? ds2_rand_below(st, n) : 0;
    int32_t dead = 0;
    for (int32_t pass = 0; pass < 2; pass++) {
        int32_t from = (pass == 0) ? start : 0;
        int32_t to = (pass == 0) ? n : start;
        for (int32_t i = from; i < to; i++) {
            int32_t k = st->list[i];
            if (k < 0) continue;
            double v = st->viol[k];
            if (v <= 0.0) {
                if (!st->ban[k]) {
                    st->mark[k] = 0;
                    st->list[i] = -1;   /* compacted after the two passes */
                    dead++;
                }
                continue;
            }
            if (st->ban[k]) continue;
            double merit = v * v;
            if (use_w) {
                double w = weights[k];
                if (w < wfloor) { w = wfloor; weights[k] = w; }
                merit /= w;
            }
            if (merit > best) {
                best = merit;
                out.basis_pos = k;
                out.sigma = st->sig[k];
                out.violation = v;
            }
        }
    }
    if (dead > 0) {
        int32_t put = 0;
        for (int32_t i = 0; i < st->count; i++)
            if (st->list[i] >= 0) st->list[put++] = st->list[i];
        st->count = put;
    }
    *best_merit_out = best;
    return out;
}

/* ---- entry points ------------------------------------------------------ */

DS2Leaving ds2_chuzr_dense_reference(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    const double *weights, int32_t m, double feas_tol,
    int rule, double weight_floor)
{
    DS2Leaving out = {-1, 0, 0.0};
    double best = 0.0;
    for (int32_t k = 0; k < m; k++) {
        if (weights != NULL && weights[k] >= DS2_WEIGHT_BANNED) continue;
        int32_t j = basis[k];
        double viol = 0.0;
        int sigma = 0;
        if (isfinite(lo_ext[j]) && x_B[k] < lo_ext[j] - feas_tol) {
            viol = lo_ext[j] - x_B[k];
            sigma = 1;
        }
        if (isfinite(hi_ext[j]) && x_B[k] > hi_ext[j] + feas_tol) {
            double v2 = x_B[k] - hi_ext[j];
            if (v2 > viol) { viol = v2; sigma = -1; }
        }
        if (viol <= 0.0) continue;
        double merit;
        if (rule == DS2_RULE_DANTZIG || weights == NULL) {
            merit = viol * viol;
        } else {
            double w = weights[k];
            if (w < weight_floor) w = weight_floor;
            merit = (viol * viol) / w;
        }
        if (merit > best) {
            best = merit;
            out.basis_pos = k;
            out.sigma = sigma;
            out.violation = viol;
        }
    }
    return out;
}

DS2Leaving ds2_chuzr(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    double *weights,
    int32_t m, double feas_tol,
    void *pricing_state)
{
    DS2Pricing *st = (DS2Pricing *)pricing_state;
    if (st == NULL) {
        return ds2_chuzr_dense_reference(basis, x_B, lo_ext, hi_ext, weights,
                                         m, feas_tol, DS2_RULE_DSE, 1e-4);
    }
    DS2Leaving out = {-1, 0, 0.0};
    if (st->m != m) return out;

    if (st->core_direct) {
        st->stats.calls++;
        st->stats.dense_calls++;
        st->stats.dense_scanned += m;
        DS2Leaving direct = {-1, 0, 0.0};
        double best = 0.0;
        for (int32_t k = 0; k < m; k++) {
            if (weights != NULL &&
                weights[k] >= DS2_WEIGHT_BANNED) continue;
            int sigma;
            const double violation = ds2_violation_at(
                basis, x_B, lo_ext, hi_ext, k, feas_tol, &sigma);
            if (violation <= 0.0) continue;
            double merit = violation * violation;
            if (st->rule != DS2_RULE_DANTZIG && weights != NULL) {
                double weight = weights[k];
                if (weight < st->weight_floor)
                    weight = st->weight_floor;
                merit /= weight;
            }
            if (st->churn_on) {
                const int32_t col = basis[k];
                int32_t count =
                    (col >= 0 && col < st->n_total)
                        ? st->enter_count[col] : 0;
                if (count > st->churn_cap) count = st->churn_cap;
                if (count > st->churn_dead) {
                    merit /=
                        1.0 + st->churn_alpha *
                                  (double)(count - st->churn_dead);
                }
            }
            if (merit > best) {
                best = merit;
                direct.basis_pos = k;
                direct.sigma = sigma;
                direct.violation = violation;
            }
        }
        return direct;
    }

    st->stats.calls++;
    if (st->have_tol && feas_tol != st->feas_tol) st->valid = 0;
    if (!st->valid) {
        ds2_recompute_all(st, basis, x_B, lo_ext, hi_ext, feas_tol);
        ds2_build_list(st, weights, 1);
    }

    double best_merit = 0.0;
    out = ds2_choose(st, weights, &best_merit);

    /* A cutoff list hides every row below the cutoff. If nothing was found,
     * or the best is no better than the cutoff itself, the answer is not
     * trustworthy: rebuild without a cutoff and scan again. */
    if (!st->dense_mode && st->cutoff > 0.0 &&
        (out.basis_pos < 0 || best_merit <= st->cutoff * 0.99)) {
        st->stats.cutoff_misses++;
        ds2_build_list(st, weights, 0);
        out = ds2_choose(st, weights, &best_merit);
    }

    if (st->paranoid) {
        DS2Leaving ref = ds2_chuzr_dense_reference(
            basis, x_B, lo_ext, hi_ext, weights, m, feas_tol, st->rule,
            st->weight_floor);
        int bad = 0;
        if ((ref.basis_pos < 0) != (out.basis_pos < 0)) {
            bad = 1;
        } else if (ref.basis_pos >= 0 && st->n_ban == 0) {
            /* With a random start ties may resolve differently, so only a
             * strictly worse MERIT counts as a disagreement. */
            double mr = ds2_merit_of(st, ref.violation, weights,
                                     ref.basis_pos);
            double mo = ds2_merit_of(st, out.violation, weights,
                                     out.basis_pos);
            if (mo < mr * (1.0 - 1e-12)) bad = 1;
        }
        if (bad) st->stats.paranoid_mismatch++;
    }

    return out;
}

int32_t ds2_chuzr_audit(
    const DS2Pricing *st,
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    double feas_tol, double tol, int32_t *first_bad_row)
{
    if (st == NULL) return 0;
    int32_t bad = 0;
    if (first_bad_row != NULL) *first_bad_row = -1;
    for (int32_t k = 0; k < st->m; k++) {
        int sigma;
        double v =
            ds2_violation_at(basis, x_B, lo_ext, hi_ext, k, feas_tol, &sigma);
        if (fabs(v - st->viol[k]) > tol ||
            (v > 0.0 && sigma != (int)st->sig[k])) {
            if (bad == 0 && first_bad_row != NULL) *first_bad_row = k;
            bad++;
        } else if (v > 0.0 && !st->dense_mode && !st->mark[k] &&
                   st->cutoff <= 0.0) {
            /* Infeasible but missing from an uncut list: the superset
             * invariant CHUZR relies on is broken. */
            if (bad == 0 && first_bad_row != NULL) *first_bad_row = k;
            bad++;
        }
    }
    return bad;
}

#ifdef LINPROGX_DS2_IFACE_H
static int ds2_core_env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    if (value == NULL || *value == '\0') return fallback;
    return atoi(value);
}

static double ds2_core_env_double(const char *name, double fallback) {
    const char *value = getenv(name);
    if (value == NULL || *value == '\0') return fallback;
    return atof(value);
}

void *ds2_pricing_state_new(int32_t m, int32_t n_total) {
    int rule = ds2_core_env_int("LINPROGX_DS2_RULE", DS2_RULE_DANTZIG);
    if (rule != DS2_RULE_DSE) rule = DS2_RULE_DANTZIG;
    DS2Pricing *st = ds2_pricing_new(m, rule);
    if (st == NULL) return NULL;
    st->core_direct = 1;
    st->edge_update =
        ds2_core_env_int("LINPROGX_DS2_EDGE_UPDATE", 0);
    st->edge_period =
        ds2_core_env_int("LINPROGX_DS2_EDGE_PERIOD", 1);
    if (st->edge_period < 1) st->edge_period = 1;
    st->weight_floor = 1e-12;
    st->w_keep = (double *)calloc((size_t)m, sizeof(double));
    st->tau = (double *)calloc((size_t)m, sizeof(double));
    st->pat = (int32_t *)calloc((size_t)m, sizeof(int32_t));
    st->n_total = n_total;
    st->churn_on =
        ds2_core_env_int("LINPROGX_DS2_CHURN", 0) != 0;
    st->churn_alpha =
        ds2_core_env_double("LINPROGX_DS_CHURN_ALPHA", 1.0);
    st->churn_cap = (int32_t)ds2_core_env_int(
        "LINPROGX_DS_CHURN_CAP", INT32_MAX);
    st->churn_dead = (int32_t)ds2_core_env_int(
        "LINPROGX_DS_CHURN_DEADBAND", 0);
    if (st->churn_on) {
        st->enter_count = (int32_t *)calloc(
            (size_t)(n_total > 0 ? n_total : 1), sizeof(int32_t));
    }
    if (st->w_keep == NULL || st->tau == NULL || st->pat == NULL) {
        ds2_pricing_free(st);
        return NULL;
    }
    if (st->churn_on && st->enter_count == NULL) {
        ds2_pricing_free(st);
        return NULL;
    }
    return st;
}

void ds2_pricing_state_free(void *state) {
    ds2_pricing_free((DS2Pricing *)state);
}

void ds2_pricing_reset(void *state, double *weights, int32_t m,
                       int logical_basis, const DS2LinAlg *la) {
    DS2Pricing *st = (DS2Pricing *)state;
    if (st == NULL || st->m != m || weights == NULL) return;
    ds2_chuzr_invalidate(st);
    ds2_chuzr_clear_bans(st);
    if (st->rule == DS2_RULE_DANTZIG) return;

    if (!st->have_weights) {
        if (logical_basis || la == NULL || la->btran_unit == NULL) {
            for (int32_t k = 0; k < m; k++) st->w_keep[k] = 1.0;
        } else {
            memset(st->tau, 0, (size_t)m * sizeof(double));
            for (int32_t k = 0; k < m; k++) {
                const int32_t nnz =
                    la->btran_unit(la->ctx, k, st->tau, st->pat);
                double gamma = 0.0;
                for (int32_t i = 0; i < nnz; i++) {
                    const int32_t row = st->pat[i];
                    const double value = st->tau[row];
                    gamma += value * value;
                    st->tau[row] = 0.0;
                }
                st->w_keep[k] = gamma > 1e-12 ? gamma : 1e-12;
            }
        }
        st->have_weights = 1;
    }
    memcpy(weights, st->w_keep, (size_t)m * sizeof(double));
}

void ds2_pricing_update(
    void *state,
    int32_t leaving_pos, int32_t entering,
    const double *rho, const int32_t *rho_pattern, int32_t rho_nnz,
    const double *alpha_col, const int32_t *ftran_pattern, int32_t ftran_nnz,
    double alpha_pivot, double *weights, int32_t m,
    const DS2LinAlg *la) {
    (void)entering;
    DS2Pricing *st = (DS2Pricing *)state;
    if (st != NULL && st->enter_count != NULL &&
        entering >= 0 && entering < st->n_total) {
        st->enter_count[entering]++;
    }
    if (st == NULL || st->m != m || st->rule == DS2_RULE_DANTZIG ||
        weights == NULL || alpha_pivot == 0.0) {
        return;
    }

    const double inv_pivot = 1.0 / alpha_pivot;
    st->edge_count++;
    if (st->edge_update == 1 ||
        (st->edge_period > 1 &&
         st->edge_count % st->edge_period != 0)) {
        for (int32_t i = 0; i < ftran_nnz; i++) {
            const int32_t k = ftran_pattern[i];
            if (k == leaving_pos) continue;
            const double ratio = alpha_col[k] * inv_pivot;
            const double gamma = ratio * ratio;
            if (gamma > st->w_keep[k]) {
                st->w_keep[k] = gamma;
                weights[k] = gamma;
            }
        }
        double gamma_r = inv_pivot * inv_pivot;
        if (gamma_r < 1e-12) gamma_r = 1e-12;
        st->w_keep[leaving_pos] = gamma_r;
        weights[leaving_pos] = gamma_r;
        return;
    }
    if (la == NULL || la->ftran == NULL) return;

    double gamma_r = 0.0;
    for (int32_t i = 0; i < rho_nnz; i++) {
        const double value = rho[rho_pattern[i]];
        gamma_r += value * value;
    }
    if (gamma_r < 1e-12) gamma_r = 1e-12;
    st->w_keep[leaving_pos] = gamma_r;

    la->ftran(la->ctx, rho, st->tau);
    for (int32_t i = 0; i < ftran_nnz; i++) {
        const int32_t k = ftran_pattern[i];
        if (k == leaving_pos) continue;
        const double ratio = alpha_col[k] * inv_pivot;
        double gamma = st->w_keep[k] - 2.0 * ratio * st->tau[k]
                     + ratio * ratio * gamma_r;
        if (gamma < 1e-12) gamma = 1e-12;
        st->w_keep[k] = gamma;
        weights[k] = gamma;
    }
    double new_gamma_r = gamma_r * inv_pivot * inv_pivot;
    if (new_gamma_r < 1e-12) new_gamma_r = 1e-12;
    st->w_keep[leaving_pos] = new_gamma_r;
    weights[leaving_pos] = new_gamma_r;
}
#endif
