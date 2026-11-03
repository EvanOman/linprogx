/*
 * _ds2_stub_chuzc.c -- DS2 component A STUB: textbook Harris ratio test.
 *
 * PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * Placeholder so ds2_core can be developed end-to-end before the real
 * ratio test lands.  Two passes:
 *
 *   pass 1  Theta_max = min over eligible j of (|r_j| + tau) / |alpha_j|
 *   pass 2  entering  = argmax |alpha_j| over eligible j with the PLAIN
 *                       ratio -r_j / (sigma*alpha_j) <= Theta_max
 *
 * plus an EXPAND-style guaranteed minimum dual step (Gill/Murray/Saunders,
 * dual-side adaptation) so exactly-degenerate pivots cannot repeat a tie
 * forever.  tau grows by a fixed dose per call and is reset by the core at
 * every refactorization via ds2_ratio_state_new's counterpart -- here it is
 * simply clamped at tau_max, which bounds accumulated dual infeasibility
 * well below the 1e-7 exit-certification tolerances.
 *
 * NO bound flips: n_flip is always 0.  The bound-flipping (longest-step)
 * ratio test is component A's job, and ds2_core applies whatever flips it
 * returns.
 *
 * Included by _csparse.c only when src/linprogx/_ds2_chuzc.c is absent.
 */

typedef struct {
    double tau;        /* current expanded dual tolerance */
    double tau0;
    double tau_max;
    double dtau;       /* per-call growth = guaranteed minimum step dose */
} DS2StubRatioState;

/* Columns with |alpha| below this cannot define a break point. */
#define DS2_STUB_ALPHA_TOL 1e-9

void *ds2_ratio_state_new(int32_t m, int32_t n_total) {
    (void)m; (void)n_total;
    DS2StubRatioState *st = (DS2StubRatioState *)calloc(1, sizeof(DS2StubRatioState));
    if (st == NULL) return NULL;
    st->tau0 = 5e-10;
    st->tau_max = 1e-8;
    st->dtau = 5e-11;
    st->tau = st->tau0;
    return st;
}

void ds2_ratio_state_free(void *state) { free(state); }

void ds2_ratio_prepare(void *state, double delta, int32_t update_count) {
    (void)state;
    (void)delta;
    (void)update_count;
}

void ds2_ratio_bounds_changed(void *state, const double *lo_ext,
                              const double *hi_ext) {
    (void)state;
    (void)lo_ext;
    (void)hi_ext;
}

DS2Entering ds2_chuzc(
    const double *alpha_row,
    const int32_t *alpha_pattern, int32_t alpha_nnz,
    const double *r_ext, const int8_t *bound_status,
    const double *lo_ext, const double *hi_ext,
    int leaving_sigma, double dual_tol,
    void *ratio_state)
{
    (void)lo_ext; (void)hi_ext;
    DS2StubRatioState *st = (DS2StubRatioState *)ratio_state;
    double tau = dual_tol;
    double dtau = 0.0;
    if (st != NULL) {
        tau = st->tau;
        dtau = st->dtau;
        st->tau += st->dtau;
        if (st->tau > st->tau_max) st->tau = st->tau0;
    }

    DS2Entering out;
    out.entering = -1;
    out.theta_dual = 0.0;
    out.alpha_pivot = 0.0;
    out.n_flip = 0;
    out.flip_cols = NULL;

    const double sigma = (double)leaving_sigma;

    /* ---- pass 1: Harris-relaxed bound on the dual step ---- */
    double theta_max = HUGE_VAL;
    for (int32_t t = 0; t < alpha_nnz; t++) {
        int32_t j = alpha_pattern[t];
        int8_t bs = bound_status[j];
        if (bs == DS2_BASIC || bs == DS2_FIXED) continue;
        double a = sigma * alpha_row[j];
        if (bs == DS2_AT_LO) {
            if (a > -DS2_STUB_ALPHA_TOL) continue;
        } else if (bs == DS2_AT_HI) {
            if (a < DS2_STUB_ALPHA_TOL) continue;
        } else { /* DS2_FREE */
            if (fabs(a) < DS2_STUB_ALPHA_TOL) continue;
        }
        double bound = (fabs(r_ext[j]) + tau) / fabs(a);
        if (bound < theta_max) theta_max = bound;
    }
    if (theta_max == HUGE_VAL) return out;   /* dual unbounded */

    /* ---- pass 2: most stable pivot within the relaxed bound ---- */
    double best_alpha = 0.0;
    double best_ratio = 0.0;
    for (int32_t t = 0; t < alpha_nnz; t++) {
        int32_t j = alpha_pattern[t];
        int8_t bs = bound_status[j];
        if (bs == DS2_BASIC || bs == DS2_FIXED) continue;
        double a = sigma * alpha_row[j];
        if (bs == DS2_AT_LO) {
            if (a > -DS2_STUB_ALPHA_TOL) continue;
        } else if (bs == DS2_AT_HI) {
            if (a < DS2_STUB_ALPHA_TOL) continue;
        } else {
            if (fabs(a) < DS2_STUB_ALPHA_TOL) continue;
        }
        double ratio = -r_ext[j] / a;
        if (ratio < 0.0) ratio = 0.0;
        if (ratio > theta_max) continue;
        double abs_a = fabs(a);
        if (abs_a > best_alpha ||
            (abs_a == best_alpha && out.entering >= 0 && j < out.entering)) {
            best_alpha = abs_a;
            best_ratio = ratio;
            out.entering = j;
        }
    }
    if (out.entering < 0) return out;

    /* ---- EXPAND guaranteed minimum step ---- */
    double theta_eff = best_ratio;
    if (dtau > 0.0) {
        double floor_step = dtau / best_alpha;
        if (theta_eff < floor_step) theta_eff = floor_step;
    }
    /* theta_dual = r_q / alpha_row[q] = -sigma * theta_eff */
    out.theta_dual = -sigma * theta_eff;
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
    return ds2_chuzc(alpha_row, alpha_pattern, alpha_nnz, r_ext,
                     bound_status, lo_ext, hi_ext, leaving_sigma, dual_tol,
                     ratio_state);
}
