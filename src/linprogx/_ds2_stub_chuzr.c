/*
 * _ds2_stub_chuzr.c -- DS2 component B STUB: plain Dantzig leaving rule.
 *
 * PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * This is a placeholder so ds2_core can be developed and measured
 * end-to-end before the real pricing component lands.  It picks the basic
 * variable with the largest absolute bound violation and maintains no edge
 * weights at all -- it only honours the DS2_WEIGHT_BANNED protocol, which
 * a weight-free rule has to test explicitly.
 *
 * Included by _csparse.c only when src/linprogx/_ds2_chuzr.c is absent.
 */

DS2Leaving ds2_chuzr(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    double *weights,
    int32_t m, double feas_tol,
    void *pricing_state)
{
    (void)pricing_state;
    DS2Leaving out;
    out.basis_pos = -1;
    out.sigma = 0;
    out.violation = 0.0;

    double best = feas_tol;
    for (int32_t k = 0; k < m; k++) {
        if (weights[k] >= DS2_WEIGHT_BANNED) continue;
        int32_t j = basis[k];
        double v = x_B[k];
        double below = lo_ext[j] - v;   /* > 0 when below the lower bound */
        if (below > best) {
            best = below;
            out.basis_pos = k;
            out.sigma = 1;
            out.violation = below;
        }
        double above = v - hi_ext[j];   /* > 0 when above the upper bound */
        if (above > best) {
            best = above;
            out.basis_pos = k;
            out.sigma = -1;
            out.violation = above;
        }
    }
    return out;
}

void *ds2_pricing_state_new(int32_t m, int32_t n_total) {
    (void)m; (void)n_total;
    return NULL;
}

void ds2_pricing_state_free(void *state) { (void)state; }

void ds2_pricing_update(
    void *state,
    int32_t leaving_pos, int32_t entering,
    const double *rho, const int32_t *rho_pattern, int32_t rho_nnz,
    const double *alpha_col, const int32_t *ftran_pattern, int32_t ftran_nnz,
    double alpha_pivot, double *weights, int32_t m,
    const DS2LinAlg *la)
{
    (void)state; (void)leaving_pos; (void)entering;
    (void)rho; (void)rho_pattern; (void)rho_nnz;
    (void)alpha_col; (void)ftran_pattern; (void)ftran_nnz;
    (void)alpha_pivot; (void)weights; (void)m; (void)la;
}

void ds2_pricing_reset(void *state, double *weights, int32_t m,
                       int logical_basis, const DS2LinAlg *la)
{
    (void)state; (void)weights; (void)m; (void)logical_basis; (void)la;
}
