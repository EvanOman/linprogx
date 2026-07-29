/*
 * _ds2_core.c -- DS2: the dual-simplex rewrite's core.
 *
 * PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 * See docs/PROVENANCE.md.  The architecture below (logical form, in-place
 * dual Phase 1 by bound substitution, logical-basis start) was derived from
 * reading the HiGHS implementation under explicit owner authorisation.
 * No verbatim copying: this is an independent reimplementation.
 *
 * Reached two ways, both default OFF:
 *   LINPROGX_DS2=1        routes solve_eq_box_dual_simplex here
 *   solve_eq_box_ds2(...) calls it directly
 * The shipped dual simplex is untouched and stays byte-identical with the
 * gate unset (greenbea trace digest 679168a4baad36d6, 4,399 pivots).
 *
 * This file owns the loop, the basis, the LU and its refactorization cadence,
 * the phase bounds, the primal/dual updates, and the exit certification.  It
 * CALLS ds2_chuzr and ds2_chuzc through the fixed interfaces in _ds2_iface.h
 * and never inspects their state.
 *
 * WHAT IS DIFFERENT FROM THE SHIPPED PATH, AND WHY
 * ------------------------------------------------
 * 1. LOGICAL FORM, unconditionally.  The right-hand side is carried in the
 *    bounds of the logical columns -- lo[n+i] = hi[n+i] = -b_i -- and the
 *    simplex system's RHS is identically zero.  Row i reads
 *    `sum_j A[i,j] x_j + x_{n+i} = 0`, so x_{n+i} = -b_i reproduces Ax = b.
 *    Verified result-identical on the shipped path (4,399 pivots).  It is the
 *    prerequisite that makes dual phase 1 a bound swap: with the RHS living in
 *    bounds, overwriting the bound arrays deletes b, and there is nowhere else
 *    for it to hide.
 *
 * 2. NO BIG-M ANYWHERE.  The shipped path buys initial dual feasibility by
 *    parking one-sided columns at an invented bound 1e5 x scale away -- 93.4%
 *    of greenbea's columns -- converting a dual-sign problem into a huge
 *    primal-distance problem, and then pays a certification tax on the way out
 *    (gap-damage budgets, the non-committal "dual_unbounded_boxed" downgrade,
 *    an M x 100 retry).  DS2 establishes dual feasibility by the phase-1 bound
 *    substitution instead, so an empty ratio test is a GENUINE primal
 *    infeasibility certificate, optimality is tested against the true bounds
 *    with no artificial bound present at all, and none of that tax exists.
 *
 * 3. PHASES ARE DATA, NOT CONTROL FLOW.  There is no `if (phase == 1)` in the
 *    iteration body.  Phase 1 substitutes a bound table keyed only on which
 *    true bounds are finite; phase 2 restores the true bounds; the same loop
 *    runs on the same basis, factorization and weights.
 *
 * Non-negotiables preserved: eps untouched, every accepted answer re-certifies
 * dual feasibility in ORIGINAL units against the TRUE bounds, and every
 * threshold below is global -- no per-problem tuning.
 */

#include "_ds2_iface.h"

/* ---- global constants (no per-problem tuning) ---------------------------- */

/* Phase-1 bound substitution, keyed only on which true bounds are finite.
 * Free columns get the wide box so phase 1 prioritises killing their dual
 * infeasibilities; boxed and fixed columns collapse to a point because their
 * dual infeasibility is always removable by a bound flip, which changes no
 * duals, and is therefore never an obstruction. */
#define DS2_P1_FREE_BOX 1000.0

/* Dual feasibility tolerance handed to CHUZC. */
#define DS2_DUAL_TOL 1e-9
/* Sign tolerance for the drift-repair sweep at each refactorization. */
#define DS2_DRIFT_TOL 1e-9
/* A pivot below this is refused: the row is banned and re-chosen, never
 * pivoted on. */
#define DS2_PIVOT_MIN 1e-12
/* Refactorize immediately when the eta this pivot would create is violent. */
#define DS2_PIVOT_VIOLENT_LO 1e-6
#define DS2_PIVOT_VIOLENT_HI 1e6
/* Acceptance tolerance for the exit certificate and for the clean-state
 * audit that precedes it. */
#define DS2_CERT_TOL 1e-7
/* Bounded number of phase-1 re-entries triggered by the audit, and of
 * recoveries from a fully banned CHUZR.  Exhausting either reports honestly
 * rather than fabricating an optimum. */
#define DS2_MAX_AUDIT_ROUNDS 20
#define DS2_MAX_BAN_ROUNDS 20

/* ---- env gates (all default to the shipped-safe choice) ------------------ */

static int ds2_env_int(const char *name, int fallback)
{
    const char *e = getenv(name);
    if (e == NULL || *e == '\0') return fallback;
    return atoi(e);
}

static int ds2_enabled(void) { return ds2_env_int("LINPROGX_DS2", 0) != 0; }

static double ds2_perturb_fraction(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x *= UINT64_C(0x2545F4914F6CDD1D);
    return (double)(x >> 11) * 0x1.0p-53;
}

static void ds2_la_ftran(void *ctx, const double *rhs, double *out)
{
    lu_ftran((const LUContext *)ctx, rhs, out);
}

static int32_t ds2_la_btran_unit(void *ctx, int32_t pos, double *out,
                                 int32_t *pattern)
{
    return lu_btran_sparse((LUContext *)ctx, pos, out, pattern);
}

/* ------------------------------------------------------------------------ *
 * The solver.
 * ------------------------------------------------------------------------ */

static PyObject *ds2_solve(
    CSRMatrixObject *self,
    PyObject *c_obj, PyObject *b_obj, PyObject *lo_obj, PyObject *hi_obj,
    Py_ssize_t max_iter_arg, double tol)
{
    if (self->rows > INT32_MAX || self->cols > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError,
                        "matrix too large for 32-bit factorization");
        return NULL;
    }

    const Py_ssize_t m_s = self->rows;
    const Py_ssize_t n_s = self->cols;
    const int32_t m = (int32_t)m_s;
    const int32_t n = (int32_t)n_s;
    const int32_t n_total = n + m;

    const int opt_logical_basis = ds2_env_int("LINPROGX_DS2_LOGICAL_BASIS", 0);
    const int opt_phase1 = ds2_env_int("LINPROGX_DS2_PHASE1", 1);
    const int opt_report = ds2_env_int("LINPROGX_DS2_REPORT", 0);
    const int opt_perturb = ds2_env_int("LINPROGX_DS2_PERTURB", 0);
    const int opt_scale = ds2_env_int("LINPROGX_DS2_SCALE", 0);
    uint64_t perturb_rng = UINT64_C(0x9E3779B97F4A7C15);
    int32_t refac_interval = (int32_t)ds2_env_int("LINPROGX_DS2_REFAC", 500);
    if (refac_interval < 1) refac_interval = 1;

    const Py_ssize_t max_iter =
        max_iter_arg > 0 ? max_iter_arg
                         : (Py_ssize_t)(50 * (m_s + n_s) < 100000
                                            ? 50 * (m_s + n_s) : 100000);

    PyObject *result = NULL;
    const char *status = "numerical_error";
    Py_ssize_t iterations = 0;
    LUContext *lu = NULL;
    void *pricing_state = NULL;
    void *ratio_state = NULL;
    int states_created = 0;

    /* statistics */
    int64_t total_refacs = 0, stat_flips = 0, stat_degenerate = 0;
    int64_t stat_banned_rows = 0, stat_cost_shifts = 0;
    int64_t stat_audit_rounds = 0, phase1_iters = 0;
    double phase1_dual_obj = 0.0;

#define DS2_CALLOC_D(count) calloc((size_t)((count) > 0 ? (count) : 1), sizeof(double))
#define DS2_CALLOC_I(count) calloc((size_t)((count) > 0 ? (count) : 1), sizeof(int32_t))
#define DS2_CALLOC_B(count) calloc((size_t)((count) > 0 ? (count) : 1), sizeof(int8_t))

    double *c_ext = DS2_CALLOC_D(n_total);
    double *c_orig = DS2_CALLOC_D(n);
    double *c_shift = DS2_CALLOC_D(n_total);
    double *lo_ext = DS2_CALLOC_D(n_total);
    double *hi_ext = DS2_CALLOC_D(n_total);
    double *lo_true = DS2_CALLOC_D(n_total);
    double *hi_true = DS2_CALLOC_D(n_total);
    double *x_ext = DS2_CALLOC_D(n_total);
    double *r_ext = DS2_CALLOC_D(n_total);
    int8_t *bound_status = DS2_CALLOC_B(n_total);
    int32_t *basis_pos = DS2_CALLOC_I(n_total);

    double *b = DS2_CALLOC_D(m);
    double *y = DS2_CALLOC_D(m);
    double *x_B = DS2_CALLOC_D(m);
    double *rhs = DS2_CALLOC_D(m);
    double *rho = DS2_CALLOC_D(m);
    double *alpha_col = DS2_CALLOC_D(m);
    double *e_i = DS2_CALLOC_D(m);
    double *c_B = DS2_CALLOC_D(m);
    double *weights = DS2_CALLOC_D(m);
    double *flip_rhs = DS2_CALLOC_D(m);
    int32_t *basis = DS2_CALLOC_I(m);
    int32_t *rho_pat = DS2_CALLOC_I(m);
    int32_t *ftran_pat = DS2_CALLOC_I(m);

    double *alpha_scratch = DS2_CALLOC_D(n_total);
    int32_t *alpha_pattern = DS2_CALLOC_I(n_total);
    int8_t *alpha_touched = DS2_CALLOC_B(n_total);

    const int32_t b_nnz_max = (int32_t)(self->nnz > 0 ? self->nnz : 1) + m;
    int32_t *b_indptr = DS2_CALLOC_I(m + 1);
    int32_t *b_indices = DS2_CALLOC_I(b_nnz_max);
    double *b_values = DS2_CALLOC_D(b_nnz_max);

    double *row_scale = DS2_CALLOC_D(m);
    double *col_scale = DS2_CALLOC_D(n);
    double *scaled_csc = DS2_CALLOC_D(self->nnz);
    double *scaled_csr = DS2_CALLOC_D(self->nnz);
    int32_t *csr_idx32 = DS2_CALLOC_I(self->nnz);

#undef DS2_CALLOC_D
#undef DS2_CALLOC_I
#undef DS2_CALLOC_B

    if (c_ext == NULL || c_orig == NULL || c_shift == NULL || lo_ext == NULL ||
        hi_ext == NULL || lo_true == NULL || hi_true == NULL || x_ext == NULL ||
        r_ext == NULL || bound_status == NULL || basis_pos == NULL ||
        b == NULL || y == NULL || x_B == NULL || rhs == NULL || rho == NULL ||
        alpha_col == NULL || e_i == NULL || c_B == NULL || weights == NULL ||
        flip_rhs == NULL || basis == NULL || rho_pat == NULL ||
        ftran_pat == NULL || alpha_scratch == NULL || alpha_pattern == NULL ||
        alpha_touched == NULL || b_indptr == NULL || b_indices == NULL ||
        b_values == NULL || row_scale == NULL || col_scale == NULL ||
        scaled_csc == NULL || scaled_csr == NULL || csr_idx32 == NULL) {
        PyErr_NoMemory();
        goto done;
    }

    /* ---- read the problem ------------------------------------------------ */
    if (fill_double_array(c_obj, n_s, c_ext, "c") != 0 ||
        fill_double_array(b_obj, m_s, b, "b") != 0 ||
        fill_double_array(lo_obj, n_s, lo_ext, "lo") != 0 ||
        fill_double_array(hi_obj, n_s, hi_ext, "hi") != 0) {
        goto done;
    }
    memcpy(c_orig, c_ext, (size_t)n * sizeof(double));
    for (Py_ssize_t p = 0; p < self->nnz; p++)
        csr_idx32[p] = (int32_t)self->indices[p];

    /* ---- Ruiz equilibration ---------------------------------------------- *
     * Same rule as the shipped path: skip entirely when the matrix is already
     * balanced (row inf-norm ratio < 100), because the round-trip
     * scale/unscale error can exceed tight absolute tolerances on
     * well-conditioned models. */
    int ruiz_active = 0;
    {
        double *row_norms = rhs;      /* borrowed, cleared below */
        double *col_norms = r_ext;    /* borrowed, cleared below */

        for (int32_t j = 0; j < n; j++) col_scale[j] = 1.0;
        for (int32_t i = 0; i < m; i++) row_scale[i] = 1.0;

        double min_rn = 1e300, max_rn = 0.0;
        for (int32_t i = 0; i < m; i++) row_norms[i] = 0.0;
        for (int32_t j = 0; j < n; j++) {
            for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                int32_t row = (int32_t)self->csc_rows[p];
                double av = fabs(self->csc_data[p]);
                if (av > row_norms[row]) row_norms[row] = av;
            }
        }
        for (int32_t i = 0; i < m; i++) {
            if (row_norms[i] > 0.0) {
                if (row_norms[i] < min_rn) min_rn = row_norms[i];
                if (row_norms[i] > max_rn) max_rn = row_norms[i];
            }
        }
        if (opt_scale != 1 &&
            ((min_rn > 0.0 && max_rn / min_rn >= 100.0) ||
             opt_scale == 2 || opt_scale == 3)) {
            ruiz_active = 1;
        }

        if (ruiz_active) {
            if (opt_scale == 3) {
                double *row_mins = x_B;  /* borrowed, overwritten by solve */
                double *col_mins = x_ext;
                for (int pass = 0; pass < 6; pass++) {
                    for (int32_t i = 0; i < m; i++) {
                        row_norms[i] = 0.0;
                        row_mins[i] = 1e300;
                    }
                    for (int32_t j = 0; j < n; j++) {
                        for (Py_ssize_t p = self->csc_indptr[j];
                             p < self->csc_indptr[j + 1]; p++) {
                            const int32_t row =
                                (int32_t)self->csc_rows[p];
                            const double value = fabs(
                                self->csc_data[p] * row_scale[row] *
                                col_scale[j]);
                            if (value == 0.0) continue;
                            if (value > row_norms[row])
                                row_norms[row] = value;
                            if (value < row_mins[row])
                                row_mins[row] = value;
                        }
                    }
                    for (int32_t i = 0; i < m; i++) {
                        if (row_norms[i] > 0.0 &&
                            row_mins[i] < 1e300) {
                            row_scale[i] /=
                                sqrt(row_mins[i] * row_norms[i]);
                        }
                    }

                    for (int32_t j = 0; j < n; j++) {
                        col_norms[j] = 0.0;
                        col_mins[j] = 1e300;
                        for (Py_ssize_t p = self->csc_indptr[j];
                             p < self->csc_indptr[j + 1]; p++) {
                            const int32_t row =
                                (int32_t)self->csc_rows[p];
                            const double value = fabs(
                                self->csc_data[p] * row_scale[row] *
                                col_scale[j]);
                            if (value == 0.0) continue;
                            if (value > col_norms[j])
                                col_norms[j] = value;
                            if (value < col_mins[j])
                                col_mins[j] = value;
                        }
                        if (col_norms[j] > 0.0 &&
                            col_mins[j] < 1e300) {
                            col_scale[j] /=
                                sqrt(col_mins[j] * col_norms[j]);
                        }
                    }
                }
            } else {
                for (int pass = 0; pass < 10; pass++) {
                    for (int32_t i = 0; i < m; i++) row_norms[i] = 0.0;
                    for (int32_t j = 0; j < n; j++) col_norms[j] = 0.0;
                    for (int32_t j = 0; j < n; j++) {
                        for (Py_ssize_t p = self->csc_indptr[j];
                             p < self->csc_indptr[j + 1]; p++) {
                            int32_t row = (int32_t)self->csc_rows[p];
                            double v = fabs(self->csc_data[p] *
                                            row_scale[row] * col_scale[j]);
                            if (v > row_norms[row]) row_norms[row] = v;
                            if (v > col_norms[j]) col_norms[j] = v;
                        }
                    }
                    for (int32_t i = 0; i < m; i++)
                        if (row_norms[i] > 0.0)
                            row_scale[i] /= sqrt(row_norms[i]);
                    for (int32_t j = 0; j < n; j++)
                        if (col_norms[j] > 0.0)
                            col_scale[j] /= sqrt(col_norms[j]);
                }
                /* one l2 balancing pass */
                for (int32_t i = 0; i < m; i++) row_norms[i] = 0.0;
                for (int32_t j = 0; j < n; j++) col_norms[j] = 0.0;
                for (int32_t j = 0; j < n; j++) {
                    for (Py_ssize_t p = self->csc_indptr[j];
                         p < self->csc_indptr[j + 1]; p++) {
                        int32_t row = (int32_t)self->csc_rows[p];
                        double v = self->csc_data[p] * row_scale[row] *
                                   col_scale[j];
                        row_norms[row] += v * v;
                        col_norms[j] += v * v;
                    }
                }
                for (int32_t i = 0; i < m; i++)
                    if (row_norms[i] > 0.0)
                        row_scale[i] /= sqrt(sqrt(row_norms[i]));
                for (int32_t j = 0; j < n; j++)
                    if (col_norms[j] > 0.0)
                        col_scale[j] /= sqrt(sqrt(col_norms[j]));
            }

            for (int32_t i = 0; i < m; i++) {
                if (row_scale[i] < 1e-8) row_scale[i] = 1e-8;
                else if (row_scale[i] > 1e8) row_scale[i] = 1e8;
                if (opt_scale == 2)
                    row_scale[i] = exp2(round(log2(row_scale[i])));
            }
            for (int32_t j = 0; j < n; j++) {
                if (col_scale[j] < 1e-8) col_scale[j] = 1e-8;
                else if (col_scale[j] > 1e8) col_scale[j] = 1e8;
                if (opt_scale == 2)
                    col_scale[j] = exp2(round(log2(col_scale[j])));
            }
            for (int32_t j = 0; j < n; j++) {
                for (Py_ssize_t p = self->csc_indptr[j];
                     p < self->csc_indptr[j + 1]; p++) {
                    int32_t row = (int32_t)self->csc_rows[p];
                    scaled_csc[p] = self->csc_data[p] * row_scale[row] * col_scale[j];
                }
            }
            for (int32_t i = 0; i < m; i++) {
                double rs = row_scale[i];
                for (Py_ssize_t p = self->indptr[i]; p < self->indptr[i + 1]; p++)
                    scaled_csr[p] = self->data[p] * rs * col_scale[csr_idx32[p]];
            }
            for (int32_t j = 0; j < n; j++) {
                c_ext[j] *= col_scale[j];
                if (isfinite(lo_ext[j])) lo_ext[j] /= col_scale[j];
                if (isfinite(hi_ext[j])) hi_ext[j] /= col_scale[j];
            }
            for (int32_t i = 0; i < m; i++) b[i] *= row_scale[i];
        } else {
            memcpy(scaled_csc, self->csc_data, (size_t)self->nnz * sizeof(double));
            memcpy(scaled_csr, self->data, (size_t)self->nnz * sizeof(double));
        }

        memset(rhs, 0, (size_t)m * sizeof(double));
        memset(r_ext, 0, (size_t)n_total * sizeof(double));
    }
    const double *a_data = scaled_csc;

    /* ---- logical form: the RHS moves into the logical column bounds ------ */
    for (int32_t i = 0; i < m; i++) {
        c_ext[n + i] = 0.0;
        lo_ext[n + i] = -b[i];
        hi_ext[n + i] = -b[i];
    }
    /* True bounds for the whole extended problem, saved before any phase-1
     * substitution.  Every optimality and certification test reads these. */
    memcpy(lo_true, lo_ext, (size_t)n_total * sizeof(double));
    memcpy(hi_true, hi_ext, (size_t)n_total * sizeof(double));

    /* ---- 1. starting basis ---------------------------------------------- *
     * Either the pure logical basis (B = I exactly -- HiGHS's default, and the
     * unique basis on which exact steepest-edge weights are free) or the
     * singleton-cascade triangular crash.  The crash is DS2's default because
     * it measures better here: with the stub components it wins greenbea
     * (5,292 vs 6,730 pivots), degen2 (1,446 vs 2,081) and sierra (779 vs
     * 1,390), and loses 25fv47 and greenbeb.  Component B may prefer B = I for
     * free exact edge weights, hence the switch. */
    int basis_is_logical = 1;
    if (opt_logical_basis) {
        for (int32_t k = 0; k < m; k++) basis[k] = n + k;
    } else {
        int8_t *row_covered = calloc((size_t)(m > 0 ? m : 1), sizeof(int8_t));
        int32_t *uncov = malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
        int8_t *col_done = calloc((size_t)(n > 0 ? n : 1), sizeof(int8_t));
        int32_t *queue = malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
        DsCrashCand *cand = malloc((size_t)(n > 0 ? n : 1) * sizeof(DsCrashCand));
        if (row_covered == NULL || uncov == NULL || col_done == NULL ||
            queue == NULL || cand == NULL) {
            free(row_covered); free(uncov); free(col_done); free(queue); free(cand);
            PyErr_NoMemory();
            goto done;
        }
        for (int32_t j = 0; j < n; j++) {
            int32_t nnz_j = (int32_t)(self->csc_indptr[j + 1] - self->csc_indptr[j]);
            uncov[j] = nnz_j;
            int lo_fin = isfinite(lo_ext[j]);
            int hi_fin = isfinite(hi_ext[j]);
            int32_t pen;
            if (!lo_fin && !hi_fin) pen = 0;
            else if (lo_fin && hi_fin) pen = (hi_ext[j] - lo_ext[j] <= 1e-30) ? 3 : 2;
            else pen = 1;
            cand[j].col = j;
            cand[j].penalty = pen;
            cand[j].nnz = nnz_j;
        }
        qsort(cand, (size_t)(n > 0 ? n : 0), sizeof(DsCrashCand), ds_crash_cand_cmp);
        int32_t qhead = 0, qtail = 0, n_basis = 0;
        for (int32_t idx = 0; idx < n; idx++)
            if (uncov[cand[idx].col] == 1) queue[qtail++] = cand[idx].col;
        while (qhead < qtail && n_basis < m) {
            int32_t j = queue[qhead++];
            if (col_done[j] || uncov[j] != 1) continue;
            col_done[j] = 1;
            int32_t pr = -1;
            double pv = 0.0, colmax = 0.0;
            for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                double av = fabs(a_data[p]);
                if (av > colmax) colmax = av;
                int32_t row = (int32_t)self->csc_rows[p];
                if (!row_covered[row]) { pr = row; pv = av; }
            }
            if (pr < 0 || colmax <= 1e-12) continue;
            if (pv < DS_CRASH_STAB * colmax) continue;
            basis[n_basis++] = j;
            row_covered[pr] = 1;
            for (Py_ssize_t p = self->indptr[pr]; p < self->indptr[pr + 1]; p++) {
                int32_t jj = csr_idx32[p];
                if (!col_done[jj] && uncov[jj] > 0) {
                    uncov[jj]--;
                    if (uncov[jj] == 1 && qtail < n) queue[qtail++] = jj;
                }
            }
        }
        if (n_basis > 0) basis_is_logical = 0;
        for (int32_t i = 0; i < m; i++)
            if (!row_covered[i]) basis[n_basis++] = n + i;
        free(row_covered); free(uncov); free(col_done); free(queue); free(cand);
    }

    /* ---- 2. factorize, with the identity as the always-available fallback - */
    {
        lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows, a_data,
                                basis, b_indptr, b_indices, b_values);
        int reject = (lu == NULL || lu->singular_step >= 0);
        if (!reject && !basis_is_logical) {
            double dmax = 0.0, dmin = 1e300;
            for (int32_t k = 0; k < m; k++) {
                double d = fabs(lu->u_diag[k]);
                if (d > dmax) dmax = d;
                if (d > 0.0 && d < dmin) dmin = d;
            }
            double growth = (dmin > 0.0 && dmin < 1e300) ? dmax / dmin : INFINITY;
            if (growth > DS_CRASH_MAX_GROWTH) reject = 1;
        }
        if (reject) {
            lu_context_free(lu);
            lu = NULL;
            for (int32_t i = 0; i < m; i++) basis[i] = n + i;
            basis_is_logical = 1;
            lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows,
                                    a_data, basis, b_indptr, b_indices, b_values);
            if (lu == NULL || lu->singular_step >= 0) {
                status = "numerical_error";
                goto build_result;
            }
        }
        lu_build_transposes(lu);
    }

    for (int32_t j = 0; j < n_total; j++) basis_pos[j] = -1;
    for (int32_t k = 0; k < m; k++) basis_pos[basis[k]] = k;

    /* A component is free to return NULL from its state constructor -- the
     * Dantzig stub does -- so NULL is never treated as failure. */
    pricing_state = ds2_pricing_state_new(m, n_total);
    ratio_state = ds2_ratio_state_new(m, n_total);
    states_created = 1;
    if (ratio_state == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    DS2LinAlg la = {lu, ds2_la_ftran, ds2_la_btran_unit};
    for (int32_t k = 0; k < m; k++) weights[k] = 1.0;
    ds2_pricing_reset(pricing_state, weights, m, basis_is_logical, &la);

    /* ---- 3. duals, phase decision, nonbasic placement -------------------- */
    for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
    lu_btran(lu, c_B, y);
    memset(lu->ws_v, 0, (size_t)m * sizeof(double));

    int32_t dual_infeas_count = 0;
    for (int32_t j = 0; j < n_total; j++) {
        if (basis_pos[j] >= 0) { bound_status[j] = DS2_BASIC; continue; }
        double rj = c_ext[j];
        if (j < n) {
            for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++)
                rj -= a_data[p] * y[(int32_t)self->csc_rows[p]];
        } else {
            rj -= y[j - n];
        }
        r_ext[j] = rj;
        /* Count only the infeasibilities phase 1 exists to remove: a boxed or
         * fixed column's wrong sign is removable by a bound flip, which
         * changes no duals, so it is never an obstruction. */
        int lo_fin = isfinite(lo_true[j]);
        int hi_fin = isfinite(hi_true[j]);
        if ((!lo_fin && !hi_fin) || lo_fin != hi_fin) {
            if ((rj > 0.0 && !lo_fin) || (rj < 0.0 && !hi_fin)) dual_infeas_count++;
        }
    }

    int ds_phase = (opt_phase1 && dual_infeas_count > 0) ? 1 : 2;

    if (ds_phase == 1) {
        /* Phase-1 bound substitution.  Keyed ONLY on which true bounds are
         * finite -- the finite bound VALUES are discarded, which is exactly
         * why the phase-1 trajectory is invariant to b. */
        for (int32_t j = 0; j < n_total; j++) {
            int lo_fin = isfinite(lo_true[j]);
            int hi_fin = isfinite(hi_true[j]);
            if (!lo_fin && !hi_fin)  { lo_ext[j] = -DS2_P1_FREE_BOX;
                                       hi_ext[j] =  DS2_P1_FREE_BOX; }
            else if (!lo_fin)        { lo_ext[j] = -1.0; hi_ext[j] = 0.0; }
            else if (!hi_fin)        { lo_ext[j] =  0.0; hi_ext[j] = 1.0; }
            else                     { lo_ext[j] =  0.0; hi_ext[j] = 0.0; }
        }
    }

    /* Place every nonbasic at the end of its box that its reduced cost wants.
     * In phase 1 every box is finite, so this always succeeds and dual
     * feasibility holds by construction -- no big-M is ever needed. */
    for (int32_t j = 0; j < n_total; j++) {
        if (basis_pos[j] >= 0) {
            bound_status[j] = DS2_BASIC; x_ext[j] = 0.0; continue;
        }
        int lo_fin = isfinite(lo_ext[j]);
        int hi_fin = isfinite(hi_ext[j]);
        if (lo_fin && hi_fin && lo_ext[j] == hi_ext[j]) {
            bound_status[j] = DS2_FIXED; x_ext[j] = lo_ext[j];
        } else if (r_ext[j] >= 0.0 && lo_fin) {
            bound_status[j] = DS2_AT_LO; x_ext[j] = lo_ext[j];
        } else if (r_ext[j] < 0.0 && hi_fin) {
            bound_status[j] = DS2_AT_HI; x_ext[j] = hi_ext[j];
        } else if (lo_fin) {
            bound_status[j] = DS2_AT_LO; x_ext[j] = lo_ext[j];
        } else if (hi_fin) {
            bound_status[j] = DS2_AT_HI; x_ext[j] = hi_ext[j];
        } else {
            bound_status[j] = DS2_FREE; x_ext[j] = 0.0;
        }
    }
    ds2_ratio_bounds_changed(ratio_state, lo_ext, hi_ext);

    /* ---- 4. the loop ----------------------------------------------------- */
    {
        int x_B_needs_recompute = 1;
        int x_B_fresh = 0;
        int32_t iters_since_refac = 0;
        int32_t n_banned = 0;
        int32_t ban_rounds = 0;
        Py_ssize_t iter = 0;

        for (iter = 0; iter < max_iter; iter++) {
            iterations = iter;

            /* ---- 4a. x_B = B^{-1} (0 - A_N x_N) ---- */
            if (x_B_needs_recompute) {
                memset(rhs, 0, (size_t)m * sizeof(double));
                for (int32_t j = 0; j < n_total; j++) {
                    if (basis_pos[j] >= 0) continue;
                    double xj = x_ext[j];
                    if (xj == 0.0) continue;
                    if (j < n) {
                        for (Py_ssize_t p = self->csc_indptr[j];
                             p < self->csc_indptr[j + 1]; p++)
                            rhs[(int32_t)self->csc_rows[p]] -= a_data[p] * xj;
                    } else {
                        rhs[j - n] -= xj;
                    }
                }
                lu_ftran(lu, rhs, x_B);
                x_B_needs_recompute = 0;
                x_B_fresh = 1;
            }

            /* ---- 4b. CHUZR ---- */
            DS2Leaving leaving = ds2_chuzr(basis, x_B, lo_ext, hi_ext,
                                           weights, m, tol, pricing_state);

            if (leaving.basis_pos < 0) {
                if (!x_B_fresh) {
                    /* Never conclude anything from incrementally maintained
                     * primal values -- force one clean solve first. */
                    x_B_needs_recompute = 1;
                    continue;
                }
                if (n_banned > 0) {
                    /* "No leaving row" may just mean every violated row is
                     * banned.  Refactorize -- which lifts every ban -- and
                     * look again before believing it. */
                    if (ban_rounds >= DS2_MAX_BAN_ROUNDS) {
                        status = "numerical_error";
                        iterations = iter;
                        break;
                    }
                    ban_rounds++;
                    lu_context_free(lu);
                    lu = ds_factorize_basis(m, n, self->csc_indptr,
                                            self->csc_rows, a_data, basis,
                                            b_indptr, b_indices, b_values);
                    if (lu != NULL && lu->singular_step >= 0) {
                        lu = ds_repair_singular_basis(
                            lu, m, n, self->csc_indptr, self->csc_rows, a_data,
                            basis, basis_pos, bound_status, x_ext, r_ext,
                            lo_ext, hi_ext, b_indptr, b_indices, b_values, 10);
                    }
                    if (lu == NULL || lu->singular_step >= 0) {
                        status = "numerical_error";
                        iterations = iter;
                        break;
                    }
                    lu_build_transposes(lu);
                    total_refacs++;
                    iters_since_refac = 0;
                    n_banned = 0;
                    for (int32_t k = 0; k < m; k++) weights[k] = 1.0;
                    la.ctx = lu;
                    ds2_pricing_reset(pricing_state, weights, m, 0, &la);
                    x_B_needs_recompute = 1;
                    x_B_fresh = 0;
                    continue;
                }

                if (ds_phase == 1) {
                    /* Phase-1 optimality.  The phase-1 dual objective
                     * sum_j x_j r_j over nonbasics is exactly minus the
                     * weighted sum of dual infeasibilities, so reaching zero
                     * IS dual feasibility for the true bounds. */
                    double z1 = 0.0;
                    for (int32_t j = 0; j < n_total; j++)
                        if (basis_pos[j] < 0) z1 += x_ext[j] * r_ext[j];
                    phase1_dual_obj = z1;
                    phase1_iters = (int64_t)iter;

                    memcpy(lo_ext, lo_true, (size_t)n_total * sizeof(double));
                    memcpy(hi_ext, hi_true, (size_t)n_total * sizeof(double));
                    ds2_ratio_bounds_changed(ratio_state, lo_ext, hi_ext);

                    /* Re-place every nonbasic against the TRUE bounds.  A
                     * column whose reduced cost still points at an infinite
                     * bound is a residual dual infeasibility phase 1 could not
                     * remove within tolerance; shift its cost so its dual is
                     * exactly zero and record the shift, which the clean-state
                     * audit strips before anything is certified. */
                    for (int32_t j = 0; j < n_total; j++) {
                        if (basis_pos[j] >= 0) {
                            bound_status[j] = DS2_BASIC; x_ext[j] = 0.0;
                            continue;
                        }
                        int lo_fin = isfinite(lo_ext[j]);
                        int hi_fin = isfinite(hi_ext[j]);
                        if (lo_fin && hi_fin && lo_ext[j] == hi_ext[j]) {
                            bound_status[j] = DS2_FIXED; x_ext[j] = lo_ext[j];
                        } else if (r_ext[j] >= 0.0 && lo_fin) {
                            bound_status[j] = DS2_AT_LO; x_ext[j] = lo_ext[j];
                        } else if (r_ext[j] < 0.0 && hi_fin) {
                            bound_status[j] = DS2_AT_HI; x_ext[j] = hi_ext[j];
                        } else {
                            double shift = -r_ext[j];
                            c_ext[j] += shift;
                            c_shift[j] += shift;
                            r_ext[j] = 0.0;
                            stat_cost_shifts++;
                            if (!lo_fin && !hi_fin) {
                                bound_status[j] = DS2_FREE; x_ext[j] = 0.0;
                            } else if (lo_fin) {
                                bound_status[j] = DS2_AT_LO; x_ext[j] = lo_ext[j];
                            } else {
                                bound_status[j] = DS2_AT_HI; x_ext[j] = hi_ext[j];
                            }
                        }
                    }
                    ds_phase = 2;
                    for (int32_t k = 0; k < m; k++) weights[k] = 1.0;
                    la.ctx = lu;
                    ds2_pricing_reset(pricing_state, weights, m, 0, &la);
                    n_banned = 0;
                    x_B_needs_recompute = 1;
                    x_B_fresh = 0;
                    if (opt_report) {
                        fprintf(stderr,
                                "[ds2] phase1 = %lld pivots, dual objective "
                                "%.6e, cost shifts %lld\n",
                                (long long)phase1_iters, phase1_dual_obj,
                                (long long)stat_cost_shifts);
                        fflush(stderr);
                    }
                    iterations = iter + 1;
                    continue;
                }

                /* ---- phase-2 optimality: the CLEAN-STATE AUDIT ----
                 *
                 * The loop has reached primal feasibility for the WORKING
                 * problem, which may still carry anti-degeneracy cost shifts
                 * and reduced costs the Harris band has let drift.  Nothing
                 * here is trusted: every shift is stripped, the duals are
                 * recomputed from the factorization, and every nonbasic is
                 * re-tested against the true bounds.
                 *
                 * Three outcomes, in increasing severity:
                 *   - a wrong-signed reduced cost with an opposite finite
                 *     bound is repaired by a FLIP, which changes no duals;
                 *   - a wrong-signed reduced cost with NO opposite bound is a
                 *     genuine dual infeasibility only pivoting can fix, so
                 *     phase 1 is re-entered to drive it out;
                 *   - neither, and the basis is optimal.
                 * The phase-1 re-entry is bounded, and exhausting the bound
                 * reports dual_infeasible rather than a fabricated optimum. */
                {
                    int removed_shift = 0;
                    for (int32_t j = 0; j < n_total; j++) {
                        if (c_shift[j] != 0.0) {
                            c_ext[j] -= c_shift[j];
                            c_shift[j] = 0.0;
                            removed_shift = 1;
                        }
                    }
                    for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
                    lu_btran(lu, c_B, y);
                    memset(lu->ws_v, 0, (size_t)m * sizeof(double));

                    int flipped = 0;
                    int32_t unfixable = 0;
                    double worst_viol = 0.0;
                    for (int32_t j = 0; j < n_total; j++) {
                        if (basis_pos[j] >= 0) { r_ext[j] = 0.0; continue; }
                        double rj = c_ext[j];
                        if (j < n) {
                            for (Py_ssize_t p = self->csc_indptr[j];
                                 p < self->csc_indptr[j + 1]; p++)
                                rj -= a_data[p] * y[(int32_t)self->csc_rows[p]];
                        } else {
                            rj -= y[j - n];
                        }
                        r_ext[j] = rj;
                        if (bound_status[j] == DS2_FIXED) continue;
                        double dtol = DS2_CERT_TOL * (1.0 + fabs(c_ext[j]));
                        if (bound_status[j] == DS2_AT_LO && rj < -dtol) {
                            if (isfinite(hi_ext[j])) {
                                bound_status[j] = DS2_AT_HI;
                                x_ext[j] = hi_ext[j];
                                flipped = 1;
                            } else {
                                unfixable++;
                                if (-rj > worst_viol) worst_viol = -rj;
                            }
                        } else if (bound_status[j] == DS2_AT_HI && rj > dtol) {
                            if (isfinite(lo_ext[j])) {
                                bound_status[j] = DS2_AT_LO;
                                x_ext[j] = lo_ext[j];
                                flipped = 1;
                            } else {
                                unfixable++;
                                if (rj > worst_viol) worst_viol = rj;
                            }
                        } else if (bound_status[j] == DS2_FREE &&
                                   fabs(rj) > dtol) {
                            unfixable++;
                            if (fabs(rj) > worst_viol) worst_viol = fabs(rj);
                        }
                    }

                    if (unfixable > 0) {
                        if (stat_audit_rounds >= DS2_MAX_AUDIT_ROUNDS) {
                            status = "dual_infeasible";
                            iterations = iter;
                            break;
                        }
                        stat_audit_rounds++;
                        if (opt_report) {
                            fprintf(stderr,
                                    "[ds2] audit at %lld: %d unfixable dual "
                                    "infeasibilities (worst %.3e) -> "
                                    "re-entering phase 1 (round %lld)\n",
                                    (long long)iter, (int)unfixable, worst_viol,
                                    (long long)stat_audit_rounds);
                            fflush(stderr);
                        }
                        for (int32_t j = 0; j < n_total; j++) {
                            int lf = isfinite(lo_true[j]);
                            int hf = isfinite(hi_true[j]);
                            if (!lf && !hf) { lo_ext[j] = -DS2_P1_FREE_BOX;
                                              hi_ext[j] =  DS2_P1_FREE_BOX; }
                            else if (!lf)   { lo_ext[j] = -1.0; hi_ext[j] = 0.0; }
                            else if (!hf)   { lo_ext[j] =  0.0; hi_ext[j] = 1.0; }
                            else            { lo_ext[j] =  0.0; hi_ext[j] = 0.0; }
                            if (basis_pos[j] >= 0) {
                                bound_status[j] = DS2_BASIC; x_ext[j] = 0.0;
                            } else if (lo_ext[j] == hi_ext[j]) {
                                bound_status[j] = DS2_FIXED; x_ext[j] = lo_ext[j];
                            } else if (r_ext[j] >= 0.0) {
                                bound_status[j] = DS2_AT_LO; x_ext[j] = lo_ext[j];
                            } else {
                                bound_status[j] = DS2_AT_HI; x_ext[j] = hi_ext[j];
                            }
                        }
                        ds2_ratio_bounds_changed(ratio_state, lo_ext, hi_ext);
                        ds_phase = 1;
                        for (int32_t k = 0; k < m; k++) weights[k] = 1.0;
                        la.ctx = lu;
                        ds2_pricing_reset(pricing_state, weights, m, 0, &la);
                        n_banned = 0;
                        x_B_needs_recompute = 1;
                        x_B_fresh = 0;
                        iterations = iter + 1;
                        continue;
                    }

                    if (removed_shift || flipped) {
                        /* The audit changed the point; re-test from a clean
                         * primal solve before claiming anything. */
                        x_B_needs_recompute = 1;
                        x_B_fresh = 0;
                        iterations = iter + 1;
                        continue;
                    }
                }

                status = "optimal";
                iterations = iter;
                break;
            }

            const int32_t leaving_pos = leaving.basis_pos;
            const int leaving_sigma = leaving.sigma;

            /* ---- 4c. rho = B^{-T} e_leaving ---- */
            int32_t rho_nnz = lu_btran_sparse(lu, leaving_pos, rho, rho_pat);

            /* ---- 4d. pivot row by CSR scatter over rho's support ---- */
            int32_t alpha_nnz = 0;
            for (int32_t ri = 0; ri < rho_nnz; ri++) {
                int32_t row = rho_pat[ri];
                double rv = rho[row];
                const Py_ssize_t p_end = self->indptr[row + 1];
                for (Py_ssize_t p = self->indptr[row]; p < p_end; p++) {
                    int32_t col = csr_idx32[p];
                    if (!alpha_touched[col]) {
                        alpha_touched[col] = 1;
                        alpha_pattern[alpha_nnz++] = col;
                    }
                    alpha_scratch[col] += rv * scaled_csr[p];
                }
            }
            /* Logical column n+i is the unit vector e_i, so its pivot-row
             * entry is exactly rho[i]. */
            for (int32_t ri = 0; ri < rho_nnz; ri++) {
                int32_t row = rho_pat[ri];
                int32_t art = n + row;
                alpha_scratch[art] = rho[row];
                alpha_touched[art] = 1;
                alpha_pattern[alpha_nnz++] = art;
            }

            /* ---- 4d'. CHUZC ---- */
            ds2_ratio_prepare(ratio_state, leaving.violation,
                              iters_since_refac);
            DS2Entering entering = ds2_chuzc_core(
                alpha_scratch, alpha_pattern, alpha_nnz, r_ext, bound_status,
                lo_ext, hi_ext, leaving_sigma, DS2_DUAL_TOL, ratio_state);

            if (entering.entering < 0) {
                /* Empty ratio test: the dual is unbounded, which certifies the
                 * primal is infeasible.  With no artificial bounds anywhere in
                 * DS2 this is a genuine certificate, not a maybe. */
                for (int32_t ki = 0; ki < alpha_nnz; ki++) {
                    alpha_scratch[alpha_pattern[ki]] = 0.0;
                    alpha_touched[alpha_pattern[ki]] = 0;
                }
                for (int32_t ri = 0; ri < rho_nnz; ri++) rho[rho_pat[ri]] = 0.0;
                status = (ds_phase == 1) ? "numerical_error" : "infeasible";
                iterations = iter;
                break;
            }

            const int32_t entering_col = entering.entering;

            /* ---- 4d''. bound flips requested by the ratio test ---- */
            if (entering.n_flip > 0 && entering.flip_cols != NULL) {
                memset(flip_rhs, 0, (size_t)m * sizeof(double));
                for (int32_t fi = 0; fi < entering.n_flip; fi++) {
                    int32_t j = entering.flip_cols[fi];
                    double delta;
                    if (bound_status[j] == DS2_AT_LO) {
                        delta = hi_ext[j] - x_ext[j];
                        x_ext[j] = hi_ext[j];
                        bound_status[j] = DS2_AT_HI;
                    } else {
                        delta = lo_ext[j] - x_ext[j];
                        x_ext[j] = lo_ext[j];
                        bound_status[j] = DS2_AT_LO;
                    }
                    if (j < n) {
                        for (Py_ssize_t p = self->csc_indptr[j];
                             p < self->csc_indptr[j + 1]; p++)
                            flip_rhs[(int32_t)self->csc_rows[p]] += a_data[p] * delta;
                    } else {
                        flip_rhs[j - n] += delta;
                    }
                }
                stat_flips += entering.n_flip;
                lu_ftran(lu, flip_rhs, e_i);
                for (int32_t k = 0; k < m; k++) {
                    x_B[k] -= e_i[k];
                    e_i[k] = 0.0;
                }
            }

            /* ---- 4e. alpha_col = B^{-1} a_entering ---- */
            int32_t ftran_nnz;
            {
                if (entering_col < n) {
                    Py_ssize_t col_start = self->csc_indptr[entering_col];
                    Py_ssize_t col_end = self->csc_indptr[entering_col + 1];
                    int32_t col_nnz = (int32_t)(col_end - col_start);
                    int32_t *sp_idx = ftran_pat;
                    double *sp_val = lu->ws_w;
                    for (int32_t k = 0; k < col_nnz; k++) {
                        sp_idx[k] = (int32_t)self->csc_rows[col_start + k];
                        sp_val[k] = a_data[col_start + k];
                    }
                    ftran_nnz = lu_ftran_sparse(lu, col_nnz, sp_idx, sp_val,
                                                alpha_col, ftran_pat);
                } else {
                    int32_t sp_idx[1];
                    double sp_val[1];
                    sp_idx[0] = entering_col - n;
                    sp_val[0] = 1.0;
                    ftran_nnz = lu_ftran_sparse(lu, 1, sp_idx, sp_val,
                                                alpha_col, ftran_pat);
                }
            }

            double pivot = alpha_col[leaving_pos];
            if (fabs(pivot) < DS2_PIVOT_MIN) {
                /* The pivot row said this column was usable and the FTRAN
                 * disagrees; the two disagree only when the basis
                 * representation has degraded.  Ban the row (the CHUZR
                 * contract forbids re-selecting it), leave the basis alone,
                 * and let the next refactorization lift the ban. */
                stat_banned_rows++;
                weights[leaving_pos] = DS2_WEIGHT_BANNED;
                n_banned++;
                for (int32_t ki = 0; ki < ftran_nnz; ki++)
                    alpha_col[ftran_pat[ki]] = 0.0;
                for (int32_t ri = 0; ri < rho_nnz; ri++) rho[rho_pat[ri]] = 0.0;
                for (int32_t ki = 0; ki < alpha_nnz; ki++) {
                    alpha_scratch[alpha_pattern[ki]] = 0.0;
                    alpha_touched[alpha_pattern[ki]] = 0;
                }
                continue;
            }

            /* ---- 4f. primal step ----
             * theta_dual is r_entering / alpha_row[entering] by the interface
             * contract, so the reduced-cost update below zeroes r_entering. */
            const double theta_dual = entering.theta_dual;
            double bound_leaving = (leaving_sigma == 1)
                                       ? lo_ext[basis[leaving_pos]]
                                       : hi_ext[basis[leaving_pos]];
            double dx_entering = (x_B[leaving_pos] - bound_leaving) / pivot;

            for (int32_t ki = 0; ki < ftran_nnz; ki++) {
                int32_t k = ftran_pat[ki];
                x_B[k] -= alpha_col[k] * dx_entering;
            }
            double entering_old_x = x_ext[entering_col];
            double entering_new_x = entering_old_x + dx_entering;

            /* ---- 4g. reduced-cost update over the pivot row's support ---- */
            for (int32_t ki = 0; ki < alpha_nnz; ki++) {
                int32_t j = alpha_pattern[ki];
                double alpha_j = alpha_scratch[j];
                alpha_scratch[j] = 0.0;
                alpha_touched[j] = 0;
                if (basis_pos[j] >= 0) continue;
                if (j == entering_col) continue;
                /* FIXED columns are updated too.  The shipped solver can skip
                 * them because a fixed column stays fixed for the whole solve
                 * and its reduced cost is never read again -- but under the
                 * phase-1 bound map EVERY boxed column is temporarily fixed at
                 * [0,0], and its reduced cost is exactly what decides which
                 * side it is placed on when the true bounds come back.
                 * Skipping them here hands phase 2 a stale r. */
                if (alpha_j != 0.0) r_ext[j] -= theta_dual * alpha_j;
            }
            double r_leaving = -theta_dual;
            r_ext[entering_col] = 0.0;

            /* ---- 4h. pricing weights, BEFORE the basis changes ---- */
            la.ctx = lu;
            ds2_pricing_update(pricing_state, leaving_pos, entering_col,
                               rho, rho_pat, rho_nnz,
                               alpha_col, ftran_pat, ftran_nnz,
                               pivot, weights, m, &la);

            /* ---- 4i. basis bookkeeping ---- */
            int32_t leaving_col = basis[leaving_pos];
            if (leaving_sigma == 1) {
                bound_status[leaving_col] = DS2_AT_LO;
                x_ext[leaving_col] = lo_ext[leaving_col];
            } else {
                bound_status[leaving_col] = DS2_AT_HI;
                x_ext[leaving_col] = hi_ext[leaving_col];
            }
            r_ext[leaving_col] = r_leaving;
            basis_pos[leaving_col] = -1;

            basis[leaving_pos] = entering_col;
            basis_pos[entering_col] = leaving_pos;
            bound_status[entering_col] = DS2_BASIC;
            x_ext[entering_col] = entering_new_x;
            x_B[leaving_pos] = entering_new_x;
            if (fabs(entering_old_x) > 1e6 * (1.0 + fabs(entering_new_x))) {
                /* Catastrophic cancellation in the incremental value: rebuild
                 * the primal from the factorization instead of trusting it. */
                x_B_needs_recompute = 1;
            }
            x_B_fresh = 0;

            /* ---- 4j. LU update and refactorization cadence ---- */
            {
                int need_refac = 0;
                int rc = lu_update_with_ftran_sparse(lu, leaving_pos, alpha_col,
                                                     ftran_nnz, ftran_pat);
                if (rc != 0) {
                    need_refac = 1;
                } else {
                    iters_since_refac++;
                    double ap = fabs(pivot);
                    if (ap < DS2_PIVOT_VIOLENT_LO || ap > DS2_PIVOT_VIOLENT_HI)
                        need_refac = 1;
                }
                if (!need_refac &&
                    (lu_should_refactor(lu) || iters_since_refac >= refac_interval)) {
                    need_refac = 1;
                }

                /* Clear the sparse workspaces before any refactorization frees
                 * the context they belong to. */
                for (int32_t ki = 0; ki < ftran_nnz; ki++)
                    alpha_col[ftran_pat[ki]] = 0.0;
                for (int32_t ri = 0; ri < rho_nnz; ri++) rho[rho_pat[ri]] = 0.0;

                if (need_refac) {
                    lu_context_free(lu);
                    lu = ds_factorize_basis(m, n, self->csc_indptr,
                                            self->csc_rows, a_data, basis,
                                            b_indptr, b_indices, b_values);
                    if (lu != NULL && lu->singular_step >= 0) {
                        lu = ds_repair_singular_basis(
                            lu, m, n, self->csc_indptr, self->csc_rows, a_data,
                            basis, basis_pos, bound_status, x_ext, r_ext,
                            lo_ext, hi_ext, b_indptr, b_indices, b_values, 10);
                    }
                    if (lu == NULL || lu->singular_step >= 0) {
                        status = "numerical_error";
                        iterations = iter + 1;
                        break;
                    }
                    lu_build_transposes(lu);
                    total_refacs++;
                    iters_since_refac = 0;

                    /* Drift repair.  Recompute the duals from scratch, then
                     * restore dual feasibility: flip a nonbasic whose sign has
                     * drifted when it has an opposite bound to flip to, and
                     * otherwise shift its cost just inside the feasible side.
                     * Every shift is recorded, and the clean-state audit
                     * strips all of them before anything is certified. */
                    for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
                    lu_btran(lu, c_B, y);
                    memset(lu->ws_v, 0, (size_t)m * sizeof(double));
                    for (int32_t j = 0; j < n_total; j++) {
                        if (basis_pos[j] >= 0) continue;
                        double rj = c_ext[j];
                        if (j < n) {
                            for (Py_ssize_t p = self->csc_indptr[j];
                                 p < self->csc_indptr[j + 1]; p++)
                                rj -= a_data[p] * y[(int32_t)self->csc_rows[p]];
                        } else {
                            rj -= y[j - n];
                        }
                        int8_t bs = bound_status[j];
                        if (bs == DS2_FIXED) {
                            /* Cannot be dual infeasible; only its value had to
                             * be refreshed for the phase boundary. */
                        } else if (bs == DS2_AT_LO && rj < -DS2_DRIFT_TOL &&
                                   isfinite(hi_ext[j])) {
                            bound_status[j] = DS2_AT_HI; x_ext[j] = hi_ext[j];
                        } else if (bs == DS2_AT_HI && rj > DS2_DRIFT_TOL &&
                                   isfinite(lo_ext[j])) {
                            bound_status[j] = DS2_AT_LO; x_ext[j] = lo_ext[j];
                        } else if ((bs == DS2_AT_LO && rj < -DS2_DRIFT_TOL) ||
                                   (bs == DS2_AT_HI && rj > DS2_DRIFT_TOL) ||
                                   (bs == DS2_FREE && fabs(rj) > DS2_DRIFT_TOL)) {
                            double target;
                            if (bs == DS2_FREE) {
                                target = 0.0;
                            } else {
                                double magnitude = DS2_DRIFT_TOL;
                                if (opt_perturb) {
                                    magnitude *=
                                        1.0 + ds2_perturb_fraction(&perturb_rng);
                                }
                                target =
                                    (bs == DS2_AT_LO) ? magnitude : -magnitude;
                            }
                            double shift = target - rj;
                            c_ext[j] += shift;
                            c_shift[j] += shift;
                            stat_cost_shifts++;
                            rj = target;
                        }
                        r_ext[j] = rj;
                    }
                    n_banned = 0;
                    for (int32_t k = 0; k < m; k++) weights[k] = 1.0;
                    la.ctx = lu;
                    ds2_pricing_reset(pricing_state, weights, m, 0, &la);
                    x_B_needs_recompute = 1;
                    x_B_fresh = 0;
                }
            }

            if (fabs(theta_dual) < 1e-12) stat_degenerate++;
            /* A completed pivot is progress, so the ban-recovery budget
             * bounds CONSECUTIVE failures to make any, not the number of
             * recoveries a long trajectory happens to need. */
            ban_rounds = 0;
            iterations = iter + 1;
        }

        if (iterations >= max_iter && strcmp(status, "optimal") != 0 &&
            strcmp(status, "infeasible") != 0) {
            status = "iteration_limit";
        }
        if (opt_report) {
            fprintf(stderr,
                    "[ds2] total = %lld pivots (status %s), refacs %lld, "
                    "degenerate %lld, banned rows %lld, flips %lld, "
                    "cost shifts %lld\n",
                    (long long)iterations, status, (long long)total_refacs,
                    (long long)stat_degenerate, (long long)stat_banned_rows,
                    (long long)stat_flips, (long long)stat_cost_shifts);
            fflush(stderr);
        }
    }

    /* ---- 5. exit: rebuild from scratch and certify ------------------------ *
     * Nothing here trusts the incremental state.  The basis is refactorized,
     * x_B and y are recomputed, every cost shift is removed, and dual
     * feasibility is verified in ORIGINAL units against the TRUE bounds.
     * Because DS2 has no artificial bounds, the shipped path's gap-damage
     * budget and big-M retry have no analogue -- the test below is the whole
     * certificate. */
    if (strcmp(status, "optimal") == 0) {
        lu_context_free(lu);
        lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows, a_data,
                                basis, b_indptr, b_indices, b_values);
        if (lu != NULL && lu->singular_step >= 0) {
            lu = ds_repair_singular_basis(
                lu, m, n, self->csc_indptr, self->csc_rows, a_data, basis,
                basis_pos, bound_status, x_ext, r_ext, lo_ext, hi_ext,
                b_indptr, b_indices, b_values, 10);
        }
        if (lu == NULL || lu->singular_step >= 0) {
            status = "numerical_error";
        } else {
            memset(rhs, 0, (size_t)m * sizeof(double));
            for (int32_t j = 0; j < n_total; j++) {
                if (basis_pos[j] >= 0) continue;
                double xj = x_ext[j];
                if (xj == 0.0) continue;
                if (j < n) {
                    for (Py_ssize_t p = self->csc_indptr[j];
                         p < self->csc_indptr[j + 1]; p++)
                        rhs[(int32_t)self->csc_rows[p]] -= a_data[p] * xj;
                } else {
                    rhs[j - n] -= xj;
                }
            }
            lu_ftran(lu, rhs, x_B);
            for (int32_t k = 0; k < m; k++)
                if (basis[k] < n) x_ext[basis[k]] = x_B[k];

            for (int32_t j = 0; j < n_total; j++) {
                if (c_shift[j] != 0.0) { c_ext[j] -= c_shift[j]; c_shift[j] = 0.0; }
            }
            for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
            lu_btran(lu, c_B, y);
            memset(lu->ws_v, 0, (size_t)m * sizeof(double));

            /* Primal feasibility of the basics against the TRUE bounds.  Under
             * the logical form a basic logical must sit at its own bound -b_i,
             * which is exactly the statement that row i of Ax = b holds. */
            for (int32_t k = 0; k < m; k++) {
                int32_t j = basis[k];
                double lo = lo_true[j], hi = hi_true[j];
                if ((isfinite(lo) && x_B[k] < lo - 1e-6 * (1.0 + fabs(lo))) ||
                    (isfinite(hi) && x_B[k] > hi + 1e-6 * (1.0 + fabs(hi)))) {
                    status = "infeasible";
                    break;
                }
            }

            /* Dual feasibility in ORIGINAL units against the TRUE bounds. */
            if (strcmp(status, "optimal") == 0) {
                for (int32_t j = 0; j < n; j++) {
                    if (basis_pos[j] >= 0) continue;
                    double cj = c_orig[j];
                    double rj = cj;
                    for (Py_ssize_t p = self->csc_indptr[j];
                         p < self->csc_indptr[j + 1]; p++) {
                        int32_t row = (int32_t)self->csc_rows[p];
                        rj -= self->csc_data[p] * (row_scale[row] * y[row]);
                    }
                    double dtol = DS2_CERT_TOL * (1.0 + fabs(cj));
                    int8_t bs = bound_status[j];
                    if (bs == DS2_AT_LO && rj < -dtol) {
                        status = "dual_infeasible"; break;
                    }
                    if (bs == DS2_AT_HI && rj > dtol) {
                        status = "dual_infeasible"; break;
                    }
                    if (bs == DS2_FREE && fabs(rj) > dtol) {
                        status = "dual_infeasible"; break;
                    }
                    /* A nonbasic sitting at a bound that does not exist in the
                     * true problem can only be optimal at zero reduced cost. */
                    if (bs == DS2_AT_LO && !isfinite(lo_true[j]) &&
                        fabs(rj) > dtol) {
                        status = "dual_infeasible"; break;
                    }
                    if (bs == DS2_AT_HI && !isfinite(hi_true[j]) &&
                        fabs(rj) > dtol) {
                        status = "dual_infeasible"; break;
                    }
                }
            }
        }
    } else {
        for (int32_t k = 0; k < m; k++)
            if (basis[k] < n) x_ext[basis[k]] = x_B[k];
    }

build_result:
    if (ruiz_active) {
        for (int32_t j = 0; j < n; j++) x_ext[j] *= col_scale[j];
        for (int32_t i = 0; i < m; i++) b[i] /= row_scale[i];
    }
    {
        double objective = 0.0;
        double max_residual = 0.0;
        for (int32_t j = 0; j < n; j++) objective += c_orig[j] * x_ext[j];
        for (int32_t i = 0; i < m; i++) {
            double row_sum = 0.0;
            for (Py_ssize_t p = self->indptr[i]; p < self->indptr[i + 1]; p++)
                row_sum += self->data[p] * x_ext[csr_idx32[p]];
            double res = fabs(row_sum - b[i]);
            if (res > max_residual) max_residual = res;
        }

        PyObject *x_list = PyList_New(n_s);
        PyObject *y_list = PyList_New(m_s);
        if (x_list == NULL || y_list == NULL) {
            Py_XDECREF(x_list);
            Py_XDECREF(y_list);
            goto done;
        }
        for (int32_t j = 0; j < n; j++)
            PyList_SET_ITEM(x_list, j, PyFloat_FromDouble(x_ext[j]));
        for (int32_t i = 0; i < m; i++)
            PyList_SET_ITEM(y_list, i,
                            PyFloat_FromDouble(ruiz_active ? row_scale[i] * y[i]
                                                           : y[i]));

        result = Py_BuildValue(
            "{s:s,s:d,s:d,s:n,s:N,s:N,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:d}",
            "status", status,
            "objective", objective,
            "max_primal_residual", max_residual,
            "iterations", iterations,
            "x", x_list,
            "y", y_list,
            "refactorizations", (long long)total_refacs,
            "bound_flips", (long long)stat_flips,
            "degenerate_pivots", (long long)stat_degenerate,
            "banned_rows", (long long)stat_banned_rows,
            "cost_shifts", (long long)stat_cost_shifts,
            "audit_rounds", (long long)stat_audit_rounds,
            "phase1_iterations", (long long)phase1_iters,
            "phase1_dual_objective", phase1_dual_obj);
    }

done:
    lu_context_free(lu);
    if (states_created) {
        ds2_pricing_state_free(pricing_state);
        ds2_ratio_state_free(ratio_state);
    }
    free(c_ext); free(c_orig); free(c_shift);
    free(lo_ext); free(hi_ext); free(lo_true); free(hi_true);
    free(x_ext); free(r_ext); free(bound_status); free(basis_pos);
    free(b); free(y); free(x_B); free(rhs); free(rho); free(alpha_col);
    free(e_i); free(c_B); free(weights); free(flip_rhs);
    free(basis); free(rho_pat); free(ftran_pat);
    free(alpha_scratch); free(alpha_pattern); free(alpha_touched);
    free(b_indptr); free(b_indices); free(b_values);
    free(row_scale); free(col_scale); free(scaled_csc); free(scaled_csr);
    free(csr_idx32);
    return result;
}

/* ---- Python entry point -------------------------------------------------- */

static PyObject *CSRMatrix_solve_eq_box_ds2(
    CSRMatrixObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *c_obj, *b_obj, *lo_obj, *hi_obj;
    Py_ssize_t max_iter_arg = 0;
    double tol = 1e-8;
    static char *kwlist[] = {"c", "b", "lo", "hi", "max_iter", "tol", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO|nd", kwlist,
                                     &c_obj, &b_obj, &lo_obj, &hi_obj,
                                     &max_iter_arg, &tol)) {
        return NULL;
    }
    return ds2_solve(self, c_obj, b_obj, lo_obj, hi_obj, max_iter_arg, tol);
}
