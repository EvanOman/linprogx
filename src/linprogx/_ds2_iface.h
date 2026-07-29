/*
 * _ds2_iface.h -- DS2 shared component contract.
 *
 * PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 * See docs/PROVENANCE.md.  The DS2 architecture (logical form, in-place
 * dual Phase 1 by bound substitution, logical-basis start) was derived from
 * reading the HiGHS implementation under explicit owner authorisation.
 * No verbatim copying: this is an independent reimplementation.
 *
 * DS2 is split into three independently developed components:
 *
 *   ds2_core   -- the iteration loop, basis/LU management, phase handling,
 *                 the primal/dual updates and the exit/certification path.
 *                 It CALLS the two entry points below.
 *   component A -- ds2_chuzc: the dual ratio test (entering column + flips).
 *   component B -- ds2_chuzr: dual pricing (leaving row + edge weights).
 *
 * The two signatures below are FIXED by docs/DS2-REWRITE.md and must not
 * change.  Everything else in this header is a core-provided extension
 * that exists because a component needs state or per-pivot notification
 * that the fixed signatures cannot carry; each has a documented default.
 *
 * Integration: _csparse.c includes this header, then includes an
 * implementation for each component -- the real one if the file exists,
 * otherwise the stub shipped alongside ds2_core:
 *
 *     _ds2_chuzr.c  (component B)   else  _ds2_stub_chuzr.c
 *     _ds2_chuzc.c  (component A)   else  _ds2_stub_chuzc.c
 *
 * so a component can be dropped in by adding its file, with no edit to
 * ds2_core or to _csparse.c.
 */

#ifndef LINPROGX_DS2_IFACE_H
#define LINPROGX_DS2_IFACE_H

#include <stdint.h>

/* ---- Nonbasic bound status (shared vocabulary) ------------------------ */
#define DS2_AT_LO   0   /* nonbasic at its lower bound                     */
#define DS2_AT_HI   1   /* nonbasic at its upper bound                     */
#define DS2_FREE    2   /* nonbasic free variable, value 0                 */
#define DS2_FIXED   3   /* lo == hi                                        */
#define DS2_BASIC   4   /* in the basis (skip in ds2_chuzc)                */

/* ---- Row ban protocol ------------------------------------------------
 * The core has one numerical escape it cannot express through the fixed
 * CHUZR signature: after a leaving row turns out to yield an unusably
 * small pivot, that row must not be re-selected on the retry.  The core
 * signals this by writing DS2_WEIGHT_BANNED into weights[p].
 *
 * CONTRACT: ds2_chuzr must never return a basis position whose weight is
 * >= DS2_WEIGHT_BANNED.  Any pricing rule that divides a merit by the
 * weight satisfies this automatically; a rule that ignores weights (e.g.
 * plain Dantzig) must test explicitly.  The core resets all weights at
 * every refactorization, so a ban lasts at most one refactorization
 * interval. */
#define DS2_WEIGHT_BANNED 1e30

/* ---- CHUZR: choose the leaving row (component B) ---------------------- */
typedef struct {
    int32_t basis_pos;   /* leaving basis position, -1 if none (optimal)   */
    int     sigma;       /* +1 if x_B below lower, -1 if above upper       */
    double  violation;   /* the bound violation that selected it           */
} DS2Leaving;

/*
 * basis   : basis[k] = column index occupying basis position k
 * x_B     : x_B[k]   = primal value of basis[k]
 * lo_ext,
 * hi_ext  : CURRENT working bounds, indexed by column.  In Phase 1 these
 *           are the substituted boxes, in Phase 2 the true bounds; CHUZR
 *           is phase-agnostic and must not care which.
 * weights : edge weights, one per basis POSITION, updated in place.
 * feas_tol: a basic variable counts as infeasible only beyond this.
 *
 * Returns basis_pos = -1 when every basic variable is within feas_tol of
 * its box -- which the core reads as "optimal for the current phase".
 */
DS2Leaving ds2_chuzr(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    double *weights,
    int32_t m, double feas_tol,
    void *pricing_state);

/* ---- CHUZC: choose the entering column + bound flips (component A) ---- */
typedef struct {
    int32_t entering;        /* entering column, -1 if dual unbounded      */
    double  theta_dual;      /* dual step                                  */
    double  alpha_pivot;     /* pivot element                              */
    int32_t n_flip;          /* number of bound flips to apply             */
    const int32_t *flip_cols;/* columns to flip (owned by callee)          */
} DS2Entering;

/*
 * alpha_row     : DENSE scatter of the pivot row, indexed by column.
 *                 ONLY the entries listed in alpha_pattern are valid --
 *                 every other entry is stale from an earlier pivot.
 * alpha_pattern,
 * alpha_nnz     : the support of the pivot row (structural and logical
 *                 columns alike).  It includes BASIC columns; the callee
 *                 must skip any j with bound_status[j] == DS2_BASIC.
 * r_ext         : reduced costs, indexed by column.
 * bound_status  : DS2_AT_LO / DS2_AT_HI / DS2_FREE / DS2_FIXED / DS2_BASIC.
 * lo_ext,hi_ext : current working bounds (see CHUZR).
 * leaving_sigma : +1 if the leaving variable was below its lower bound,
 *                 -1 if above its upper bound.
 * dual_tol      : dual feasibility tolerance.
 *
 * Sign convention the core relies on, for sigma in {+1,-1}:
 *
 *     ratio_j = -r_j / (sigma * alpha_row[j])   >= 0 for eligible j
 *     eligible: at LO requires sigma*alpha_row[j] < 0
 *               at HI requires sigma*alpha_row[j] > 0
 *               FREE requires alpha_row[j] != 0
 *     theta_dual = r_entering / alpha_row[entering]
 *
 * After the pivot the core applies r_j -= theta_dual * alpha_row[j] over
 * the pattern, which zeroes r_entering.  A returned theta_dual that does
 * not satisfy that identity will silently corrupt the reduced costs.
 *
 * entering = -1 means no eligible column exists: the dual is unbounded,
 * hence the primal is infeasible.
 *
 * flip_cols must stay valid until the next ds2_chuzc call; the core copies
 * nothing and applies the flips immediately.
 */
DS2Entering ds2_chuzc(
    const double *alpha_row,
    const int32_t *alpha_pattern, int32_t alpha_nnz,
    const double *r_ext, const int8_t *bound_status,
    const double *lo_ext, const double *hi_ext,
    int leaving_sigma, double dual_tol,
    void *ratio_state);

/* ---- Component lifecycle and notification (core extensions) ----------
 * The fixed signatures take an opaque state pointer but say nothing about
 * where it comes from, and component B cannot maintain steepest-edge
 * weights without seeing the pivot it was not asked to choose.  These six
 * hooks close both gaps.  Every one has a valid no-op default, so a
 * component that needs none of them can ignore them entirely. */

void *ds2_pricing_state_new(int32_t m, int32_t n_total);
void  ds2_pricing_state_free(void *state);
void *ds2_ratio_state_new(int32_t m, int32_t n_total);
void  ds2_ratio_state_free(void *state);

/*
 * Called once per accepted pivot, AFTER the pivot row and entering column
 * are known and BEFORE the basis changes.
 *   leaving_pos : basis position being vacated
 *   entering    : entering column index
 *   rho         : B^{-T} e_leaving_pos, with support rho_pattern/rho_nnz
 *   alpha_col   : B^{-1} a_entering, with support ftran_pattern/ftran_nnz
 *   alpha_pivot : alpha_col[leaving_pos]
 *   weights     : edge weights, to be updated in place
 */
void ds2_pricing_update(
    void *state,
    int32_t leaving_pos, int32_t entering,
    const double *rho, const int32_t *rho_pattern, int32_t rho_nnz,
    const double *alpha_col, const int32_t *ftran_pattern, int32_t ftran_nnz,
    double alpha_pivot, double *weights, int32_t m);

/*
 * Called after every (re)factorization, with the basis freshly inverted.
 * The core has already reset every weight to 1.0; a rule that wants exact
 * weights on a non-identity basis computes them here.  `logical_basis` is
 * nonzero when B == I, for which unit weights are already exact.
 */
void ds2_pricing_reset(void *state, double *weights, int32_t m,
                       int logical_basis);

#endif /* LINPROGX_DS2_IFACE_H */
