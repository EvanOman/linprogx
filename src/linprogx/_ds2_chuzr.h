/* DS2 CHUZR -- dual simplex leaving-row selection with dual steepest edge.
 *
 * PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.
 *
 * This component was written after reading HiGHS's HEkkDualRHS under the
 * 2026-07-25 owner authorisation recorded in docs/PROVENANCE.md. No HiGHS
 * code was copied; the data structure (a maintained array of per-row primal
 * infeasibilities plus a candidate list with a merit cutoff, rebuilt on
 * demand, with a dense-scan fallback when too many rows are infeasible) was
 * understood and reimplemented independently.
 *
 * Interface contract: docs/DS2-REWRITE.md. `ds2_chuzr` matches the fixed
 * signature declared there exactly. Everything else in this header is state
 * management that the DS2 core calls around it.
 *
 * The component is self-contained C99: no Python.h, no linprogx headers, no
 * dependency on the shipped dual simplex. It can be compiled and tested on
 * its own (see experiments/ds2_chuzr_validate.py).
 *
 * WHY IT LOOKS LIKE THIS
 * ----------------------
 * The shipped linprogx CHUZR (_csparse.c 14469-14607) rescans all m basis
 * positions every pivot, and for each one gathers lo_ext[basis[k]] and
 * hi_ext[basis[k]] -- two random accesses into n_total-sized arrays -- before
 * it can even tell whether the row is infeasible. This component maintains
 * the violation of every row in one dense array, updates only the rows a
 * pivot actually touched, and scans a candidate list instead of all m rows
 * whenever the infeasible set is small enough for that to pay.
 *
 * THE CONTRACT WITH ds2_core
 * --------------------------
 *   ds2_pricing_new                once per solve
 *   ds2_chuzr_invalidate           after anything this component was not told
 *                                  about: initial basis, refactorisation,
 *                                  phase switch, bound change, x_B recompute
 *   ds2_chuzr_rows_changed         after every pivot, with the basis positions
 *                                  whose x_B moved (the FTRAN pattern of the
 *                                  entering column, which always includes the
 *                                  pivot row)
 *   ds2_chuzr                      to select
 *
 * ds2_chuzr with pricing_state == NULL is the reference implementation: a
 * stateless dense scan with identical selection semantics, so the maintained
 * path can be validated against it row for row.
 */

#ifndef LINPROGX_DS2_CHUZR_H
#define LINPROGX_DS2_CHUZR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- The fixed interface from docs/DS2-REWRITE.md --------------------- */

#ifndef LINPROGX_DS2_IFACE_H
typedef struct {
    int32_t basis_pos;   /* leaving basis position, -1 if none (optimal)   */
    int     sigma;       /* +1 if x_B below lower, -1 if above upper       */
    double  violation;   /* the bound violation that selected it           */
} DS2Leaving;
#define DS2_WEIGHT_BANNED 1e30
#endif

/* Opaque pricing state; component B (this one) owns it. */
typedef struct DS2Pricing DS2Pricing;

/* Selection rules.
 *
 * DANTZIG maximises the raw violation and never reads `weights`; it is what
 * linprogx ships (leaving_rule=1). DSE maximises violation^2 / gamma_i.
 * One code path serves both: for violations >= 0, argmax v == argmax v^2, so
 * DANTZIG is DSE with every weight equal to 1. */
#define DS2_RULE_DANTZIG 0   /* merit = violation^2 (weights ignored)      */
#define DS2_RULE_DSE     1   /* merit = violation^2 / gamma_i              */

/* Create/destroy. `m` is the number of basic variables (rows).
 * Returns NULL on allocation failure. */
DS2Pricing *ds2_pricing_new(int32_t m, int rule);
void        ds2_pricing_free(DS2Pricing *st);

/* Tunables. All global, no per-problem tuning; these exist so the harness
 * can measure each knob's effect, not so a caller can fit an instance. */
void ds2_pricing_set_weight_floor(DS2Pricing *st, double floor);  /* dflt 1e-4 */
void ds2_pricing_set_random_start(DS2Pricing *st, int on);        /* dflt 0    */
void ds2_pricing_set_seed(DS2Pricing *st, uint64_t seed);
void ds2_pricing_set_rule(DS2Pricing *st, int rule);
/* Density of the last FTRAN'd column, as a fraction of m. The merit cutoff is
 * only worth building when the pivot column is genuinely sparse, because that
 * is what keeps the candidate list from being rebuilt every pivot. */
void ds2_pricing_set_column_density(DS2Pricing *st, double density);
/* 0 disables the candidate list (maintained dense scan only); 1 is the
 * default; 2 keeps the list even when most rows are infeasible. */
void ds2_pricing_set_list_mode(DS2Pricing *st, int mode);
#define DS2_LIST_OFF    0
#define DS2_LIST_ON     1
#define DS2_LIST_ALWAYS 2
/* 0 disables the hyper-sparse merit cutoff (list = every infeasible row). */
void ds2_pricing_set_cutoff_enabled(DS2Pricing *st, int on);
/* Paranoid mode: every ds2_chuzr additionally runs the full dense reference
 * scan and counts disagreements. Validation only -- it destroys the cost
 * advantage by construction. */
void ds2_pricing_set_paranoid(DS2Pricing *st, int on);

/* ---- CHUZR: choose the leaving row ------------------------------------ */
/* Exactly the signature fixed by docs/DS2-REWRITE.md.
 *
 * `weights` is the edge-weight array indexed by basis position. It is read
 * here and floored in place, which is the only in-place update CHUZR is
 * responsible for; the Forrest-Goldfarb recurrence belongs to ds2_core, which
 * owns the pivot. Passing NULL is legal and means unit weights.
 *
 * With `pricing_state == NULL` this is a stateless dense scan over all m rows
 * (the reference). With a state, it scans the maintained infeasibility array
 * or the candidate list. Both return the same row; that equality is what
 * experiments/ds2_chuzr_validate.py checks on real trajectories. */
DS2Leaving ds2_chuzr(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    double *weights,              /* edge weights, updated in place        */
    int32_t m, double feas_tol,
    void *pricing_state);         /* component B owns this                 */

/* ---- State maintenance, called by ds2_core ---------------------------- */

/* Declare that x_B changed at the given basis positions -- i.e. the FTRAN
 * pattern of the entering column, plus the pivot row. Recomputes those rows'
 * violations and adds any that became infeasible to the candidate list.
 * O(n_rows). Duplicate entries are harmless. */
void ds2_chuzr_rows_changed(
    DS2Pricing *st,
    const int32_t *rows, int32_t n_rows,
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext, double feas_tol);

/* Declare that the whole of x_B, the basis or the bounds may have changed.
 * Forces a full recompute on the next ds2_chuzr. O(1). */
void ds2_chuzr_invalidate(DS2Pricing *st);

/* Ban a basis position for the remainder of this pivot; used by the DSE
 * weight-acceptance test, which discards a row whose stored weight was far
 * below the exact one and re-selects. Cleared by ds2_chuzr_clear_bans. */
void ds2_chuzr_ban(DS2Pricing *st, int32_t basis_pos);
void ds2_chuzr_clear_bans(DS2Pricing *st);

/* ---- Instrumentation (free when unread) ------------------------------- */
typedef struct {
    long long calls;             /* ds2_chuzr invocations                  */
    long long rebuilds;          /* full m-row rebuilds of the list        */
    long long recomputes;        /* full m-row violation recomputes        */
    long long scanned;           /* candidate entries examined             */
    long long dense_scanned;     /* rows examined in dense-mode scans      */
    long long changed_rows;      /* rows pushed through rows_changed       */
    long long dense_calls;       /* calls served in dense mode             */
    long long cutoff_misses;     /* cutoff list too thin -> rebuilt+rescan */
    long long cutoff_installed;  /* rebuilds that set a nonzero cutoff     */
    long long list_len_sum;      /* sum of candidate-list length per call  */
    long long infeas_sum;        /* sum of #infeasible rows per rebuild    */
    long long paranoid_mismatch; /* disagreements vs the full dense scan   */
} DS2ChuzrStats;

const DS2ChuzrStats *ds2_chuzr_stats(const DS2Pricing *st);
void ds2_chuzr_stats_reset(DS2Pricing *st);
int32_t ds2_chuzr_list_len(const DS2Pricing *st);   /* <0 means dense mode */
int32_t ds2_chuzr_num_infeasible(const DS2Pricing *st);

/* Validation hook: recompute every violation from the caller's arrays and
 * return the number of rows where the maintained value disagrees by more than
 * `tol`. Not on any hot path. */
int32_t ds2_chuzr_audit(
    const DS2Pricing *st,
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    double feas_tol, double tol, int32_t *first_bad_row);

/* Reference dense scan, exposed so the harness can compare selection and cost
 * against the shipped behaviour without linking the shipped solver. This is a
 * faithful transcription of the shipped scan's arithmetic: gather the leaving
 * variable's bounds through basis[], test both sides, score, keep the strict
 * argmax with ascending-index tie-breaking. */
DS2Leaving ds2_chuzr_dense_reference(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    const double *weights, int32_t m, double feas_tol,
    int rule, double weight_floor);

#ifdef __cplusplus
}
#endif

#endif /* LINPROGX_DS2_CHUZR_H */
