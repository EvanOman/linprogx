# DS2 — dual simplex rewrite: shared contract

**PROVENANCE: SOURCE-INFORMED (HiGHS).** This rewrite is informed by reading the
HiGHS implementation under an explicit owner authorisation covering greenbea.
Read `docs/PROVENANCE.md` before making any public claim. **No verbatim copying** —
understand the algorithm, reimplement independently.

## Why a rewrite (the evidence)

Measured across ~30 LPnetlib instances (`experiments/category_verdict_2026_07_26.md`):

| route | cells | iteration ratio vs HiGHS |
|---|---|---|
| **dual simplex** | 5 | **0.79–2.74x** |
| IPM | ~25 | 0.00–0.35x |

linprogx's **IPM is exceptional**; its **dual simplex is 1.5–2.7x behind HiGHS on
trajectory** across a whole structural class. Every individual mechanism
transplanted from HiGHS — Phase-1 bound swap, BFRT, DSE, weight floor,
re-selection, tie-breaks, row-cost perturbation, logical-basis start — was
implemented and measured, and **none is a net wall win**. With HiGHS's *exact*
configuration linprogx still needs 4,334 pivots to HiGHS's 2,836, with the excess
split evenly (1.57x Phase 1, 1.50x Phase 2).

**Conclusion: the deficit is the coupled quality of the whole pivot-selection
regime, not a missing component.** Hence DS2.

## Ground truth to beat

| instance | linprogx now | HiGHS | target |
|---|---:|---:|---|
| greenbea | 4,399 (4,283 with shipped churn) | 2,836 | **-35.5% pivots**; the cell is 1.527 and its loss is entirely pivot count |
| greenbeb | 8,919 | 4,910 | same |
| 25fv47 | 8,300 | 3,033 | same |
| degen2 | 1,447 | 537 | same |
| sierra | 725 | 914 | do not regress (we already win) |

**CORRECTED 2026-07-26 -- the previous version of this paragraph was wrong.**
It claimed linprogx's per-pivot cost is "1.73x better than HiGHS's (85.7us vs
148.4us)" and told you to protect that advantage. There is no such advantage. A
direct v3 paired measurement putting both solvers on the same clean Modal hosts
(`assets/modal_bench_af6bd89823fd_paired_hosts3.json`, loadavg 0.00) gives:

| host | lx us/pivot | hx us/iter | lx per-pivot advantage |
|---|---:|---:|---:|
| 0 | 125.4 | 130.6 | 1.041x |
| 1 | 136.9 | 137.6 | 1.005x |
| 2 | 92.9 | 94.4 | 1.016x |

**Per-pivot cost is at PARITY.** The 1.73x compared linprogx measured on a quiet
local box against HiGHS measured in a different context.

Two consequences you must reason with:

1. **greenbea's board ratio is ~1.527, not 1.156** (0/21 pairs, hosts
   [1.4897, 1.5428]). The 1.215/1.156 record was a favourable host draw. envab
   results are still valid -- both arms share a container -- but never compose an
   envab gain onto a paired ratio from an older host draw.
2. **pivot ratio 4,399/2,836 = 1.551 and wall ratio 1.527 AGREE**, so greenbea's
   loss is *entirely* its pivot-count deficit. **The target is -35.5% pivots.**
   The old claim that closing the pivot gap is "worth -35.5% against a 13.46%
   bar, a 2.6x margin" is VOID: there is no margin, they are the same quantity.

Pivot-count work and per-pivot work therefore pay at the **same** rate (both are
~99.5% of the cell). Neither is privileged; choose by expected effect size.

## Architecture

DS2 is a **new implementation alongside the existing one**, selected by
`LINPROGX_DS2=1`, default OFF. The shipped dual simplex is not modified. This
lets DS2 be developed, compared and abandoned without risk.

Existing entry point (do not change its behaviour):
`CSRMatrix_solve_eq_box_dual_simplex` in `src/linprogx/_csparse.c`.

## Component interfaces (fixed, so components compose)

Three components, developed independently. **Agree to these signatures.**

```c
/* ---- CHUZR: choose the leaving row ------------------------------------ */
typedef struct {
    int32_t basis_pos;   /* leaving basis position, -1 if none (optimal)   */
    int     sigma;       /* +1 if x_B below lower, -1 if above upper       */
    double  violation;   /* the bound violation that selected it           */
} DS2Leaving;

DS2Leaving ds2_chuzr(
    const int32_t *basis, const double *x_B,
    const double *lo_ext, const double *hi_ext,
    double *weights,              /* edge weights, updated in place        */
    int32_t m, double feas_tol,
    void *pricing_state);         /* component B owns this                 */

/* ---- CHUZC: choose the entering column + bound flips ------------------ */
typedef struct {
    int32_t entering;        /* entering column, -1 if dual unbounded      */
    double  theta_dual;      /* dual step                                  */
    double  alpha_pivot;     /* pivot element                              */
    int32_t n_flip;          /* number of bound flips to apply             */
    const int32_t *flip_cols;/* columns to flip (owned by callee)          */
} DS2Entering;

DS2Entering ds2_chuzc(
    const double *alpha_row,      /* pivot row over nonbasic columns       */
    const int32_t *alpha_pattern, int32_t alpha_nnz,
    const double *r_ext, const int8_t *bound_status,
    const double *lo_ext, const double *hi_ext,
    int leaving_sigma, double dual_tol,
    void *ratio_state);           /* component A owns this                 */
```

`ds2_core` owns the loop, the basis, the LU, the phase bounds and the updates,
and calls the two above. Components A and B must be **self-contained** and
**independently testable** against the existing solver on real instances.

## Non-negotiable constraints

- **eps = 2e-5** fixed; certificate-backed optimality only; every accepted answer
  re-certifies in ORIGINAL units.
- **No per-problem tuning.** Global mechanisms and thresholds only.
- **No verbatim copying** from HiGHS. Both projects are MIT so it would be legal;
  we are choosing understand-and-reimplement so the claim is "we built our own".
- **Do not modify the existing dual simplex.** DS2 is additive and gated.
- The **trace-hash oracle** (`LINPROGX_DS_TRACE_HASH=1`) folds the raw bits of
  every BTRAN/FTRAN output into an FNV-1a digest — use it to prove the shipped
  path is untouched (greenbea baseline digest `679168a4baad36d6`, 4,399 pivots).

## Measurement discipline (hard-won; do not relearn)

- This box runs at high load. **Cross-process wall comparisons drift 4–19%** and
  are unusable. Use `experiments/harris_alternating_ab.py` — alternating
  within-process A/B with untouched control phases.
- **Iteration counts are load-invariant.** Prefer them while developing;
  they are the actual target anyway.
- Six idealised models (element counts, ceilings, "coupled system") have already
  failed to predict cycles this campaign. **Fund on measurement, not projection.**

## Reference material in-repo

- `experiments/highs_study_S1_dual_phase1_2026_07_25.md` — Phase 1 mechanism
- `experiments/highs_study_S2_basis_and_start_2026_07_25.md` — basis/start
- `experiments/highs_study_S3_per_pivot_machinery_2026_07_25.md` — CHUZR/CHUZC,
  with file:line citations into HiGHS
- `experiments/category_verdict_2026_07_26.md` — why the rewrite is justified
- HiGHS source (read-only): `/tmp/highs-study/HiGHS/highs/simplex/`
