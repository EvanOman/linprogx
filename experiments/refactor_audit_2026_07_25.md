# Refactorization audit — efficient; one candidate left (2026-07-25)

**Verdict: the elimination is tight. The only visible slack is setup/teardown,
worth ~3% of wall, and it is a real unexplored candidate.**

`refactor` is 6.75% of wall across 33 refactorizations. Measured with the
repo's existing `LINPROGX_LU_PROFILE=1` instrument (no new instrumentation
needed).

## Per-refactor profile (greenbea, m=1,525)

| sub-phase | seconds | share of refactor |
|---|---:|---:|
| init | 0.0002 | ~22% |
| pivot search (Markowitz) | 0.0001 | ~11% |
| elimination | 0.00035 | ~39% |
| assemble | 0.0002 | ~22% |

Total ~0.7–1.0 ms per refactor × 33 ≈ 30 ms ≈ 8% of the 377 ms wall, consistent
with the 6.75% phase measurement.

## The elimination is already good — not a target

- `nnzlu` runs **6,860 → 7,740** at m=1,525, i.e. **~4.6 nonzeros per column**.
  The fill is excellent; there is no fill-reduction opportunity of the kind that
  would move a 12.8% gap.
- `steps = 1,461` with `cols_scanned` 1,461 → 5,011, so the Markowitz search
  examines only **1.0–3.4 columns per step** against its Suhl candidate budget
  of 8. The search is already tight, not scanning wastefully.
- The code carries a dated note that fusing its two walks was measured as a
  **paired-A/B −11% LOSS (2026-07-12)** — independently consistent with this
  session's five other failed fusion/work-count models.

## The one candidate: per-refactor context allocation

`init` + `assemble` together are **~44% of refactor time ≈ 3% of wall**, and
`ds_factorize_basis` allocates a **fresh `LUContext` for every refactorization** —
`lu_factorize` mallocs and initialises roughly twenty arrays of size m (and
nnz), 33 times over the run, with the previous context freed each time.

Reusing one context across refactorizations (reset rather than realloc, since m
and the sparsity budget are fixed for the life of the solve) would target that
setup cost. It is structurally bit-identical — the same values are computed, only
the buffers' provenance changes.

**Predeclared kill criterion:** if an alternating within-process A/B on the
reuse gate does not move `refactor` by ≥5% with control drift below half that,
it is dead. Given this session's record — five work-count models predicted gains
that did not materialise — the honest prior is that allocator traffic will prove
as cheap as the redundant passes did, because the allocations are large,
few (33), and immediately reused while hot in cache.

**This report does not claim it.** It is the last unexplored item in the phase
map, recorded so the next wave does not have to rediscover it, and it must be
funded by direct cycle measurement rather than by counting allocations.

## Phase map now complete

Every phase of the dual simplex has been audited this wave:

| phase | share | status |
|---|---:|---|
| pivot_row | 21.97% | index cache shipped (−1.14%); scatter is real work |
| ftran_col | 20.59% | U′ + boundary audited; closed |
| btran_rho | 17.05% | U′ 2.64% of run, L^T skip banked; closed |
| ratio_test | 13.04% | **Harris dead-division shipped (−2.22%)** |
| rcost_update | 11.04% | K5 fusion measured at 0.05%; closed |
| lu_update | 7.06% | ~430 cyc per elimination step — real work |
| refactor | 6.75% | elimination tight; **init/assemble ~3% is the last candidate** |
| everything else | <2% | whole-wall complement killed at 1.9% |
