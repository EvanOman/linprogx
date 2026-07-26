# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

# In-place dual Phase 1 — implemented, correct-by-default, NOT yet winning

**Status: WORK IN PROGRESS. Both gates default OFF; the shipped path is
untouched (trace digest `679168a4baad36d6`, 4,399 pivots, 93 tests green).**

## What is built and verified

### 1. Logical form (`LINPROGX_DS_LOGICAL_FORM=1`) — WORKS

Carries the RHS in the artificial (logical) column bounds instead of a separate
RHS vector: `lo[n+i] = hi[n+i] = -b_i`, RHS `0`. Row `i` reads
`sum_j A[i,j] x_j + x_{n+i} = 0`, so `x_{n+i} = -b_i` reproduces `Ax = b`.

Verified result-identical: **4,399 pivots, optimal, residual 1.769e-07**,
objective matching to 1e-15 relative (the last-digit difference is expected —
the RHS arithmetic path changed). The basis, crash, LU and every kernel are
untouched.

This is the prerequisite: it is what lets Phase 1 be a bound swap.

### 2. In-place dual Phase 1 (`LINPROGX_DS_PHASE1=1`) — RUNS, INCOMPLETE

Installs the Phase-1 bound map before the main loop, runs the **same** iteration
loop on the **same** basis and factorization, then restores the true bounds at
the `leaving_basis_pos < 0` exit and continues into Phase 2 without breaking.

Two bugs found and fixed along the way, both worth recording:

- **The map must be built from the TRUE bounds.** Reading `lo_ext/hi_ext` sees
  the bounds *after* big-M has already replaced every infinite one with a finite
  artificial. The map then classifies everything as "boxed", emits `[0,0]`
  everywhere, sets all nonbasics to 0, and Phase 1 correctly concludes it has
  nothing to do — **DuPh1 = 1 pivot**. Fixed by sourcing from
  `lo_true/hi_true`. After the fix: **DuPh1 = 2,921 pivots**.
- The `basic artificial with |x_B| > tol => infeasible` test assumes artificials
  sit at 0, which is false under the logical form (they sit at `-b_i`). Fixed to
  compare against the artificial's own bound.

## Measured, and it does not yet win

| logical | phase1 | bfrt | pivots | ms | status |
|---|---|---|---:|---:|---|
| — | — | — | **4,399** | 582.8 | optimal (baseline) |
| — | — | ✓ | 4,298 | 781.7 | optimal — fewer pivots, SLOWER (the recorded BFRT net loss) |
| ✓ | — | — | 4,399 | 542.4 | optimal |
| ✓ | — | ✓ | 4,298 | 757.6 | optimal |
| ✓ | ✓ | — | 5,124 | 589.3 | **dual_infeasible** |
| ✓ | ✓ | ✓ | 4,675 | 829.2 | optimal |
| **HiGHS** | | | **2,836** | | DuPh1 1448 + DuPh2 1376 + PrPh2 12 |

## Why it does not win yet — the specific missing piece

**HiGHS checks `info.dual_objective_value == 0` at Phase-1 optimality
(`HEkkDual.cpp:688`) before entering Phase 2.** linprogx does not. My Phase 1
terminates at Phase-1 primal feasibility with dual infeasibilities still
present, and Phase 2 then correctly reports `dual_infeasible`.

That is a real, addressable gap, not a refutation of the mechanism:

1. The Phase-1 exit needs the dual-objective test, and a Phase-1 that has
   *not* reached zero must keep going (HiGHS loops back through `rebuild()`)
   rather than hand a dual-infeasible basis to Phase 2.
2. Even when it completes, our Phase 1 costs **2,921** pivots where HiGHS's
   costs **1,448**, and our Phase 2 costs ~2,200 where HiGHS's costs 1,376.
   **We are ~2x worse in both phases**, which reproduces the standalone Python
   simulation (2,418 + 2,399) and confirms the structure alone is not the win.

## What the ranked evidence now says

Worker S3's reading holds up: the gap is **mostly the Phase-2 ratio test and
pricing**, not the phase architecture. Its remaining ranked items are unbuilt:

- **DSE with HiGHS's acceptance test** — recompute exact gamma each CHUZR and
  **re-select the row** if the stored weight was `< 0.25x` exact
  (`HEkkDual.cpp:1449-1490`, `1504-1512`). linprogx anchors gamma only *after*
  committing: it fixes the bookkeeping, not the bad selection.
- **DSE weight floor `1e-4`** (`SimplexConst.h:162`) versus our `1e-12` — eight
  orders of magnitude for a drifted weight to make a row "unreasonably
  attractive".
- **Cost perturbation applied to row/logical costs too** (`HEkk.cpp:2551-2560`),
  which breaks ties among slacks; our variant perturbs structural costs only.
- **Randomised CHUZR scan start and CHUZC tie-break**
  (`HEkkDualRHS.cpp:60-63`, `HEkkDualRow.cpp:517-523`) versus our
  lowest-index resolution — and our own ledger records that index-order
  tie-breaking cost 2x pivots on cre_d.

## Safety

Both gates default **OFF**. Default path re-verified: trace digest
`679168a4baad36d6` over 6,016 solve vectors, 4,399 pivots, objective
`-72555248.12984592`, 93 targeted tests passing. The board is unaffected and
remains **23W-0P-1L**.
