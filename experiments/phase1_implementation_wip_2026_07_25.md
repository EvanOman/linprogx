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

---

## CORRECTION (same session): Phase 1 is NOT incomplete — it works, and it loses

Above I attributed the `dual_infeasible` result to a missing
`dual_objective_value == 0` check. **That diagnosis was wrong.** Measuring the
quantity HiGHS actually tests:

```
[phase] DuPh1 dual_objective = 0.000000e+00 ; columns with no valid true placement = 94
[phase] DuPh1 = 2921 pivots
[phase] TOTAL = 5124 pivots (status optimal)
```

**The Phase-1 dual objective reaches exactly zero** — HiGHS's own success
criterion (`HEkkDual.cpp:688`) is satisfied. Phase 1 is doing its job: it
removes every dual infeasibility and hands Phase 2 a dual-feasible basis, which
then completes to `optimal` with the correct objective.

The "94 columns with no valid true placement" is a **tolerance artifact**: those
are columns whose `|r_j|` lies below the dual feasibility tolerance, so the
solver legitimately treats them as feasible while my strict `r_j > 0.0` test
counts them as violations.

### The real verdict

The mechanism is **correctly implemented and it does not help**:

| | Phase 1 | Phase 2 | total |
|---|---:|---:|---:|
| linprogx in-place two-phase | 2,921 | 2,203 | **5,124** |
| linprogx big-M (shipped) | — | — | **4,399** |
| **HiGHS** | **1,448** | **1,376 (+12)** | **2,836** |

We are roughly **2x worse than HiGHS in both phases independently**. That
reproduces the standalone Python simulation (2,418 + 2,399 = 4,817) from a
completely different code path, so it is a property of our pivot selection, not
of this implementation.

### What this settles

The phase architecture was **necessary to understand** — it dissolved the
conservation law and explained the b-invariance — but it is **not the source of
HiGHS's advantage**. Worker S3 predicted exactly this and ranked the ratio test
and pricing above the phase structure. That ranking is now confirmed
experimentally from two independent directions.

The remaining 1,563-pivot gap is in **CHUZR/CHUZC quality**, and the specific
unbuilt items are enumerated above (DSE acceptance test with row re-selection,
the `1e-4` weight floor, row-cost perturbation, randomised tie-breaks). Those
are the next wave, and they are pivot-selection changes rather than
architectural ones.

Both gates remain default **OFF**; the shipped path is untouched.

---

## Pricing wave: the edge-weight floor is NOT the lever

HiGHS clamps every updated dual edge weight to `>= 1e-4`
(`SimplexConst.h:162`, applied `HEkk.cpp:2234-2235`); linprogx shipped `1e-12`.
Worker S3 flagged this as eight orders of magnitude in which a drifted-small
weight can make a row "unreasonably attractive". Made tunable
(`LINPROGX_DS_EDGE_FLOOR`) and swept:

| floor | Devex (rule 0) | exact DSE (rule 5) |
|---|---:|---:|
| 1e-12 (shipped) | 6,807 | 4,675 |
| 1e-8 | 6,807 | 4,675 |
| 1e-6 | 6,614 | 4,675 |
| **1e-4 (HiGHS)** | 7,060 | 4,675 |
| 1e-2 | 6,162 | 4,675 |

Dantzig baseline **4,399**; HiGHS **2,836**.

**KILLED as a lever.** Exact DSE is completely insensitive to the floor (flat at
4,675 across five orders of magnitude); Devex moves erratically and is far worse
than Dantzig throughout. The floor alone does not explain anything.

## Cumulative pricing/architecture results on greenbea

| change | pivots | verdict |
|---|---:|---|
| Dantzig, big-M (shipped) | **4,399** | baseline |
| BFRT on | 4,298 | fewer pivots, **slower wall** |
| logical form | 4,399 | result-identical (prerequisite) |
| in-place two-phase | 5,124 | correct, **loses** |
| two-phase + BFRT | 4,675 | loses |
| exact DSE (any weight floor) | 4,675 | loses |
| Devex (any weight floor) | 6,162–7,060 | loses badly |
| **HiGHS** | **2,836** | — |

## The one structural lever left untested

**DSE weight-error rejection with row RE-SELECTION**
(`HEkkDual.cpp:1423-1490`, `acceptDualSteepestEdgeWeight` `:1504-1512`,
threshold `SimplexConst.h:160`). HiGHS's CHUZR is a **loop**: pick a row, BTRAN,
compute the exact `‖row_ep‖²`, overwrite the stored weight — and if the
*previously stored* weight was `< 0.25 x` exact, **discard that row and choose
another**.

linprogx anchors `gamma_r` with the exact value at `_csparse.c:14494-14507`, but
**after** the row is already committed. It corrects the bookkeeping; it never
undoes the bad selection. That is a genuine algorithmic difference, not a
constant.

Why it is plausibly the real lever: it is the only mechanism found that changes
*which row leaves* on the basis of information that arrives *after* the
tentative choice. Every other difference tested (floor, rule family, phase
structure, BFRT) only re-scores the same candidate set.

Implementation cost is real: HiGHS gets the exact weight free because `row_ep`
is the BTRAN it was going to compute anyway, then re-selects if the check fails
(paying a wasted BTRAN). linprogx computes `rho` only *after* choosing, so
re-selection means restructuring the iteration to tolerate a discarded BTRAN —
which HiGHS also pays.

**This is the next wave, and it is the last item on S3's ranked list that has not
been either built or killed.**

---

## KILL: DSE row re-selection — the last ranked lever, and it does nothing

Implemented HiGHS's acceptance test faithfully: after the BTRAN, compare the
**previously stored** weight against the exact `‖rho‖²` and, if it was
`< 0.25x` exact, ban that row and re-select (`LINPROGX_DS_RESELECT=1`,
tolerance and retry cap both tunable). The discarded BTRAN is a real cost —
HiGHS pays it too.

| reselect | tol | rule | phase1 | pivots | ms | status |
|---|---|---|---|---:|---:|---|
| off | — | DSE | — | 4,675 | 940.0 | optimal |
| **on** | 0.25 | DSE | — | **4,675** | 908.0 | optimal |
| on | 0.50 | DSE | — | 4,675 | 950.6 | optimal |
| on | 0.90 | DSE | — | 4,679 | 1955.3 | optimal |
| off | — | DSE | ✓ | 9,895 | 2425.9 | dual_infeasible |
| on | 0.25 | DSE | ✓ | 11,864 | 2699.1 | dual_infeasible |

**Zero effect at HiGHS's own threshold**, and at 0.9 it only wastes BTRANs
(wall doubles for +4 pivots).

**Why — and this is the interesting part.** The acceptance test cannot fire
because linprogx **already anchors `gamma_r` with the exact `‖rho‖²` on every
single pivot** (`_csparse.c`, the "Continuous drift anchor" block). Our weights
are never stale enough to reject. HiGHS needs the test because its weights can
drift between selections; ours cannot, because we pay for an exact anchor every
iteration that HiGHS does not.

So the difference S3 identified is real in the code but **not operative in our
implementation** — we had already solved the problem it protects against, by a
more expensive route.

## Final tally of the source-informed attack

Every item on the ranked list is now built or killed:

| # | mechanism | outcome |
|---|---|---|
| 1 | in-place dual Phase 1 (bound swap) | **built, correct** (dual objective = 0), **loses**: 5,124 vs 4,399 |
| 1b | logical form (RHS in logical bounds) | **built, result-identical** — the prerequisite |
| 2 | BFRT with boxed columns | 4,298 pivots but **slower wall** |
| 3 | DSE weight floor `1e-4` | **KILLED** — exact DSE flat across 5 orders of magnitude |
| 4 | DSE re-selection on stale weights | **KILLED** — zero effect; we already anchor exactly every pivot |
| 5 | row-cost perturbation | untested (ranked below) |
| 6 | randomised CHUZR/CHUZC tie-breaks | untested (ranked below) |

**Nothing tested beats the shipped 4,399.** HiGHS's 2,836 is not explained by any
single mechanism we have isolated.

## What the source-informed attack DID deliver

1. **The Phase-1 mystery is solved completely.** HiGHS constructs nothing; the
   conservation law was an artefact of assuming Phase 1 must be built rather
   than entered. That question is closed permanently.
2. **Two campaign assumptions corrected.** HiGHS takes 2,836 iterations (not the
   inferred ~3,334), and **linprogx's per-pivot cost is 1.73x better** — on this
   box linprogx is faster in absolute wall. Every clean-room wave was optimising
   the slice where we were already ahead.
3. **The residual is diffuse.** The gap is accumulated pivot-selection quality
   across ~1,500 pivots, not a single missing mechanism. That is a much harder
   and much less glamorous target than "the one trick we could not see".

## Safety

All four new gates default **OFF**: `g_logical_form=0`, `g_ds_phase1=0`,
`g_reselect=0`, `g_edge_floor=1e-12`. Default path verified: trace digest
`679168a4baad36d6`, 4,399 pivots, objective `-72555248.12984592`, **111 tests
passing**. Board unchanged at **23W-0P-1L**.

---

## KILL: the last two ranked items — tie-breaks and row-cost perturbation

### Rotating CHUZR scan start (`LINPROGX_DS_ROT_START=1`)

HiGHS randomises the CHUZR scan start (`HEkkDualRHS.cpp:60-63`) so merit ties are
not always resolved to the lowest row index. Our own ledger records index-order
tie-breaking costing **2x pivots on cre_d**, and greenbea's matrix is ±1-heavy,
so ties should be pervasive. Implemented as a rotating start (`iter % m`), which
keeps the argmax global and changes only tie resolution:

| SIMD | rotate | pivots |
|---|---|---:|
| on | off | 4,399 |
| on | on | 4,399 |
| off | off | 4,399 |
| off | on | **4,399** |

**Zero effect in every configuration. KILLED.** Whatever ties exist on greenbea
do not change the trajectory.

### Row/logical cost perturbation (`LINPROGX_DS_PERTURB_ROWS=1`)

HiGHS perturbs row/logical costs as well as structural ones
(`HEkk.cpp:2551-2560`), breaking ties among slacks; ours covered structural
columns only.

| pricing | perturb rows | pivots |
|---|---|---:|
| 0 | off | 4,399 |
| 0 | on | 4,399 |
| 1 | off | 4,453 |
| 1 | on | **4,453** |

**Zero effect. KILLED.** The reason is structural: greenbea's presolved model is
all-equality, so every logical is *fixed*. There are no slack ranges and
therefore no slack ties to break. HiGHS's mechanism is real but has no purchase
on this instance.

---

# FINAL: the source-informed attack is complete and did not close the gap

Every mechanism identified by reading HiGHS is now built or killed:

| # | mechanism | outcome | pivots |
|---|---|---|---:|
| — | shipped baseline (Dantzig, big-M) | — | **4,399** |
| 1a | logical form (RHS in logical bounds) | built, **result-identical** | 4,399 |
| 1b | in-place dual Phase 1 (bound swap) | built, **correct** (dual obj = 0), loses | 5,124 |
| 2 | BFRT with boxed columns | fewer pivots, **slower wall** | 4,298 |
| 3 | DSE weight floor `1e-4` | **KILLED** (flat over 5 orders) | 4,675 |
| 4 | DSE re-selection on stale weights | **KILLED** (zero effect) | 4,675 |
| 5 | rotating CHUZR tie-break | **KILLED** (zero effect) | 4,399 |
| 6 | row/logical cost perturbation | **KILLED** (no slack ties exist) | 4,399 |
| — | **HiGHS** | — | **2,836** |

**Nothing beats 4,399.** HiGHS's 1,563-pivot advantage is not attributable to any
single mechanism we could isolate, implement and measure.

## What the attack delivered anyway

1. **The Phase-1 mystery is closed permanently.** HiGHS constructs nothing. The
   "conservation law" was an artefact of assuming Phase 1 must be *built* rather
   than *entered*. We had the right mathematical object and the wrong
   implementation strategy.
2. **Two campaign assumptions corrected.** HiGHS takes **2,836** iterations, not
   the inferred ~3,334; and **our per-pivot cost is 1.73x better** (85.7us vs
   148.4us), so on this box linprogx is *faster in absolute wall*. Every
   clean-room wave optimised the slice where we were already ahead.
3. **A negative result with teeth.** The residual is **diffuse** — accumulated
   pivot-selection quality across ~1,500 pivots, not one missing trick. Six
   mechanisms, each real in HiGHS and each individually worth nothing here.

## Honest assessment of what would be required

Matching 2,836 would need HiGHS's *entire* pivot-selection regime as a coupled
system — DSE from a logical basis with its full acceptance/fallback machinery,
its BFRT, its perturbation and shifting, and its rebuild cadence — not
individual mechanisms grafted onto a Dantzig/big-M core. That is a rewrite of
the dual simplex, not a patch, and this wave demonstrates that the pieces do not
pay off individually.

## Safety

All six gates default **OFF**. Default path verified unchanged: trace digest
`679168a4baad36d6`, 4,399 pivots, objective `-72555248.12984592`, residual
1.769e-07. Board remains **23W-0P-1L**, greenbea **~1.156**.

---

## KILL: the coupled configuration too — the "pieces must work together" hypothesis is FALSE

The previous section concluded that matching HiGHS would need its *entire*
pivot-selection regime as a coupled system rather than individual mechanisms
grafted on. **That hypothesis has now been tested and it is false.**

Full HiGHS-shaped configuration and its ablations:

| phase1 | rule | BFRT | reselect | pivots | ms | status |
|---|---|---|---|---:|---:|---|
| — | Dantzig | — | — | **4,399** | 544.7 | optimal (shipped) |
| ✓ | DSE | ✓ | ✓ | **6,771** | 1863.2 | optimal — **full coupled config** |
| ✓ | DSE | ✓ | — | 6,769 | 1920.8 | optimal |
| ✓ | Devex | ✓ | ✓ | 5,919 | 1436.9 | optimal |
| — | DSE | ✓ | ✓ | **4,368** | 1480.4 | optimal |
| ✓ | DSE | — | ✓ | 11,864 | 3762.4 | dual_infeasible |
| **HiGHS** | | | | **2,836** | | |

**The full coupled configuration is 54% WORSE than baseline** (6,771 vs 4,399).
The only variant that beats baseline on pivots is DSE+BFRT+reselect *without* the
phase structure, at 4,368 — a 0.7% pivot gain for **2.7x the wall time**.

## The definitive negative result

The source-informed attack has now tested:

- every mechanism **individually** (6 of them),
- selected **pairs** (phase1+BFRT, DSE+floor, DSE+reselect),
- the **full coupled configuration**, and
- **four ablations** of that configuration.

**Nothing beats the shipped 4,399 pivots at a competitive wall.** Adding HiGHS's
mechanisms to linprogx's dual simplex makes it worse, individually and jointly.

The conclusion is uncomfortable but well-evidenced: **the gap is not in which
mechanisms are present, it is in the quality of every implementation of them.**
HiGHS's DSE, BFRT, perturbation and rebuild cadence are each tuned against each
other over years; ours are individually reasonable and collectively
counterproductive on this instance. That is not something a mechanism transplant
can fix, and it is not something this campaign can close by patching.

What linprogx *does* have is a **1.73x better per-pivot cost** — which is why it
is faster than HiGHS in absolute wall on the dev box despite 55% more pivots, and
why the board loss exists only on Modal's 4-vCPU hosts.

## Final state

All six gates default **OFF**. Default path verified: digest
`679168a4baad36d6`, 4,399 pivots, objective `-72555248.12984592`, residual
1.769e-07, **522 passed / 7 skipped**. No production default changed, no v3
recertification run.

**Board: 23W-0P-1L. greenbea ~1.156.**
