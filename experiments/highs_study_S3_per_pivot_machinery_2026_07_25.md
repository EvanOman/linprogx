<!-- Recovered from overmind worker 1ac2b37e transcript. -->

# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

I'll start by reading the provenance doc, then dig into both codebases.

---

## PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

This report was produced by reading `/tmp/highs-study/HiGHS/highs/simplex/` under the 2026-07-25 owner authorisation recorded in `docs/PROVENANCE.md`, which lifts the "never read solver source" rule **for greenbea only**. Nothing here may be reported as part of the 23 clean-room cells. No HiGHS code was copied; what follows is an algorithm description written to be reimplemented independently. No files were modified, no builds, no benchmarks.

---

# 1. Executive answer

HiGHS's dual simplex differs from linprogx's in **five** structural ways, but the pivot-count gap is dominated by **two coupled ones**, and they are not the two the ledger settled on:

1. **HiGHS's ratio test is a bound-flipping (longest-step) test *by default and always*, implemented without a sort.** linprogx's shipped configuration (`src/linprogx/sparse.py:240,272,376` — `leaving_rule=1, expand=1`, and `bfrt` never passed, so `bfrt=0`) does a **pure minimum-ratio Harris test with no bound flipping at all**.
2. **HiGHS's dual Phase 1 is not a separate LP solve.** It is an O(n) *bound reassignment* on the same basis, the same LU factorisation and the same edge weights (`HEkk.cpp:2653-2695`), and Phase 2 resumes by swapping the bounds back (`HEkkDual.cpp:794-796`). The construction cost the campaign priced at **0.1451 s** (44th/46th ledger entries, auxiliary solved as a standalone LP) is, in HiGHS, **≈ zero**. This is the single most decision-relevant finding in this report.

These two compound: in Phase 1 *every* variable is boxed with range ≤ 2, so the bound-flipping test is maximally productive exactly where linprogx has installed 1e5-scale big-M boxes and then explicitly excluded them from flipping.

There is also a **re-reading of the existing measurements** that changes the ranking, given in §7: the ledger's own numbers imply HiGHS's *Phase 2* needs **1,633** pivots where linprogx needs **3,529** from a transferred version of the same basis. The gap is not mostly phase architecture. It is mostly the Phase-2 ratio test.

---

# 2. Per-mechanism comparison

| Mechanism | HiGHS | linprogx | Class |
|---|---|---|---|
| **Leaving row (CHUZR) rule** | Dual steepest edge by default (`kSimplexEdgeWeightStrategyChoose`, `HighsOptions.h:921-928` → `HEkkDual.cpp:2302-2305`), merit = `infeas² / γ_i` (`HEkkDualRHS.cpp:70-75`) | Dantzig: max raw violation (`_csparse.c:14381-14384`, `leaving_rule=1`) | **Different Y** |
| **DSE weight init** | All-1 when basis is logical (exact, free) — `HEkkDual.cpp:148-155`; crash is **off** by default (`HighsOptions.h:914-919`). Exact `‖B⁻ᵀe_i‖²` by m BTRANs only if basis is non-logical (`HEkk.cpp:2056-2091`) | Exact `‖B⁻ᵀe_i‖²` by m BTRANs on a triangular **crash** basis (`_csparse.c:13843-13862`) | **Different Y** |
| **DSE weight floor** | `1e-4` (`SimplexConst.h:162`, applied `HEkk.cpp:2234-2235`) | `1e-12` (`_csparse.c:14504`, `14386-14387`) | **Different Y** |
| **Stale-weight rejection** | Recomputes exact γ from `row_ep` each CHUZR; **rejects the row and re-selects** if updated < 0.25 × exact (`HEkkDual.cpp:1449-1490`, `1504-1512`) | Overwrites γ_r *after* selection; never re-selects (`_csparse.c:14494-14507`) | **HiGHS has, linprogx lacks** |
| **DSE→Devex fallback** | Adaptive on measured NLA cost (`HEkkControl.cpp:77-130`) | None | **HiGHS has, linprogx lacks** |
| **CHUZR scan order** | Random start index, wraps (`HEkkDualRHS.cpp:60-63`, `80-83`) — ties resolve non-systematically | Strict `score > max` from `k=0`; SIMD lanes tie-break to lowest index (`_csparse.c:14369-14392`, `12633-12642`) | **Different Y** |
| **CHUZR candidate list** | Maintained hyper-sparse infeasibility list with merit cutoff, rebuilt on demand (`HEkkDualRHS.cpp:432-513`, `380-410`) | Full dense scan of all m rows every pivot | **Different Y** (per-pivot cost, not count) |
| **Entering rule (CHUZC)** | **Bound-flipping ratio test, always**: geometric pre-filter → group partition → longest admissible group with pivot-size backoff → flip all earlier groups (`HEkkDualRow.cpp:116-315`) | Min-ratio Harris band `[θ_min, θ_min+1e-7]`, largest \|α\| (`_csparse.c:14877-14923`). BFRT exists but is **off** in the shipped config | **Different Y** |
| **Harris relaxation** | Per-column *numerator* relaxation by `Td` (dual feasibility tol) — `HEkkDualRow.cpp:96-98`; scale-correct | Fixed *ratio* relaxation `harris_delta = 1e-7` (`_csparse.c:14614`); not scale-invariant | **Different Y** |
| **Min-pivot admission** | Ramps with LU age: `1e-9 / 3e-8 / 1e-6` at `update_count` <10 / <20 / ≥20 (`HEkkDualRow.cpp:83-86`) | Fixed `1e-9` (`_csparse.c:14648`) | **Different Y** |
| **Bad-pivot recovery** | `improveChooseColumnRow()` — iterative refinement of `row_ep` + quad-precision PRICE, then redo CHUZC; on later passes deletes the pivot from the pack and retries (`HEkkDual.cpp:1631-1701`, `1737-1783`, `HEkk.cpp:4281-4327`) | Skip the pivot, inflate the Devex weight, `continue` — one whole BTRAN+PRICE wasted, no progress (`_csparse.c:15228-15246`) | **HiGHS has, linprogx lacks** |
| **Dual Phase 1** | Bound-swap subproblem on the **same basis / same LU**: free→[−1000,1000], upper-only→[−1,0], lower-only→[0,1], boxed or fixed→[0,0]; applied over structurals **and row variables** (`HEkk.cpp:2666-2695`). Exit = restore bounds (`HEkkDual.cpp:794-796`) | **None.** Dual feasibility forced at start by big-M artificial bounds, `M = 1e5 × scale` (`_csparse.c:13890-13975`) | **HiGHS has, linprogx lacks** |
| **Cost perturbation** | **On by default** (`dual_simplex_cost_perturbation_multiplier` default 1.0, `HighsOptions.h:1590-1595`); scale-aware `5e-7·max_abs_cost·(1+U)·(\|c_j\|+1)` signed toward the finite bound, plus `~1e-12` perturbation on **row** costs (`HEkk.cpp:2523-2565`) | Off in shipped config; `pricing==1` gives a fixed `1e-9` deterministic hash on structurals only, no row perturbation (`_csparse.c:13600-13615`) | **Different Y** |
| **Cost shifting** | Per-iteration when `theta_dual == 0` (`HEkkDual.cpp:2073-2077`), undone for the leaving variable (`shiftBack`, `2215-2222`); plus randomised shifts to `(1+U)·Td` at every rebuild (`HEkkDual.cpp:2483-2494`) | Env-gated, default off (`_csparse.c:14094-14100`, `15048-15082`) | **Different Y** |
| **EXPAND** | **Not implemented** | Gill/Murray/Saunders EXPAND on the dual, τ₀=5e-10, δ=5e-11, cap 1e-8 (`_csparse.c:14101-14158`, `15083-15096`) | **linprogx has, HiGHS lacks** |
| **Anti-cycling** | Basis-hash cycling detection + taboo list; taboo rows have their infeasibility zeroed during CHUZR (`HEkk.cpp:3152-3200`, `3999-4012`; applied `HEkkDual.cpp:1413`) | Bland's rule after 200 consecutive degenerate pivots (`_csparse.c:15578-15589`) | **Different Y** |
| **Free-variable handling** | Kept genuinely free; `freeList` + `createFreemove` temporarily assigns a sign so the ratio test can use them (`HEkkDualRow.cpp:576-600`) | Converted to big-M boxes `[−M/2, +M/2]` (`_csparse.c:13943-13956`) | **Different Y** |
| **Rebuild-time dual repair** | Flip fixed/boxed, shift the rest with a **random** component (`HEkkDual.cpp:2440-2494`), called every rebuild (`HEkkDual.cpp:1081-1083`) | Deterministic sign-flip repair at every refactorisation, no shifting (`_csparse.c:15528-15556`) | **Different Y** |

---

# 3. Question 1 — the leaving-row rule

**Yes, dual steepest edge.** Default `simplex_dual_edge_weight_strategy = kSimplexEdgeWeightStrategyChoose` (`HighsOptions.h:921-928`, constant `SimplexConst.h:71-72`), interpreted at `HEkkDual.cpp:2302-2305` as `EdgeWeightMode::kSteepestEdge` **with** permission to fall back to Devex.

**Merit.** `HEkkDualRHS::chooseNormal` (`HEkkDualRHS.cpp:37-121`) maximises `work_infeasibility[i] / γ_i`. `work_infeasibility` holds the **squared** primal infeasibility — `HEkkDualRHS.cpp:337-341` (`updatePrimal`), `:367-372` (`updatePivots`), `:425-431` (`createArrayOfPrimalInfeasibilities`) — because `store_squared_primal_infeasibility` is set true unconditionally at `HEkk.cpp:1642` and again at `HEkkDual.cpp:2330`. It is only turned *off* for "less-infeasible DSE" candidates (`HEkkDual.cpp:2331-2338`), whose criterion requires **every** matrix entry to be ±1 (`lp_data/HighsLpUtils.cpp:3179-3183`) — greenbea does not qualify. So the merit on greenbea is genuinely `violation² / γ_i`.

**Weight initialisation — this is where the campaign's DSE probe diverges most.**

- `HighsOptions.h:914-919`: `simplex_crash_strategy` defaults to **off**, so HiGHS starts from a **logical (slack) basis**.
- `HEkkDual.cpp:148-155`: if `logicalBasis()`, the unit weights already assigned at `:146` are **exactly correct** for B = I, and `has_dual_steepest_edge_weights` is set true — zero BTRANs.
- Only if the basis is non-logical does HiGHS pay m BTRANs (`HEkk.cpp:2056-2091`), and even then it prefers Devex when the point is "near-optimal" (`HEkkDual.cpp:158-166`).

linprogx computes exact γ *for a triangular crash basis* (`_csparse.c:13843-13862`) — correct, but on a different and denser starting basis, and at a cost HiGHS never pays.

**Weight update.** Forrest–Goldfarb, `γ_i += ᾱ_i·(γ_r/α_q² · ᾱ_i + Kai·τ_i)` with `Kai = −2/α_q` (`HEkkDual.cpp:2156-2168`, `HEkk.cpp:2225-2233`). This is **algebraically the same recurrence** linprogx already has at `_csparse.c:15364-15393`. The recurrence is not the difference.

**Three things around it are the difference:**

1. **Floor.** `HEkk.cpp:2234-2235` clamps every updated weight to `≥ kMinDualSteepestEdgeWeight = 1e-4` (`SimplexConst.h:162`). linprogx clamps at `1e-12` (`_csparse.c:14504`, and `1e-12` again in the score at `:14386`). That is eight orders of magnitude of freedom for a drifted-small weight to make a row look, in HiGHS's own words, "unreasonably attractive" (`HEkkDual.cpp:1462-1464`).
2. **Rejection, not just anchoring.** `HEkkDual.cpp:1423-1490` is a **loop**: pick a row, BTRAN, compute the exact `‖row_ep‖²`, overwrite the stored weight with it, and if the *previously stored* weight was `< 0.25 ×` exact (`acceptDualSteepestEdgeWeight`, `:1504-1512`, threshold `SimplexConst.h:160`), **discard this row and choose another**. linprogx anchors γ_r with the exact value at `_csparse.c:14494-14507` — but *after* the row is already committed. It corrects the bookkeeping; it does not undo the bad selection.
3. **Escape hatch.** `HEkkControl.cpp:77-130` switches DSE→Devex when the DSE FTRAN density exceeds 1000× the other densities on ≥5% of iterations. linprogx has no such switch.

**Is that what you got wrong?** Partly. Your exact-DSE probe (4,675 pivots, 33rd entry) implemented the *recurrence* faithfully. It did not implement the *acceptance test*, the *1e-4 floor*, or the *logical-basis start where weights are exactly 1 for free*. But see §7 — I do not believe DSE is the top cause, and I would not re-run it first.

---

# 4. Question 2 — the entering-column rule

`HEkkDual::chooseColumn` (`HEkkDual.cpp:1546-1735`) drives it; `HEkkDualRow::chooseFinal` (`HEkkDualRow.cpp:116-315`) is the ratio test. **It is a bound-flipping ratio test, and it is on unconditionally.** There is no option to disable it.

### The algorithm, reimplementably

Let `Δ` = the leaving row's primal infeasibility (`workDelta`), `range_j = u_j − l_j`, `d_j` = reduced cost, `move_j` ∈ {+1,−1} the nonbasic direction.

**Stage 0 — admit (`choosePossible`, `HEkkDualRow.cpp:80-102`).** Keep column j iff `α_j·move_out·move_j > Ta`, where `Ta` **ramps with LU age**: `1e-9` for `update_count < 10`, `3e-8` for `< 20`, `1e-6` thereafter. Track `workTheta = min_j (d_j·move_j + Td)/α_j` — the relaxation is applied to the **numerator** as a dual-infeasibility allowance of `Td`, not to the ratio.

**Stage 1 — geometric pre-filter (`chooseFinal`, `HEkkDualRow.cpp:135-152`).** Start `selectTheta = 10·workTheta + 1e-7`. Repeatedly sweep the candidate list moving every j with `α_j·selectTheta ≥ move_j·d_j` to the front while accumulating `totalChange += range_j·α_j`; multiply `selectTheta` by 10 each round; stop when `totalChange ≥ |Δ|` or everything is collected. This bounds the work before any breakpoint structure is built, **with no sort**.

**Stage 2 — group partition (`chooseFinalWorkGroupQuad`, `HEkkDualRow.cpp:316-368`).** Repeatedly: sweep the remaining candidates, moving to the front every j whose `move_j·d_j ≤ selectTheta·α_j` and accumulating `totalChange += α_j·range_j`; simultaneously compute `remainTheta = min (move_j·d_j + Td)/α_j` over the *rejected* ones. Push the running count onto `workGroup`; set `selectTheta = remainTheta`; repeat until `totalChange ≥ |Δ|`. The result is the breakpoints of the piecewise-linear dual objective, **bucketed into groups, in O(passes × count), still with no sort.** (A heap variant exists at `:426-493` but `use_heap_sort` is hard-wired false at `:159-161`.)

**Stage 3 — pick the group (`chooseFinalLargeAlpha`, `HEkkDualRow.cpp:495-528`).** Compute `finalCompare = min(0.1 × max_j α_j, 1.0)`. Scan groups **from the last backwards**; within each, take the largest α (ties broken by a **random permutation** `workNumTotPermutation`, `:517-523`); accept the first group whose best α exceeds `finalCompare`.
> This is the load-bearing line. HiGHS takes the **longest** dual step — the most flips — and only backs off to a shorter step when the pivot available at the long end is smaller than 10% of the biggest pivot anywhere in the row. It is a long-step-first policy with a *relative* stability guard.

**Stage 4 — decide the flips (`HEkkDualRow.cpp:250-269`).** Every candidate in **every group strictly before** `breakGroup` is flipped to its opposite bound. `updateFlip` (`:530-546`) then accumulates `Σ change_j · a_j` into `bfrtColumn`, which gets a single FTRAN (`HEkkDual.cpp:1970-2006`) and one combined primal update. If `workTheta == 0`, the flip set is discarded (`:302`).

**Stage 5 — outer CHUZC retry (`HEkkDual.cpp:1602-1701`).** If `‖row_ep‖ · α_row ≤ dual_simplex_pivot_growth_tolerance` (default `1e-9`, `HighsOptions.h:1604-1608`): on pass 0 call `improveChooseColumnRow` and redo the whole thing; on later passes **delete that column from the packed row** and redo. The loop exits only with an acceptable pivot or with dual-unboundedness. **HiGHS never spends an iteration on a bad pivot.**

### Against linprogx

- Shipped linprogx never flips (`bfrt=0`). Its `bfrt=1` path (`_csparse.c:14795-14853`) walks breakpoints in ratio order and stops at the **first** breakpoint whose absorption ≥ the remaining slope. HiGHS stops at the **last** group before the slope turns, then backs off only for pivot size. Similar in spirit, different in policy.
- linprogx's `flippable` predicate (`_csparse.c:14686-14695`) requires `j < n && !has_art_bound[j] && isfinite(lo) && isfinite(hi)`. **Artificially-boxed columns are explicitly excluded.** Since linprogx installs big-M boxes on exactly the one-sided columns whose reduced cost points at infinity, and since HiGHS's Phase 1 makes *every* column boxed with range ≤ 2, linprogx's flip mechanism is structurally inert precisely where HiGHS's is maximally active.
- linprogx `qsort`s all admissible candidates every pivot (`_csparse.c:14795`). HiGHS **never sorts** in its default path. Your measured BFRT regression (411.1 → 530.7 ms, −101 pivots) is consistent with paying an O(k log k) sort per pivot for a flip mechanism that had almost nothing to flip. That is an implementation cost, not a verdict on the algorithm.

---

# 5. Question 3 — `improveChooseColumnRow()`

`HEkkDual.cpp:1737-1783`. Called from exactly one site (`:1657-1661`) when the chosen pivot fails the growth test on the **first** CHUZC pass.

It does four things:
1. Undo the temporary `nonbasicMove` values on free columns (`deleteFreemove`, `:1751-1753`).
2. **One step of iterative refinement on `row_ep`** (`HEkk.cpp:4281-4327`): form the residual `B^T·row_ep − e_p` in **quad precision** (`unitBtranResidual`, `:4330-4350`, using `HighsCDouble`), scale it by the nearest power of two, BTRAN the residual, subtract the correction, and re-sparsify.
3. Recompute the pivotal row by PRICE **in quad precision** (`quad_precision = true`, `:1774-1775`) versus the `false` used on the normal path (`:1571-1572`).
4. Re-pack, and let the caller redo CHUZC on the improved row.

**Why it exists.** The dual ratio test divides by α_row. A pivot that is small *relative to `‖row_ep‖`* is usually not a genuinely small pivot — it is cancellation error in an inaccurate `B⁻ᵀe_p`. Cheaply refining `row_ep` and re-running the ratio test typically reveals that the true pivot is fine, or that a *different* column was always the right answer. It converts a would-be-wasted iteration into a real one. Rate is instrumented (`num_improve_choose_column_row_call`, `:1656`) — it is expected to be rare.

**linprogx has no analogue.** Its tiny-pivot path (`_csparse.c:15228-15246`) inflates the Devex weight and `continue`s: the BTRAN, the CSR scatter and the ratio test for that iteration are thrown away, and — because the loop counter is unconditional (`_csparse.c:14195`) — **the wasted trip still counts as an "iteration"**. See the caveat in §9.

---

# 6. Question 4 — what HiGHS has that linprogx has *nothing* of

Strictly "HiGHS does X, linprogx does not do anything in that place":

1. **A dual Phase 1 (`HEkk.cpp:2653-2695`).** The whole mechanism, described below.
2. **DSE weight-error rejection with re-selection (`HEkkDual.cpp:1449-1490`, `1504-1512`).**
3. **Adaptive DSE→Devex switching (`HEkkControl.cpp:77-130`).**
4. **`improveChooseColumnRow` + the CHUZC retry loop (`HEkkDual.cpp:1602-1701`, `1737-1783`).**
5. **Basis-hash cycling detection with a taboo list (`HEkk.cpp:3152-3200`).** Hashes the prospective basis; if the same basis hash recurs on successive iterations, the (row_out, var_out, var_in) triple is marked taboo, and taboo rows' infeasibilities are zeroed in CHUZR (`HEkk.cpp:3999-4012`, applied `HEkkDual.cpp:1413`, restored `:1476`). linprogx's Bland fallback after 200 degenerate pivots is a different and far blunter device.
6. **Genuine free-variable handling (`HEkkDualRow.cpp:576-600`).** A free nonbasic gets a *temporary* sign from the pivot row so the ratio test can consider it, cleared afterwards (`deleteFreemove`, `:602-609`). linprogx replaces free variables with `[−M/2, +M/2]` boxes.
7. **The age-ramped minimum-pivot threshold (`HEkkDualRow.cpp:83-86`).**

### The Phase 1, in detail — because this is the one that matters

`HEkk::initialiseBound(kDual, kSolvePhase1)` (`HEkk.cpp:2653-2695`) loops over `num_tot = num_col + num_row` — **structural and row variables alike** — and rewrites bounds:

| original | Phase-1 bounds |
|---|---|
| `(−∞, +∞)` (free) | `[−1000, 1000]` |
| `(−∞, u]` | `[−1, 0]` |
| `[l, +∞)` | `[0, 1]` |
| `[l, u]` boxed or fixed | `[0, 0]` |

That is the entire construction. **A, b and the basis are untouched. The LU is not refactorised. The edge weights are kept.** Then `initialiseNonbasicValueAndMove` (`:2716-2790`) sets nonbasic values from the new bounds, and `solvePhase1` (`HEkkDual.cpp:573-812`) runs **the same `iterate()`** (`:1185-1268`) as Phase 2.

Three consequences follow directly from the source:

- **b vanishes.** HiGHS's internal LP is `Ax − s = 0` with `s` carrying the row bounds. Replacing *all* bounds — including the row variables' — makes the Phase-1 subproblem homogeneous. This is precisely the exact b-invariance your 46th entry proved by Fenchel duality and verified behaviourally by six b-perturbations. The source confirms the mechanism at `HEkk.cpp:2666` (`for iCol < num_tot`).
- **Every variable is boxed with range ≤ 2** (0, 1, or 2000 for free). So `workRange_[j]` is tiny and uniform, `totalChange` in `chooseFinalWorkGroupQuad` accumulates fast, and the bound-flipping ratio test flips a large fraction of the pivot row every iteration. Phase 1 is BFRT-dominant by construction.
- **Exit costs nothing.** `HEkkDual.cpp:794-796`: on reaching `kSolvePhase2`, call `initialiseBound(kDual, kSolvePhase2)` and `initialiseNonbasicValueAndMove()`. Same basis, same LU, same weights, continue. Termination test is `info.dual_objective_value == 0` (`HEkkDual.cpp:696-700`) — the Phase-1 dual objective *is* the negated sum of dual infeasibilities.

**This is the answer to "we could never beat its construction cost."** There is no construction. Your 44th entry built the auxiliary as a standalone LP and paid 0.1451 s; your 46th proved B* is dual-feasible for the original and gives 3,334 native pivots, then correctly judged the pipeline (0.1451 + 0.374 = 0.519 s) worse than cold. In HiGHS the auxiliary is *the same solver run, on the same factorisation, with different bounds*, and its 1,655 iterations are already inside the 3,309 total.

---

# 7. Question 6 — ranked explanations for the pivot gap

First, a correction to the framing, from your own ledger (`experiments/greenbea_dossier_2026_07_18.md:38-40, 63-68`):

- HiGHS presolve-off, on **linprogx's** reduction: **3,309 = DuPh1 1,655 + DuPh2 1,633.**
- linprogx from **HiGHS's transferred Phase-1 basis**: **3,529.**

So HiGHS's Phase 2, from its own Phase-1 basis, needs **1,633**; linprogx, from a transferred version of essentially that basis, needs **3,529** — **2.16× worse**. And linprogx's native B*-start needs **3,334**, versus 1,633.

**The gap is not primarily the phase architecture. It is the Phase-2 engine.** The 40th/44th/46th entries concluded "phase architecture" from the observation that a phase exists; the transfer experiment in the same dossier already falsifies that as the *dominant* term, and this reading of the source says why.

### Ranked candidates

**#1 — No bound-flipping ratio test in the shipped Phase 2 (highest likelihood).**
Shipped linprogx does a pure min-ratio Harris test. HiGHS never does. Every HiGHS iteration takes the longest dual step it can while keeping a pivot ≥ 10% of the row's largest, flipping every crossed boxed breakpoint for free. That is the classic mechanism for large pivot-count reduction in the bounded-variable dual, and it is what a 2.16× Phase-2 gap looks like.
*Why your BFRT measurement doesn't refute this:* (a) your flip predicate excludes big-M-boxed columns (`_csparse.c:14688`), which on greenbea is exactly the one-sided population; (b) your walk stops at the first slope-killer rather than the last group with an acceptable pivot (`_csparse.c:14805-14818` vs `HEkkDualRow.cpp:495-528`); (c) you pay a `qsort` per pivot (`:14795`) where HiGHS pays none — that is your 411→531 ms.
Evidence: `HEkkDualRow.cpp:116-315`, `316-368`, `495-528`; `_csparse.c:14877-14923`, `14686-14695`, `14795-14853`; `sparse.py:240,272,376`.

**#2 — No dual Phase 1; big-M boxes instead (very high likelihood, tightly coupled to #1).**
`M = 1e5 × max(1, max|bound|, max|b|)` (`_csparse.c:13905-13920`). Consequences: the start is dual-feasible but *wildly* primal-infeasible, so the dual simplex must drive ~1e5-scale violations out one row at a time; the artificially-boxed columns are excluded from flipping; and a genuine Phase 1 — which HiGHS runs in 1,655 cheap, flip-heavy iterations on the same LU — never happens. And the reason your replication failed is now visible: HiGHS's Phase 1 costs one O(n) loop, not a separate LP solve.
Evidence: `HEkk.cpp:2653-2695`; `HEkkDual.cpp:610-611`, `794-796`; `_csparse.c:13890-13975`.

**#3 — Cost perturbation is on by default in HiGHS, off in linprogx (medium).**
`HEkk.cpp:2523-2565`, multiplier default 1.0 (`HighsOptions.h:1590-1595`). Perturbation is scale-aware, signed toward the finite bound, and — unlike your `pricing==1` variant — **also applied to row/logical costs** (`HEkk.cpp:2551-2560`), which breaks ties among slacks. Plus per-iteration shifting (`HEkkDual.cpp:2073-2077`) and randomised rebuild-time shifts (`:2483-2494`). Ranked third only because the dossier reports greenbea as near-degeneracy-free — but see the caveat in §9: that measurement is not trustworthy under `expand=1`.

**#4 — DSE, but with the acceptance test and the 1e-4 floor (medium-low).**
Your 4,675-pivot exact-DSE probe implemented the recurrence without the re-selection loop (`HEkkDual.cpp:1449-1490`), without the `1e-4` floor (`HEkk.cpp:2234-2235` vs your `1e-12`), and from a crash basis rather than the logical basis where HiGHS's weights are exactly 1 for free. Those are three real differences. It is ranked below #1/#2 because it does not explain a 2.16× Phase-2 gap on its own, and because your ledger correctly closed the leaving-rule *family* on the current formulation. Retest it only *after* #1/#2 change the formulation.

**#5 — Systematic tie-breaking bias (low cost, low-to-medium payoff).**
HiGHS randomises the CHUZR scan start (`HEkkDualRHS.cpp:60-63`) and the CHUZC `|α|` tie-break (`HEkkDualRow.cpp:517-523`). linprogx resolves both to the lowest index (`_csparse.c:14369-14392`, `12633-12642`, `14915-14921`). Your own ledger notes that on ±1-heavy matrices `|α|` ties are pervasive and index-order tie-breaking cost 2× pivots on cre_d. This is not the same as `leaving_rule=2` (which restricted the candidate set to one section — partial pricing, a much more damaging change): a randomised *start* keeps the argmax global and changes only tie resolution.

**#6 — No CHUZC retry / `improveChooseColumnRow` (low for count, real for wasted work).**
`HEkkDual.cpp:1602-1701`. Every linprogx tiny-pivot skip (`_csparse.c:15228-15246`) is a full BTRAN + PRICE + ratio test discarded, and it increments the iteration counter.

**#7 — Fixed vs age-ramped minimum pivot, and ratio-band vs numerator relaxation (low).**
`HEkkDualRow.cpp:83-86`, `:96-98` vs `_csparse.c:14648`, `14614`.

**Explicitly ranked out:** EXPAND is a linprogx-only mechanism; HiGHS has none, so it cannot be the source of HiGHS's advantage — but it may be *costing* linprogx pivots by suppressing the very degeneracy signal that would justify perturbation (§9).

---

# 8. The minimal change for candidate #1

Candidates #1 and #2 are one change, because a working BFRT requires boxed columns, and boxed columns are what a real Phase 1 manufactures. The minimal path:

**Step A — make `b` a bound instead of a RHS (≈ 5 lines).**
linprogx already carries artificial columns `n..n+m−1` with bounds `[0,0]` (`_csparse.c:13617-13621`) — these *are* HiGHS's logical variables. Set `lo_ext[n+i] = hi_ext[n+i] = b[i]` and use RHS `0`. The basis matrix, the crash, the LU and every kernel are unchanged; only the bound arrays and the `x_B` recompute (`_csparse.c:14216-14234`) move.

**Step B — Phase 1 as a bound swap (≈ 40 lines, no new solver).**
Before the main loop: save `(lo_ext, hi_ext)`, overwrite every `j < n_total` with the HiGHS map (free→`[−1000,1000]`, upper-only→`[−1,0]`, lower-only→`[0,1]`, boxed/fixed→`[0,0]`), re-derive nonbasic values from `bound_status`, recompute `y`/`r`. Run **the existing loop unchanged**. When CHUZR finds no violated row (`_csparse.c:14456`), restore the true bounds, re-derive nonbasic values, recompute `y`/`r`, and continue — **same basis, same LU, no refactorisation**. This is where the 0.1451 s disappears.

**Step C — turn BFRT on and delete the sort.**
Set `bfrt=1` in `sparse.py`. Replace the `qsort` at `_csparse.c:14795` with HiGHS's two-stage sweep: the geometric `selectTheta` pre-filter (`HEkkDualRow.cpp:135-152`) then the group partition (`:316-368`), both O(passes × candidates). Replace the first-slope-killer termination with the backward group scan and the `finalCompare = min(0.1·max|α|, 1.0)` backoff (`:495-528`).

**Step D — delete the big-M machinery.**
With a real Phase 1 the initial nonbasic placement no longer has to be dual-feasible, so `bigM`, `has_art_bound`, the `M`-retry, the `dual_unbounded_boxed` status and the `!has_art_bound[j]` flip exclusion all go away. Free variables become genuinely free and need HiGHS's `freeList`/`createFreemove` treatment (`HEkkDualRow.cpp:576-600`).

**Falsification gate.** Report Phase-1 and Phase-2 pivot counts separately (HiGHS: 1,655 / 1,633). Kill if total > 4,000, or if Phase 2 alone exceeds 2,500, or if µs/pivot on the Phase-2 trajectory rises more than 15% over the current 90.5 µs cold figure.

**Honest ceiling arithmetic.** This attacks the pivot term only. Your conservation law (47th entry) says `pivots × µs/pivot ≈ 0.38–0.40 s` across every construction tested. Reading HiGHS's source does **not** refute that law. What it *does* change is the accounting on the one construction that was measured: the B* pipeline was priced at `0.1451 (auxiliary) + 0.374 (DS) = 0.519 s`; if the auxiliary is a bound swap on the same LU, its cost collapses toward the 1,655 Phase-1 pivots' own kernel time, and Phase 1's kernels run on the *early*, sparse basis, not the dense B* one. Whether that clears the −41% gate is a measurement question I cannot answer from source. It is the first greenbea lever in the ledger whose *economics* have materially changed.

---

# 9. What I could not determine, and caveats

- **The two pivot counts are not counted the same way.** HiGHS increments `iteration_count_` only inside `updatePivots` (`HEkkDual.cpp:2246`), i.e. once per actual basis change. linprogx's `iterations` counts every trip of the main loop (`_csparse.c:14195`), including tiny-pivot skips (`:15244`), cost-shift restarts (`:14451`) and `x_B` recompute restarts (`:14729`). **4,399 may overstate linprogx's basis changes**, and the true structural gap may be smaller than 4,399 vs 3,309. Adding a separate pivot counter is a two-line, zero-risk clarification that should precede any of this work.
- **The reported "~1/4,399 degenerate pivots" is almost certainly an artifact.** The counter fires on `theta_d < 1e-12` (`_csparse.c:15577`), but with `expand=1` `theta_d` is floored at `expand_dtau/|α| = 5e-11/|α|` (`:15085-15095`), which exceeds `1e-12` for all `|α| < 50`. **The counter is structurally unable to observe degeneracy under the shipped configuration.** greenbea's true degeneracy is unmeasured; candidate #3 may deserve a higher rank than I gave it.
- **Whether the 3,529-from-transfer and 1,633-DuPh2 figures share a basis.** The dossier records the mapping as "validated by 0/4-pivot optimal-basis sanity" — that validates the *index* mapping, not that the transferred basis equals HiGHS's Phase-1 exit basis. My §7 reframing rests on that assumption; it should be re-verified before it drives spend.
- **Whether greenbea trips HiGHS's `near_optimal` path** (`HEkkDual.cpp:106-126`), which would suppress cost perturbation and prefer Devex over exact DSE. That depends on runtime state I did not execute.
- **Whether HiGHS's greenbea runs used `kSimplexStrategyDualPlain` or the PAMI/multi path** (`HEkkDual.cpp:1270-1310`, `HEkkDualMulti.cpp`). Serial single-thread should give plain, but I did not verify it for the benchmark configuration. PAMI's `chooseMultiGlobal` (`HEkkDualRHS.cpp:124-212`) selects a *set* of rows and would change the pivot accounting.
- **HiGHS's scaling** (`simplex_scale_strategy` default equilibration, `HighsOptions.h:905-911`) versus linprogx's conditional Ruiz (`_csparse.c:13472-13485`, skipped when the inf-norm ratio < 100). Scaling changes which pivots the ratio test admits and therefore the path, but I did not trace HiGHS's scale factors far enough to say whether greenbea is scaled at all.
- I did not read `HEkkPrimal.cpp`, `HSimplexNla*`, or the presolve. The presolve-geometry term (473 pivots) is already measured in your ledger and is out of this report's scope.