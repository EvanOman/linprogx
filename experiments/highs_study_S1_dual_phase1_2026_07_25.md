<!-- Recovered from overmind worker f3bbc43d transcript. -->

# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

I'll start by reading the provenance doc, then work through the HiGHS source.

---

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

*(I read `/tmp/highs-study/HiGHS/highs/simplex/` and `util/HighsSparseMatrix.cpp` and `lp_data/HighsOptions.h`. Read-only: no files created or modified, no git operations, no builds, no benchmarks. Cross-referenced against linprogx's `src/linprogx/_csparse.c` for the contrast section only.)*

---

# HiGHS dual Phase 1: mechanism, cost structure, and what linprogx would have to change

## 0. Executive answer

HiGHS's dual Phase 1 is **not** big-M, **not** a composite objective, and **not** a separately-constructed auxiliary LP. It is the *subproblem approach*: HiGHS **overwrites the bound arrays** with a fixed ±1/0 box that depends only on which bounds are finite, and then runs **the ordinary dual simplex phase-2 loop, unmodified**, on the same matrix, the same basis, the same LU factorization, the same pricing structures and the same edge weights.

Construction cost is **three O(n+m) passes over arrays that already exist**, plus one FTRAN whose right-hand side has as many nonzeros as there are dual infeasibilities. **Zero factorizations, zero allocations, zero matrix rebuilds, zero edge-weight recomputation.**

That is the answer to the conservation law: *HiGHS never builds a starting basis at all.* It changes two `double` arrays and keeps pivoting. The Phase-1 pivots are ordinary dual pivots at ordinary per-pivot cost — they are not amortising a construction, because there is no construction.

---

## 1. What `solvePhase1()` actually does

### 1.1 The problem it solves

Same matrix `A`, same cost vector `c` (possibly perturbed — see §5), **different bounds**. For every variable `iVar` in `0 .. num_col+num_row-1`, the true bound pair `[l, u]` is replaced by a fixed box determined *only by which of `l`, `u` are infinite*:

| true bound pattern | Phase-1 box | Phase-1 range |
|---|---|---|
| `u = +inf` (LOWER) | `[0, 1]` | 1 |
| `l = -inf` (UPPER) | `[-1, 0]` | 1 |
| both finite (BOXED **or** FIXED) | `[0, 0]` | 0 |
| both infinite (FREE) | `[-1000, 1000]` | 2000 |

`HEkk.cpp:2669-2696`. Note that BOXED and FIXED collapse to the *same* `[0,0]` — the finite bound **values are discarded entirely**.

Then nonbasic values and `nonbasicMove` are recomputed from those boxes (`HEkk.cpp:2716-2771`), which places every nonbasic variable at the end of its Phase-1 box where its *primal value is 0* whenever its dual is *feasible*. HiGHS states this design intent explicitly at `HEkk.cpp:2659-2666` and again at `HEkkDual.cpp:694-700`.

### 1.2 The objective, and why it is the sum of dual infeasibilities

The simplex dual objective is `sum over nonbasic j of workValue[j] * workDual[j]` (`HEkk.cpp:1946-1954`), with the LP objective offset deliberately **omitted in phase 1** (`HEkk.cpp:1956-1963`).

Under the boxes above:
- a **dual-feasible** nonbasic sits at value 0 → contributes 0;
- a **dual-infeasible** nonbasic is flipped to the other end of its unit box by `correctDualInfeasibilities()` (`HEkkDual.cpp:2442-2444`, `HEkk::flipBound` at `HEkk.cpp:3069-3073`) → contributes `±1 × d_j = -(its dual infeasibility)`;
- a nonbasic **free** variable contributes `±1000 × d_j`, i.e. the same thing weighted 1000 (so Phase 1 prioritises killing free-variable dual infeasibilities).

So `dual_objective_value` in Phase 1 is exactly **minus the weighted sum of dual infeasibilities**, and the dual simplex, which monotonically improves the dual objective, is driving that sum to zero. Reaching `dual_objective_value == 0` means every nonbasic is back at value 0, i.e. **dual feasible for the true LP**. That is precisely the exit test at `HEkkDual.cpp:689-703`.

Two important structural consequences fall out of the table:

1. **BOXED and FIXED variables are excluded from the Phase-1 target.** They get `[0,0]`, hence `nonbasicMove = kNonbasicMoveZe` (`HEkk.cpp:2733-2736`), hence dual infeasibility `-move*d = 0` always. This is correct and deliberate: a boxed variable's dual infeasibility is *always* removable by a bound flip, which changes no duals, so it is never an obstruction. The same logic drives the phase-selection test `computeDualInfeasibilitiesWithFixedVariableFlips()` (`HEkkDual.cpp:2341-2385`, esp. the comment at 2342-2352).
2. **There are no free variables during Phase 1** (`[-1000,1000]` is boxed), which is why `dualInfeasCount` — counted only for genuinely free variables in `correctDualInfeasibilities` (`HEkkDual.cpp:2429-2433`) — stays 0 throughout Phase 1, and why `solvePhase1`'s loop has no `if (dualInfeasCount > 0) break;` where `solvePhase2` does (`HEkkDual.cpp:895`).

So the Phase-1 target set is exactly: **the one-sided (LOWER/UPPER) and free nonbasic variables whose reduced costs have the wrong sign.** Everything else is somebody else's problem.

### 1.3 The iteration loop

`solvePhase1()`'s loop body (`HEkkDual.cpp:619-676`) is structurally identical to `solvePhase2()`'s (`HEkkDual.cpp:878-944`): `rebuild()` then repeated `iterate()`. `iterate()` (`HEkkDual.cpp:1185+`) is phase-agnostic — `chooseRow` (CHUZR), `chooseColumn` (PRICE + CHUZC + bound-flipping ratio test), `updateFtran`, `updateDual`, `updatePrimal`, `updatePivots`. **There is no `if (phase == 1)` anywhere in the iteration path.** The phase enters only through the values sitting in `workLower_`/`workUpper_`/`workRange_`/`workValue_`.

- CHUZR selects on primal infeasibility of basic variables against `baseLower/baseUpper`, which are copied from the *Phase-1* boxes of the basic variables in `computePrimal()` (`HEkk.cpp:2943-2944`) and turned into the selection array by `HEkkDualRHS.cpp:412-433`.
- The bound-flipping ratio test consumes `workRange[iCol]` (`HEkkDualRow.cpp:150, 268, 339, 397, 487`), which in Phase 1 is 1, 0 or 2000. Range-0 variables (LP-boxed) can therefore be "flipped" at zero primal cost, and range-1 variables give uniform, well-scaled break-point steps.

This is the classical *dual phase-1 subproblem* method (bounded artificial box, not big-M). The code's own comments trace it to `hsol` (`HEkk.cpp:2671`).

---

## 2. Why it is cheap — the cost argument

This is the crux, so I'll separate *transition cost* from *per-pivot cost*.

### 2.1 What the transition into Phase 1 costs

Everything `solvePhase1()` does before its first pivot:

| step | code | cost |
|---|---|---|
| reset bounds from LP | `HEkk.cpp:2573-2574` (`initialiseLpColBound`/`initialiseLpRowBound`) | 1 pass, O(n+m) writes |
| overwrite with Phase-1 boxes | `HEkk.cpp:2669-2696` | 1 pass, O(n+m) writes |
| reset nonbasic values / moves | `HEkk.cpp:2716-2771` | 1 pass, O(n+m) |
| save backtracking basis | `HEkkDual.cpp:615` → `HEkk.cpp:1897-1920` | O(n+m) copy, once |
| first `rebuild()` | `HEkkDual.cpp:621` | see below |

The first `rebuild()` at the head of Phase 1:
- **does not refactorize.** `rebuildRefactor()` returns `false` immediately when `update_count == 0` (`HEkk.cpp:1969-1971`), and on entry to `HEkkDual::solve` the basis is already inverted — asserted at `HEkkDual.cpp:53`. So the cold Phase-1 entry pays **no LU work whatsoever**.
- **does not touch the matrix.** `A` is unchanged; the row-wise partitioned matrix `ar_matrix_` depends on `nonbasicFlag_`, which the bound swap does not modify, so the reinitialise branch at `HEkkDual.cpp:1047-1055` is skipped (it asserts `backtracking_`).
- **does not recompute edge weights.** `dual_edge_weight_` is set up once in `solve()` *before* the phase is even decided (`HEkkDual.cpp:134-184`), and carries unchanged through Phase 1 and Phase 2.
- runs one `computeDual()` (BTRAN of `c_B` + PRICE), which any start would need anyway — and note the bound swap **cannot change the duals**, since duals depend only on `c` and `B`.
- runs one `computePrimal()` whose FTRAN right-hand side skips every nonbasic with `workValue == 0` (`HEkk.cpp:2925-2929`). In Phase 1 that is *everything except the dual-infeasible flipped variables and nonbasic free variables*. **The initial FTRAN is proportional to the number of dual infeasibilities, not to n.** On a cold logical start (`B = I`) it is not even a solve.
- allocates nothing: `workLower_`, `workUpper_`, `workRange_`, `workValue_`, `workShift_` are sized once in `allocateWorkAndBaseArrays()` (`HEkk.cpp:2398-2420`).

**The transition into Phase 1 is Θ(n+m) array writes plus one sparse FTRAN.** The transition *out* is symmetric: `initialiseBound(kDual, kSolvePhase2)` + `initialiseNonbasicValueAndMove()` (`HEkkDual.cpp:797-798`), two more O(n+m) passes; `initialiseBound` early-returns at `HEkk.cpp:2657` for phase 2, leaving the true LP bounds in place.

### 2.2 What a naive dual Phase 1 pays that this does not

A "solve an explicit auxiliary LP" Phase 1 pays, per entry:

1. **Matrix construction** — materialising a modified constraint matrix (extra artificial columns / rows), O(nnz) memory traffic plus allocation. HiGHS: zero, because the surrogate LP has *the same matrix*.
2. **A fresh LU factorization** of the auxiliary basis, plus rebuilding the eta/update machinery. HiGHS: zero, because the surrogate LP has *the same basis in the same column space*, so `B⁻¹` is bit-for-bit reusable.
3. **Fresh pricing structures** — the partitioned row-wise matrix, the infeasibility list, the freelist. HiGHS: unchanged, because `nonbasicFlag_` is untouched.
4. **Fresh dual steepest-edge weights** — O(m) FTRAN/BTRAN pairs if computed exactly. HiGHS: reused verbatim across the phase boundary.
5. **A transfer/mapping step** at the end, translating the auxiliary solution back into a basis for the real LP. HiGHS: zero, because the basis never left the real LP's column space — only the bounds moved.

Items 2 and 4 are the expensive ones, and they are *exactly* the two things the campaign's constructed starts had to pay for. HiGHS pays neither, because **the surrogate LP shares its entire linear algebra with the real one.** The only thing that differs between the Phase-1 LP and the Phase-2 LP is the contents of two `double[n+m]` arrays.

### 2.3 The reframing that matters

The campaign's conservation law — `pivots × μs/pivot ≈ 0.38-0.40 s` across every constructed start — measures *construction*. HiGHS's Phase 1 is not a construction; it is **extra pivots**. So the correct comparison is not

> 0.145-0.215 s to build a 3,334-pivot basis vs. what it saves

but

> `N₁` Phase-1 pivots + `N₂` Phase-2 pivots, each at the *same* per-pivot cost as an ordinary Phase-2 pivot, vs. `N` Phase-2-only pivots from a dual-feasibility hack.

And the Phase-1 pivots are, if anything, structurally *cheaper* early on: the primal infeasibility list is short (only flipped variables perturb `x_B`), the basis is still `B = I` or near it, and the ratio-test ranges are all exactly 1 or 0. There is no construction whose cost has to be amortised — which is why the conservation law does not bind here. (I am not making any claim about the resulting pivot counts on greenbea; that is your measurement to take.)

---

## 3. Is it genuinely b-invariant?

**Yes — exactly, and the mechanism is visible in three lines of code.**

The reason it is exact rather than approximate is a formulation detail worth stating precisely, because it is where linprogx differs structurally:

> **In HiGHS the simplex system has a zero right-hand side.** The constraint system is `[A | I] · [x ; s] = 0` — logical column `num_col+iRow` is `+e_iRow` (`HighsSparseMatrix.cpp:1630-1636`), `computePrimal()` computes `baseValue = -B⁻¹ · Σ_nonbasic A_j·workValue_j` with **no `b` term at all** (`HEkk.cpp:2917-2945`). The problem's right-hand side lives *entirely* in the bounds of the logical variables: `workLower[num_col+iRow] = -row_upper[iRow]`, `workUpper[num_col+iRow] = -row_lower[iRow]` (`HEkk.cpp:2431-2440`).

Therefore, **overwriting the bound arrays deletes `b`.** There is nowhere else for it to hide.

Channel-by-channel audit of every place `b` could re-enter Phase 1:

| channel | verdict | evidence |
|---|---|---|
| `x_B` computation | `b`-free; `baseValue = -B⁻¹ N x_N`, and `x_N ∈ {0, ±1, ±1000}` | `HEkk.cpp:2917-2945` |
| basic-variable bounds for CHUZR | Phase-1 boxes, not LP bounds | `HEkk.cpp:2943-2944` ← `HEkk.cpp:2669-2696` |
| duals / reduced costs | depend only on `c` and `B` | `HEkk.cpp:2954-2980` |
| ratio-test ranges | `workRange` = 1 / 0 / 2000 | `HEkk.cpp:2695` |
| initial `nonbasicMove` of logicals | logicals all start **basic** on a logical basis | `HEkk.cpp:1164-1169` |
| initial `nonbasicMove` of boxed structurals | the `|lower| < |upper|` tie-break at `HEkk.cpp:1143-1148` **is overwritten** in Phase 1: boxed → `[0,0]` → `move = Ze` | `HEkk.cpp:2691-2693`, `2733-2736` |
| cost perturbation | depends on `c`, the *finiteness* pattern (`workRange_[i] < 1e30`), and the seeded RNG — never on `b` | `HEkk.cpp:2464-2567`, esp. 2507-2509, 2534-2544 |

So the Phase-1 pivot sequence is a function of `(A, c, the finite/infinite pattern of the bounds, the initial basis, the edge-weight state, the RNG seed)` — and of **no bound value whatsoever**, row or column. That is stronger than `b`-invariance: it is invariance to the *values* of all finite bounds. Your behavioural finding and the Fenchel-duality derivation are both confirmed, and this is the mechanical reason.

**One honest caveat:** invariance is *conditional on the initial basis*. If a warm-start or crash basis is supplied whose construction consulted `b`, that dependence enters through the basis, not through Phase 1. With HiGHS's default (`simplex_crash_strategy = kSimplexCrashStrategyOff`, `HighsOptions.h:557`, `916-919`; logical basis at `HEkk.cpp:1122-1173`) there is no such channel on a cold solve.

---

## 4. Termination and handover to `solvePhase2()`

### 4.1 Termination

Phase 1's inner loop exits when a `rebuild_reason` is set; the outer loop exits when the data are fresh from `rebuild()` and no further refactorization is wanted (`HEkkDual.cpp:664-675`). The normal terminating condition is CHUZR finding no primal-infeasible basic variable — `dualRHS.chooseNormal` returns `kNoRowChosen`, which sets `rebuild_reason = kRebuildReasonPossiblyOptimal` (`HEkkDual.cpp:1426-1431`).

Then the outcome is assessed (`HEkkDual.cpp:685-762`):

- **`row_out == kNoRowChosen && dual_objective_value == 0`** → `solve_phase = kSolvePhase2`. *This is the usual exit* (`HEkkDual.cpp:689-703`).
- **`row_out == kNoRowChosen && dual_objective_value != 0`** → `assessPhase1Optimality()` (`HEkkDual.cpp:2553-2601`): if costs are perturbed, `cleanup()` removes the perturbation and recomputes; then `assessPhase1OptimalityUnperturbed()` (`2603-2654`) either declares dual feasibility w.r.t. Phase-2 bounds and goes to Phase 2, or sets `model_status = kUnboundedOrInfeasible` and `solve_phase = kSolvePhaseExit`.
- CHUZC failure / excessive primal values → `kSolvePhaseError` (`718-736`).
- Phase-1 unbounded (`variable_in == -1`) → cleanup path, otherwise error (`737-761`).
- Singularity/backtracking → `kSolvePhaseUnknown`, drop out and re-enter via `solve()`'s dispatcher (`HEkkDual.cpp:627-631`, `221-239`).

### 4.2 What crosses the boundary

At `HEkkDual.cpp:793-812`, on the way to Phase 2:

1. **True bounds restored** — `initialiseBound(kDual, kSolvePhase2)` (early-returns at `HEkk.cpp:2657` leaving the LP bounds from `initialiseLpColBound`/`initialiseLpRowBound`).
2. **Nonbasic values/moves re-derived** — `initialiseNonbasicValueAndMove()`.
3. **Cost perturbation re-armed** — `allow_cost_shifting` / `allow_cost_perturbation` set true unless the cleanup-level guard trips (`HEkkDual.cpp:799-811`).
4. If Phase 1 ended optimal-with-nonzero-objective and is going to Phase 2, `exitPhase1ResetDuals()` (`HEkkDual.cpp:2656-2709`) shifts the *costs* of genuinely free nonbasic variables so their duals become exactly zero (`2688-2691`).

State that survives, unchanged: the **basis** (`basicIndex_`, `nonbasicFlag_`), the **LU factorization** and its updates, the **dual steepest-edge / Devex weights**, the **row-wise partitioned matrix**, the **cost perturbation** (`HEkkDual.cpp:701-702` explicitly notes perturbation persists into Phase 2 until final cleanup), and `workShift_`.

State that does **not** survive — worth flagging because it is a real design choice: for a variable that is **boxed in the LP**, Phase 1 pins it at `[0,0]` with `move = Ze`, so its "at lower / at upper" status is destroyed. On return, `initialiseNonbasicValueAndMove` sees a boxed variable with an invalid incoming move and defaults it **to its lower bound** (`HEkk.cpp:2749-2753`). HiGHS accepts this because the first Phase-2 `rebuild()`'s `correctDualInfeasibilities()` will flip any that are on the wrong side for free (`HEkkDual.cpp:2442-2444`) — flips change primal values only, never duals.

---

## 5. `shiftCost()` / `shiftBack()` — and is perturbation part of the cheapness?

### 5.1 What they are

`workShift_` is a **temporary, per-variable cost bump** that lives alongside `workCost_`. `computeDual()` adds it when forming both `c_B` and the reduced costs (`HEkk.cpp:2961-2962` and `2977`), so it behaves like a genuine cost modification for as long as it is set. It is written **only** by `shiftCost` (`HEkkDual.cpp:2205`) and cleared by `shiftBack` (`2219`) and by `initialiseLpColCost`/`initialiseLpRowCost` (`HEkk.cpp:2704`, `2712`). (`HEkkDualMulti.cpp:303, 955-956` save/restore it across minor iterations in PAMI.)

The single call site is `updateDual()` (`HEkkDual.cpp:2065-2127`):

```
if (theta_dual == 0)  shiftCost(variable_in, -workDual[variable_in]);
...
workDual[variable_in] = 0;  workDual[variable_out] = -theta_dual;
shiftBack(variable_out);
```

`theta_dual` comes from `HEkkDualRow::chooseFinal` (`HEkkDualRow.cpp:242-246`):

```
if (workDual[workPivot] * workMove[workPivot] > 0) workTheta = workDual[workPivot] / workAlpha;
else                                               workTheta = 0;
```

So `theta_dual == 0` occurs **not only** when the entering variable's dual is zero, but also when it is already on the *wrong* side (a negative dual step, clamped to zero). In that second case the shift amount `-workDual[variable_in]` is **nonzero**, and the mechanism is: *perturb the entering variable's cost by exactly minus its reduced cost so its dual becomes 0* — which is what a variable about to become basic needs anyway — and record the perturbation so `computeDual()` reproduces it consistently on the next rebuild. `shiftBack(variable_out)` removes the leaving variable's own stale shift as it becomes nonbasic. `HEkkDualRow.cpp:305` also suppresses all BFRT flips when `workTheta == 0`.

So this pair is an **anti-degeneracy / local dual-feasibility repair inside the pivot**, applied identically in both phases.

### 5.2 Is cost shifting/perturbation part of Phase 1's cheapness?

**No.** They are orthogonal to the mechanism, and I want to be unambiguous about this because it is the most likely place to draw a wrong conclusion:

- **Bulk cost perturbation** (`initialiseCost(..., perturb=true)`, `HEkk.cpp:2442-2569`; default multiplier 1.0, i.e. **on** — `HighsOptions.h:1591-1596`, ctor arg order confirmed at `HighsOptions.h:92-105`) is applied **once in `solve()` before the phase is chosen** (`HEkkDual.cpp:126-127`), not by Phase 1, and it persists through Phase 2 until the final `cleanup()`. It is an anti-degeneracy device that happens to also *reduce the number of Phase-1 iterations needed* — HiGHS even uses the perturbed dual infeasibility count to decide whether Phase 1 runs at all (`HEkkDual.cpp:188-207`) — but it is not what makes the Phase-1 *construction* cheap.
- **`correctDualInfeasibilities()`'s shifting** (`HEkkDual.cpp:2472-2504`) nudges a residual dual to `±(1+rand)·tolerance`. It is a *repair inside `rebuild()`*, not a Phase-1 construction step.
- **`shiftCost`/`shiftBack`** are per-pivot and phase-agnostic.

Phase 1 is cheap because of the **bound substitution**, full stop. The shifting machinery is what keeps the whole dual simplex numerically well-behaved in both phases.

---

## 6. Interaction with the initial basis and with `rebuild()`

### 6.1 Initial basis

HiGHS's default cold start is the **logical (slack) basis**, `B = I`: `HEkk::setBasis()` makes every logical basic and every structural nonbasic (`HEkk.cpp:1122-1173`). Crash is **off by default** (`HighsOptions.h:557`, records at `916-919`). `HEkkDual::initialiseSolve` notes whether the basis is logical (`HEkkDual.cpp:553-559`), and `solve()` uses that to skip exact DSE weight computation when `B = I` (unit weights are already correct — `HEkkDual.cpp:146-155`).

Phase selection happens **before** Phase 1 is entered: duals are computed, dual infeasibilities are counted *ignoring those removable by fixed-variable flips*, and `solve_phase = dualInfeasCount > 0 ? 1 : 2` (`HEkkDual.cpp:188-207`). There is also a `force_phase2` escape when the unperturbed infeasibilities are tiny or involve only fixed variables (`HEkkDual.cpp:69-71`, consumed at `2439-2442` and cleared at `2538`).

Phase 1 therefore inherits whatever basis and factorization the solve started with. **It does not build, crash, or choose a basis.**

### 6.2 `rebuild()`

`rebuild()` (`HEkkDual.cpp:1016-1128`) is phase-agnostic and is called at the head of every outer iteration of both phases. It: optionally refactorizes (`1023-1040`, guarded by `HEkk::rebuildRefactor` at `HEkk.cpp:1969-1997` which refuses when `update_count == 0`), recomputes duals (`1074`), calls `correctDualInfeasibilities()` (`1082`), recomputes primals (`1086`), rebuilds the primal-infeasibility list for CHUZR (`1090-1091`), and recomputes the dual objective **with the current `solve_phase`** (`1096`) — that last argument is the *only* place `rebuild()` knows which phase it is in, and it only controls whether the LP objective offset is added (`HEkk.cpp:1956-1963`).

Everything phase-specific is therefore carried in data, not control flow. That is the whole trick.

---

## 7. Evidence index

| claim | file:line |
|---|---|
| Phase-1 bound substitution (the mechanism) | `HEkk.cpp:2669-2696` |
| Phase-2 bounds = LP bounds (early return) | `HEkk.cpp:2657` |
| Bounds reset from LP before substitution | `HEkk.cpp:2573-2574`, `2421-2440` |
| Rationale in HiGHS's own words | `HEkk.cpp:2659-2666`; `HEkkDual.cpp:694-700` |
| Phase-1 entry: switch bounds + nonbasic values | `HEkkDual.cpp:610-611` |
| Phase-1 loop = rebuild + iterate (same as phase 2) | `HEkkDual.cpp:619-676` vs `878-944` |
| Phase-agnostic iteration | `HEkkDual.cpp:1185-1269` |
| Simplex system has zero RHS; logicals are `+e_i` | `HighsSparseMatrix.cpp:1618-1637`; `HEkk.cpp:2917-2945` |
| `b` lives only in logical bounds | `HEkk.cpp:2431-2440` |
| Phase-1 objective = −Σ dual infeasibilities | `HEkk.cpp:1946-1963`; `HEkkDual.cpp:689-703` |
| Flip infeasible nonbasics to ±1 end | `HEkkDual.cpp:2442-2470`; `HEkk.cpp:3069-3073` |
| Boxed/fixed excluded from Phase-1 target | `HEkk.cpp:2691-2693`, `2733-2736`; `HEkkDual.cpp:2341-2352` |
| Free variables get `[-1000,1000]` | `HEkk.cpp:2670-2684` |
| No refactorization on Phase-1 entry | `HEkk.cpp:1969-1971`; `HEkkDual.cpp:53` |
| Edge weights set once, before phase choice | `HEkkDual.cpp:134-184` |
| Initial FTRAN skips zero-valued nonbasics | `HEkk.cpp:2925-2929` |
| Work arrays pre-allocated | `HEkk.cpp:2398-2420` |
| CHUZR reads Phase-1 boxes | `HEkk.cpp:2943-2944`; `HEkkDualRHS.cpp:412-433` |
| Ratio test consumes `workRange` (1/0/2000) | `HEkkDualRow.cpp:150, 268, 339, 397, 487` |
| Exit → restore true bounds, re-arm perturbation | `HEkkDual.cpp:793-812` |
| Free-variable dual zeroing at exit | `HEkkDual.cpp:2656-2709` |
| Phase-1 optimality assessment | `HEkkDual.cpp:2553-2654` |
| `cleanup()` (de-perturb) | `HEkkDual.cpp:1130-1183` |
| `shiftCost` / `shiftBack` | `HEkkDual.cpp:2199-2222`; call site `2065-2127`; read by `HEkk.cpp:2961-2962, 2977` |
| `theta_dual` clamp that makes the shift nonzero | `HEkkDualRow.cpp:242-246`, `305` |
| Cost perturbation, default ON (multiplier 1.0) | `HEkk.cpp:2442-2569`; `HighsOptions.h:1591-1596`, `92-105` |
| Logical initial basis; crash off by default | `HEkk.cpp:1122-1173`; `HighsOptions.h:557, 916-919` |
| Phase selection before Phase 1 | `HEkkDual.cpp:188-207` |
| `rebuild()` | `HEkkDual.cpp:1016-1128` |

---

## 8. CONTRAST — what linprogx does instead, and why ours costs more

**linprogx has no dual Phase 1 at all.** It buys initial dual feasibility with **big-M artificial bounds**.

In `CSRMatrix_solve_eq_box_dual_simplex` (`src/linprogx/_csparse.c`), section 3 "NONBASIC ASSIGNMENT for dual feasibility" (`_csparse.c:13870-13986`):

- `bigM = 1e5 × max(1, max|finite bound|, max|b_i|)` (`_csparse.c:13891-13911`);
- a free variable gets invented bounds `[-M/2, +M/2]` (`13942-13947`);
- a one-sided variable whose reduced cost points at its **infinite** side gets an invented finite bound at distance `M` and is **parked there**: `lo_ext[j] = hi_ext[j] - bigM` (`13961-13969`) or `hi_ext[j] = lo_ext[j] + bigM` (`13970-13978`), with `has_art_bound[j] = 1`.

Three costs follow, and they compound:

1. **Massive injected primal infeasibility.** A variable parked at `±M` (`M ≥ 1e5 × scale`) contributes `A_j · (±M)` to `x_B = B⁻¹(b − A_N x_N)` (`_csparse.c:14214-14232`). Every row touched by that column starts wildly out of bounds. The dual simplex then has to walk that back **one pivot at a time**, and every one of those pivots is a *full-price* Phase-2 pivot on the *true* problem — BTRAN, PRICE, Harris ratio test, FTRAN, LU update. HiGHS's Phase-1 pivots do the analogous repair work on a surrogate where the *entire* nonbasic value vector is bounded by 1 (or 1000), so the primal displacement being worked off is O(1) per variable, not O(M).
2. **Numerical damage that has to be paid for downstream.** Because a big-M park is not a real bound, linprogx cannot trust its own exit conditions. It carries `lo_true`/`hi_true` shadow arrays (`13885-13889`), excludes artificially-bounded columns from the bound-flipping ratio test (`14682-14690`), must downgrade a genuine empty-ratio-test infeasibility certificate to the non-committal `"dual_unbounded_boxed"` whenever any artificial bound exists (`14716-14728`, again at `15151`), and at claimed optimality has to run a **gap-damage budget test** on every parked column plus a full dual-feasibility re-verification against true bounds (`15716-15804`), with a documented "retry with M×100 once, then give up" escape (`15659-15661`). HiGHS pays none of this: after a successful Phase 1 the basis is dual feasible **for the true bounds**, so Phase 2's optimality test is the real one.
3. **The dual objective is meaningless during the walk-back.** Parked variables contribute `±M · r_j` to the objective, so early progress is dominated by artefacts of `M` rather than by the LP.

There is also a secondary structural difference worth naming: linprogx starts from a **Bixby-style triangular crash basis** (`_csparse.c:13643-13850`, with a post-crash conditioning guard and identity fallback), where HiGHS defaults to `B = I`. That is a separate axis and I make no claim about which is better — but it does mean the two solvers' Phase-2 starting points are not comparable even before the Phase-1 question.

**Summary of the contrast:** HiGHS solves a *bounded* surrogate whose whole purpose is to fix the *signs* of the reduced costs, using unit-width artificial boxes and the same linear algebra. linprogx skips the surrogate and instead pushes the *variables* out to `±M`, converting a dual-sign problem into a huge primal-distance problem that the real Phase 2 then has to pay for in real pivots — plus a certification tax on the way out.

---

## 9. MINIMAL CHANGE linprogx would need

This is deliberately scoped to reuse everything already in `CSRMatrix_solve_eq_box_dual_simplex`. **No new linear algebra, no new pricing, no new ratio test.**

### 9.1 The one structural adaptation

HiGHS's simplex system has RHS `0` with `b` in the logical bounds. **linprogx's has an explicit `b`** (`_csparse.c:14218`, `15691`: `memcpy(rhs, b, ...)` then subtract `A_N x_N`). So the bound substitution alone is *not* sufficient for linprogx — the equivalent transformation is:

> **Phase 1 ⟺ solve with `b := 0` *and* the substituted bounds.**

That is the exact mechanical content of `b`-invariance in linprogx's formulation, and it is a one-line change at the two `memcpy(rhs, b, ...)` sites (use a zero vector, or a `const double *rhs_src` that points at `b` in Phase 2 and at a zero buffer in Phase 1).

### 9.2 The change, concretely

1. **Add a `phase` flag** to the solve loop (a local `int ds_phase`), plus a zero RHS buffer. Nothing else in the loop becomes phase-aware.

2. **Phase-1 bound substitution** — one O(n_total) pass, writing into the *existing* `lo_ext`/`hi_ext` (the true values are already preserved in `lo_true`/`hi_true` at `_csparse.c:13885-13889`; extend that save to cover all `n_total` entries, not just structurals):
   - `!lo_fin && !hi_fin` → `[-1000, +1000]`
   - `hi_fin && !lo_fin` → `[-1, 0]`
   - `lo_fin && !hi_fin` → `[0, +1]`
   - both finite (incl. fixed, and all artificials, which are `[0,0]`) → `[0, 0]`

3. **Phase-1 nonbasic placement** — one O(n_total) pass, replacing the big-M block at `13891-13985`:
   - place every nonbasic at its dual-feasible end (`x_ext[j] = 0` for one-sided; `-1000` with `DS_BOUND_LO` for free);
   - compute `r_j` as today (`13922-13933`);
   - if `r_j` has the wrong sign for the assigned status, **flip** to the other end of the Phase-1 box (this is `correctDualInfeasibilities` + `flipBound`). Boxed and fixed columns are `[0,0]` and are never flipped — their dual infeasibility is not Phase 1's problem.
   - **Delete no code yet**: keep the big-M path as a fallback for the case where Phase 1 terminates with a nonzero objective.

4. **Run the existing dual simplex loop unchanged.** CHUZR already reads `lo_ext[basis[k]]`/`hi_ext[basis[k]]` (`_csparse.c:14281-14284`, `14742-14744`, `15224-15226`) and the BFRT already reads `hi_ext[j] - lo_ext[j]` (`14684`, `14897`), so the substitution propagates automatically — exactly as it does in HiGHS. Suppress the `has_art_bound` guards in Phase 1 (there are no artificial bounds).

5. **Phase-1 termination.** Phase 1 ends on the *same* condition Phase 2 uses for optimality: CHUZR finds no primal-infeasible basic variable. Then compute `Z₁ = Σ_{nonbasic j} x_ext[j] · r_j`.
   - `|Z₁| ≤ eps` → **the basis is dual feasible for the true bounds.** Restore `b`, restore `lo_true`/`hi_true`, re-derive nonbasic statuses (one-sided variables return to their true bound on the side their `bound_status` names; LP-boxed variables default to lower and are corrected by the first Phase-2 dual-feasibility repair sweep), recompute `x_B` once, and **enter the existing Phase-2 loop with no artificial bounds anywhere.** All of §8's certification tax (`15716-15804`, `14716-14728`) becomes dead code on this path.
   - `Z₁ < -eps` → the LP is dual infeasible (primal unbounded or infeasible). Fall back to the existing big-M path, or report `dual_infeasible` — either is honest, and this is exactly HiGHS's `assessPhase1Optimality` branch (`HEkkDual.cpp:2553-2654`).

6. **Do not add a Phase-1 entry test yet.** HiGHS decides between phases by counting dual infeasibilities *ignoring* those fixable by fixed/boxed flips (`HEkkDual.cpp:188-207`, `2341-2385`). The same count is trivially available from step 3: if no *one-sided or free* nonbasic has a wrong-signed `r_j`, skip Phase 1 entirely.

### 9.3 What this costs, and the invariants it preserves

Per solve: **three O(n_total) array passes** (save true bounds, substitute, place+flip), plus one extra `x_B` recompute at the phase boundary, plus one extra `y` BTRAN. No factorization (the crash basis is already factorized before section 3), no allocation beyond one zero buffer and one extended shadow-bound array, no matrix work, no pricing-structure rebuild. **This is the mechanism you could not find from the outside: there is no construction to amortise.**

Contract compliance: the substitution table is a **global rule keyed only on bound finiteness** — no per-problem tuning, no new thresholds (the `1000` for free variables is a global constant, as it is in HiGHS). `eps = 2e-5` is untouched. Optimality is still certified in original units against `lo_true`/`hi_true` by the existing exit check (`_csparse.c:15767-15804`) — indeed *more* cleanly, since the artificial-bound damage budget is no longer in the path. This must be implemented from this description, not transcribed: the table, the objective identity, and the `b := 0` adaptation are the whole specification.

---

## 10. Things I could not determine

I am flagging these rather than guessing:

1. **Whether HiGHS actually enters Phase 1 on presolved greenbea**, and how its ~3,334 pivots split between Phase 1 and Phase 2. The code tracks them separately (`info.dual_phase1_iteration_count` / `dual_phase2_iteration_count`, `HEkkDual.cpp:246-247`, `253-254`) while `iteration_count_` is incremented for both (`HEkkDual.cpp:2241`). Determining this needs a run with `output_flag` on, which is outside my brief. **If your 3,334 figure is a total, it already includes Phase-1 pivots** — worth confirming before any comparison.
2. **Whether `initialiseBound`'s omission of a `solve_phase` argument in `cleanup()` is a bug.** HiGHS's own comment says "No solve_phase term in initialiseBound is surely an omission — when cleanup called in phase 1" (`HEkkDual.cpp:1153-1155`). It does pass `solve_phase`, so I read the comment as stale, but I could not resolve the intent.
3. **Whether the `1000` free-variable box width is load-bearing or incidental.** It is a bare literal at `HEkk.cpp:2683-2684` with no comment justifying the magnitude; its only effect is to weight free-variable dual infeasibilities 1000× in the Phase-1 objective. I would carry it over as-is and not tune it.
4. **The practical frequency of the nonzero `shiftCost` branch.** The clamp at `HEkkDualRow.cpp:242-246` makes it reachable with a nonzero amount, but HiGHS's own comment near an adjacent path says it is "not executed by ctest" (`HEkkDual.cpp:741-744`, about a different branch). I could not determine how often the `theta_dual == 0` shift actually fires without instrumenting a run. I would treat it as **not** part of the Phase-1 cheapness story either way — see §5.2.
5. **Whether the loss of boxed-variable bound status across the Phase-1 boundary (§4.2) costs HiGHS pivots.** Mechanically it is real; whether it matters is a measurement I did not make.