<!-- Recovered from overmind worker 03e7917c transcript. -->

# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

I'll start by reading the mandatory provenance doc, then dig into the HiGHS source.

---

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

Source read: `/tmp/highs-study/HiGHS` @ `04024d70` (2026-07-02), MIT. No verbatim copying; identifiers and 1–3 word snippets only. Read-only session, no builds, no benchmarks, no timings measured.

---

# How HiGHS starts its dual simplex

## Headline: there is no crash, and there is no auxiliary LP

The two things linprogx assumed HiGHS must be doing, it does not do.

**There is no crash procedure in HiGHS.** A `simplex_crash_strategy` option exists (`lp_data/HighsOptions.h:916-920`) with a full enum of named strategies — LTSSF, Bixby, LTSF, priority variants (`simplex/SimplexConst.h:54-67`) — and it defaults to `kSimplexCrashStrategyOff` (`HighsOptions.h:557`). I grepped the entire `highs/` tree: `simplex_crash_strategy` is **read by nothing**. It appears only in its own declaration, its default initialiser, and its option-record registration. There is no `HCrash` class, no crash source file, no call site. The enum is a vestige of hsol. The only surviving acknowledgement that a crash could exist is a comment (`simplex/HEkkDual.cpp:101`) reasoning about "a logical or crash basis." `ICrash` (`presolve/ICrash.*`) is an unrelated research feature, default off (`HighsOptions.h:642`), and feeds a solution vector, not a simplex basis (`lp_data/Highs.cpp:1412-1442`). `ipx/basis.h`'s `CrashBasis` belongs to the interior-point crossover, not the simplex.

**There is no auxiliary or homogeneous LP.** State this plainly, because it was the load-bearing wrong assumption: HiGHS never constructs a second problem, never adds artificial columns, never allocates a second solver object, and never re-factorizes to enter dual Phase 1. Dual Phase 1 is the *same* LP, the *same* basis, and the *same* `B⁻¹`, with the variable **bounds temporarily overwritten by a constant table**. That is a single O(n+m) pass over an array (`simplex/HEkk.cpp:2676-2696`). That is the "construction cost we could never beat" — it is essentially zero because there is nothing to construct.

---

## 1. What basis does HiGHS actually start from?

The **logical (slack) basis**: every logical variable basic, every structural nonbasic. `HEkk::setBasis()` (`simplex/HEkk.cpp:1122-1172`) sets `nonbasicFlag_[iCol] = true` for all structurals (`:1131`) and `basicIndex_[iRow] = num_col + iRow` for all rows (`:1163-1169`), then records `num_basic_logicals = num_row` (`:1170`). So **B = I** exactly.

It is reached by omission, not by choice: `HEkk::initialiseForSolve()` (`:1595-1596`) calls `initialiseSimplexLpBasisAndFactor()`, whose first act is `if (!status_.has_basis) setBasis();` (`:1472`). A basis is used instead only if one was supplied — from `Highs::setBasis`, from a MIP node, or from postsolve — via the `HighsBasis` overload (`:1175-1249`) invoked at `simplex/HApp.h:198`.

The one non-trivial part of `setBasis()` is not *which* columns are basic (that is fixed) but **which bound each nonbasic structural sits at** (`HEkk.cpp:1130-1162`):

- fixed (`lower == upper`) → `kNonbasicMoveZe`
- boxed (both finite) → **the bound closer to zero** (`fabs(lower) < fabs(upper) ? Up : Dn`)
- lower-only → `Up` (at lower); upper-only → `Dn` (at upper)
- free → `kNonbasicMoveZe`, value 0

Note what this heuristic is *not*: it does not look at costs or reduced costs. It is a **primal**-magnitude heuristic (keep `x_N` small, hence keep `x_B` small). Dual feasibility is handled entirely downstream.

Rank deficiency cannot occur for B = I, but the machinery exists for supplied bases: `computeFactor()` returns a deficiency count (`:1508`) and `handleRankDeficiency()` (`:1538-1567`) swaps each pivotless column out for the logical of its uncovered row and records it as a taboo basis change.

## 2. Is there a crash? What does it cost?

No — see above. **Cost: zero.** The cost of the start is `setBasis()`'s single O(n+m) loop, plus `computeFactor()` on the identity, plus the O(n+m) initialisation pass in `initialiseForSolve()` (`:1595-1614`: cost copy, bound copy, `initialiseNonbasicValueAndMove`, one `computePrimal`, one `computeDual`).

For the dual simplex specifically, the identity basis buys something a crash basis cannot: **exact dual steepest-edge weights, free.** `HEkkDual::solve` assigns `dual_edge_weight_.assign(num_row, 1.0)` (`HEkkDual.cpp:146`), then:

```
if (ekk_instance_.logicalBasis()) {        // HEkkDual.cpp:153
  status.has_dual_steepest_edge_weights = true;   // :155  — unit weights ARE exact for B=I
} else if (near_optimal) {
  edge_weight_mode = EdgeWeightMode::kDevex;      // :166  — downgrade rather than pay
} else {
  ekk_instance_.computeDualSteepestEdgeWeights(true);  // :171 — m full BTRANs
}
```

`logicalBasis()` is `HEkk.cpp:4047-4052`. `computeDualSteepestEdgeWeights` (`HEkk.cpp:2056-2067`) loops `iRow = 0..num_row-1` doing one `btranInScaledSpace` of a unit vector each (`:2079-2091`) — **m triangular solves** before the first pivot. The default edge-weight strategy is `Choose` (`HighsOptions.h:922-930`), which `interpretDualEdgeWeightStrategy` (`HEkkDual.cpp:2300-2305`) resolves to **steepest edge with a permitted switch to Devex**.

So the logical basis is not merely cheap. It is *the unique basis for which exact DSE pricing costs nothing*. Any crash forfeits that: you pay m BTRANs, or you drop to Devex.

## 3. How is dual feasibility established?

**Neither assumed nor repaired at the start — it is made structurally unnecessary by a bound substitution, and any residue is absorbed by flips and cost shifts.** Four mechanisms, in order:

**(a) Measure, on unperturbed costs.** `HEkkDual::solve` computes duals with costs *not* perturbed and counts infeasibilities (`HEkkDual.cpp:60-66`). `force_phase2` is set if the max infeasibility is small enough that `maxInf² < dual_feasibility_tolerance` (`:69-71`) — i.e. Phase 1 is skipped when the violation can be shifted away. Near-optimality (`:108-110`, dual-feasible plus <1000 primal infeasibilities with max <1e-3) suppresses cost perturbation entirely (`:122`).

**(b) The Phase 1 bound substitution — this is the mechanism.** `HEkk::initialiseBound(kDual, kSolvePhase1)` (`HEkk.cpp:2676-2696`) replaces every variable's bounds by a constant depending only on which of its true bounds are finite:

| true bounds | Phase-1 bounds |
|---|---|
| free (−∞, +∞) | `[-1000, 1000]` (`:2683`) |
| upper only (−∞, u] | `[-1, 0]` |
| lower only [l, +∞) | `[0, 1]` (`:2689`) |
| boxed or fixed | `[0, 0]` (`:2692`) |

Costs are untouched. The matrix is untouched. The basis is untouched. The factorization is untouched. `initialiseNonbasicValueAndMove()` (`HEkk.cpp:2716-2775`) then re-places each nonbasic at a Phase-1 bound.

**Why this is free — the b-invariance you derived, confirmed in code.** HiGHS's internal form is homogeneous. Logical `n+iRow` carries bounds `[-row_upper, -row_lower]` (`HEkk.cpp:2431-2440`), so the constraint is `Ax + s = 0`. `computePrimal` (`HEkk.cpp:2917-2952`) accumulates `Σ aⱼ·workValue_[j]` over nonbasics, FTRANs, and sets `baseValue_[i] = -primal_col.array[i]` (`:2940`) — i.e. `x_B = −B⁻¹ N x_N`. **There is no `b` term.** The right-hand side is identically zero, so *any* change to nonbasic bounds is exactly a change to `x_N`, and nothing else in the model moves. The Phase-1 problem is the same LP with a different `x_N`, and it shares `B⁻¹` bit-for-bit.

The bound table is chosen so a dual-feasible nonbasic sits at value 0 and a dual-infeasible one at ±1. Hence the Phase-1 dual objective `Σ_{j∈N} xⱼ dⱼ` **is the negated sum of dual infeasibilities**, and reaching zero *is* dual feasibility for the true bounds (`HEkkDual.cpp:690-703`, and the derivation comment at `HEkk.cpp:2657-2665`). No artificial variables, no big-M, no separate objective vector.

**(c) Flip / shift, inside every rebuild.** `correctDualInfeasibilities` (`HEkkDual.cpp:2388-2540`, called from `rebuild` at `:1082`) walks all nonbasics and removes each infeasibility by one of two moves:
- **flip** for fixed variables always, and for boxed variables unless `force_phase2` (`:2441-2444`, `HEkk::flipBound` at `HEkk.cpp:3069`) — free of cost distortion;
- **cost shift** otherwise: set `workDual_[j] = ±(1 + random.fraction()) · dual_feasibility_tolerance` and add the delta to `workCost_[j]` (`:2483-2494`).

Under Phase-1 bounds *every* variable is boxed or fixed, so this pass restores dual feasibility **entirely by flips** — an O(n+m) sign assignment, no cost distortion at all. Free variables are only counted, never corrected (`:2429-2435`).

**(d) Phase decision.** `dualInfeasCount > 0 ? kSolvePhase1 : kSolvePhase2` (`HEkkDual.cpp:206`, re-evaluated at `:231`), using `computeDualInfeasibilitiesWithFixedVariableFlips` (`:2341-2386`) which drives off `nonbasicMove` and treats fixed variables as costless.

**Exit.** Zero Phase-1 dual objective → Phase 2 (`:703`). Nonzero → `assessPhase1Optimality` (`:2553-2601`) removes perturbation via `cleanup()` and re-tests; genuine residual infeasibility yields `kUnboundedOrInfeasible`. On the way out, true bounds are restored and moves recomputed (`:797-798`), and `exitPhase1ResetDuals` (`:2656-2700`) shifts free variables' costs so their duals are exactly zero.

## 4. How does scaling interact with the start?

Scaling happens **before** the basis is chosen and before the LP is moved into the simplex kernel: `considerScaling(options, incumbent_lp)` at `simplex/HApp.h:171`, then `ekk_instance.moveLp(...)` at `:194`, then `setBasis` at `:198`. The dual simplex therefore solves the **scaled** LP, and the logical basis is B = I in scaled space (row scaling doesn't disturb that — a scaled logical column is still a scaled unit vector, and HiGHS's NLA carries the scaling separately).

Scaling does not change *which* basis starts. It changes the **duals of that basis**, and therefore the Phase-1/Phase-2 decision and the whole subsequent path.

Mechanism (`lp_data/HighsLpUtils.cpp:966-1063`, `1064-1400`), default strategy = `Equilibration` (`HighsOptions.h:906-913`):

1. **Skip gate:** if all `|Aᵢⱼ| ∈ [0.2, 5]`, no scaling at all (`:990-1005`).
2. **Geometric-mean equilibration, not Ruiz.** 6 alternating passes (`:1112`); column factor `1/√(colMin · colMax)` (`:1128`), then row factor `1/√(rowMin · rowMax)` (`:1142`). Ruiz divides by `√‖·‖∞`; HiGHS uses the geometric mean of min and max.
3. **Costs folded in** when `min |cⱼ| < 0.1` (`:1088-1091`).
4. **Factors clamped** to `2^±allowed_matrix_scale_factor`, default 2^±20 (`HConst.h:359`).
5. **Rounded to exact powers of two**: `pow(2.0, floor(log(s)/log2 + 0.5))` (`:1159`, `:1163`).
6. **Accept/reject.** Compute mean-equilibration, extreme-equilibration and max/min-value improvement factors; if the product `< 1.0`, **undo the scaling entirely** (`:1330-1349`). Note: with the `Choose` default resolving to `ForcedEquilibration` (`:987`), `possibly_abandon_scaling` is false — the reject gate is *disabled* on the default path.

Point 5 is the one that should interest you. Power-of-two factors are exact in binary floating point, so scale and unscale are **round-trip lossless**. linprogx's Ruiz factors are `1/√‖·‖∞` — arbitrary reals — which is why `_csparse.c:13465-13501` had to introduce a conditioning gate (`max/min ≥ 100`) to *avoid* Ruiz on well-balanced matrices: "the round-trip scale/unscale introduces floating-point error that can exceed tight absolute tolerances." HiGHS does not need that gate, because rounding to powers of two removes the error rather than avoiding the operation. This is a strictly better trade and it is one line.

## 5. What crosses the presolve → simplex boundary?

**Just a reduced model. No basis, no factorization, no warm start.** The reduced LP path (`lp_data/Highs.cpp:1625-1694`) explicitly calls `ekk_instance_.clear()` before `solveLp(reduced_lp, ...)` (`:1657-1665`), so the presolved LP is solved from a fresh logical basis. `HPresolve` produces no `HighsBasis` — its only basis contact is inside a debug routine (`presolve/HPresolve.cpp:8142-8178`). Postsolve maps a basis *backwards* (`presolve/HighsPostsolveStack.h:90-179`), never forwards.

The one thing that does cross forward is a scalar: the factorization pivot threshold reached while solving the presolved LP is carried to the original-LP solve (`Highs.cpp:1667-1670`, `:1893-1894`).

The **postsolve → simplex** boundary is different and is the only genuine warm start in the LP path: postsolve's recovered basis is completed by `refineBasis(incumbent_lp, solution_, basis_)` (`Highs.cpp:1897`) because it is only basic/nonbasic and EKK needs bound statuses; EKK data from the reduced solve is invalidated (`:1899`); the original LP is then re-solved from that basis, typically in a handful of pivots (`:1918-1923`).

## 6. Auxiliary / homogeneous LP?

**No.** Restating for the record, since linprogx assumed the opposite: HiGHS solves exactly one LP. Dual Phase 1 is a bound-table substitution on that LP (`HEkk.cpp:2676-2696`), sharing basis, factorization, cost vector and matrix with Phase 2. The transition into Phase 1 costs one O(n+m) array pass and one O(n+m) flip pass; the transition out is the same in reverse (`HEkkDual.cpp:797-798`). Nothing is allocated, nothing is factorized, nothing is solved on the side.

---

## Contrast: what linprogx does instead

| | HiGHS | linprogx |
|---|---|---|
| Start basis | logical, B = I (`HEkk.cpp:1163-1170`) | singleton-cascade triangular crash + artificial fill (`_csparse.c:13645-13744`) |
| Extra columns | none — the m logicals are model variables (`HEkk.cpp:2431`) | **m artificial columns appended**, cost 0, bounds [0,0], `n_total = n+m` (`_csparse.c:13616-13620`) |
| Initial factorization | identity | full LU of the crash basis (`ds_factorize_basis`, `_csparse.c:13757`) + up to `16 + m/100` singularity repairs, each a refactorization (`:13778-13830`) |
| Initial DSE weights | exact, free, because B = I (`HEkkDual.cpp:153-155`) | not maintained on the shipped path; row choice is **plain max-violation** (`leaving_rule=1`, `sparse.py:240`, `_csparse.c:14381-14383`). The DSE arm exists but must pay m BTRANs (`_csparse.c:13840-13863`) |
| Dual feasibility | Phase-1 bound table `{[0,0],[0,1],[-1,0],[-1000,1000]}`, O(n+m), b-invariant because RHS ≡ 0 (`HEkk.cpp:2676-2696`, `:2940`) | **big-M artificial bounds**, `M = 1e5 · max(1, max|bound|, max|b|)`, installed per-variable whenever the reduced cost points at an infinite bound; re-solve with larger M if an artificial bound is active at exit (`_csparse.c:13880-13980`) |
| Residual dual infeasibility | flip (fixed/boxed) or cost shift to `±(1+rand)·tol` (`HEkkDual.cpp:2441-2494`) | absorbed into the big-M placement; optional cost shifting behind `cost_shift_on` (`_csparse.c:14094-14098`, `:15048-15080`) |
| Scaling | geometric-mean equilibration, 6 passes, **rounded to powers of 2**, always-on gate `[0.2,5]` (`HighsLpUtils.cpp:1112-1166`) | Ruiz inf-norm, 10 passes + one ℓ2 pass, **not** power-of-2 rounded, gated off unless row-norm ratio ≥ 100 to dodge round-trip error (`_csparse.c:13465-13540`) |
| Presolve handoff | reduced model only; pivot threshold is the sole scalar carried (`Highs.cpp:1657-1670`) | reduced model only — **same design** |

Three things fall out of this comparison.

**The big-M is doing the job of HiGHS's Phase-1 bounds, at ~1e5× the magnitude.** Both establish dual feasibility by placing each nonbasic at the bound matching its reduced-cost sign, inventing a finite bound where none exists. HiGHS invents `±1` (`±1000` for free). linprogx invents `1e5 · scale`. Both are exact under b-invariance. But the invented bound is the *step length* the dual ratio test will take when that variable enters, and a bound five orders of magnitude wider means correspondingly wilder primal excursions, more re-infeasible rows, and a longer path back. The `LINPROGX_DS_BIGM_FACTOR` knob at `_csparse.c:13903-13909` exists because someone already suspected this ("probe whether big-M magnitude drives the path explosion on one-sided-column-heavy instances"). The HiGHS source says: it does, and the right magnitude is 1.

**The crash is buying coverage and paying for pricing.** linprogx's crash is well-built — the singleton cascade is provably nonsingular, the stability filter is principled, magnitudes are read from the equilibrated matrix. It is a better crash than the one HiGHS deleted. But it lands on a basis where exact DSE costs m BTRANs, so the shipped path runs Dantzig row selection instead. HiGHS starts on the one basis where exact DSE is free and runs it from pivot one. Dual steepest edge versus max-violation is a large, well-established difference in dual pivot counts, and it is the most plausible structural account of 3,334 versus 4,399 that the source supports. I did not measure this and am not asserting the attribution — I am saying it is the mechanism the code exposes.

**Every alternative start you built cost 0.145–0.215s because you were building a start.** HiGHS's start is not an alternative construction that happens to be cheap. It is the absence of a construction: the basis is `basicIndex_[i] = num_col + i`, the factorization is skipped, the weights are `1.0`, and the dual-feasibility fix-up is a table lookup over an array. There was never a cheap construction to find, which is exactly why you could not find one.

---

## The minimal change linprogx would need

Ordered by ratio of expected effect to risk. Each is independently shippable and independently falsifiable.

**1. Replace big-M with the ±1 Phase-1 bound table.** Highest value, smallest diff, and it is the mechanism you derived but priced wrong. In the dual-feasibility block (`_csparse.c:13871-13985`), stop synthesising `lo/hi` at distance `M` and instead substitute the constant table by finiteness class — `[0,1]` lower-only, `[-1,0]` upper-only, `[0,0]` boxed/fixed, `[-1000,1000]` free — saving the true bounds, placing every nonbasic at the bound its reduced-cost sign selects, and running the dual simplex on that until the Phase-1 dual objective (`Σ xⱼ rⱼ` over nonbasics) reaches zero. Then restore true bounds, re-place nonbasics, and continue. linprogx's model is `Ax = b` with `b ≠ 0`, so you do **not** get HiGHS's exact b-invariance for free — but the substitution only touches `x_N`, and `x_B = B⁻¹(b − N x_N)` is one FTRAN you already do at start-up. The construction cost is one O(n+m) pass. This also deletes the "re-solve with larger M" retry path.

**2. Round the Ruiz factors to powers of two.** One line at the end of the Ruiz loop (`_csparse.c:13506-13527`): `s = pow(2.0, floor(log2(s) + 0.5))` for every row and column factor. Scale/unscale becomes exact, the round-trip error that forced the `ratio ≥ 100` gate (`:13496-13501`) disappears, and you can then re-test whether the gate is still earning its keep. Preserves your 10-pass Ruiz — this is not a switch to geometric-mean equilibration, and I would not recommend switching, since I have no evidence the 6-pass geometric-mean rule is better than 10-pass Ruiz.

**3. Only then, reconsider the crash.** With ±1 bounds in place, measure the crash basis against a plain identity/all-artificial start under the *same* pricing rule. If the crash still wins, keep it. If it does not, dropping it unlocks exact DSE weights at zero initialisation cost (`devex_w[k] = 1.0` is exact when B is the artificial identity), which lets you run `leaving_rule=5` on the shipped path without paying `_csparse.c:13840-13863`. That is the configuration HiGHS actually ships. Note this is a genuine fork in the road, not a strict improvement: your crash is real work that a slack basis does not do, and the trade must be measured, not assumed.

Order matters: do (1) before evaluating (3), because big-M and crash quality interact — a wide invented bound punishes a good crash basis as hard as a bad one.

---

## Flagged / not determined

- **I did not verify any of this by running HiGHS.** No builds, no benchmarks, no timings, per instruction. Every claim above is static reading of `04024d70`.
- **No timing or pivot-count attribution is asserted.** The DSE-versus-Dantzig account of 3,334 vs 4,399 is a mechanism the source exposes, not a measured cause. It needs a controlled experiment.
- **I did not determine which HiGHS build/options the campaign measured against.** Defaults are as cited; if `tools/modal_bench.py` overrides `simplex_dual_edge_weight_strategy`, `simplex_scale_strategy`, or `presolve`, parts of §2 and §4 shift.
- **I did not trace the parallel dual variants** (`HEkkDualMulti.cpp`, PAMI/SIP). `chooseSimplexStrategyThreads` (`HEkk.cpp:1051`) can select them; I read only the serial path. The start-basis logic is shared, but I did not confirm the Phase-1 handling is identical under multi.
- **I did not confirm b-invariance transfers to linprogx's `Ax = b` form.** HiGHS's RHS is identically zero (`HEkk.cpp:2940`) which makes the bound substitution free; linprogx's is not. My reading is that this costs one FTRAN at the substitution and nothing per-iteration, but that is an inference from linprogx's structure, not something I verified by reading its ratio test.
- **`HEkk::dualize()` / `permute()`** (`HEkk.cpp:452`, `:996`) run only from a logical start and are **off by default** (`HighsOptions.h:1526-1536`). I did not read them; if you ever enable HiGHS's dualize for comparison, §1 changes.
- **`kDebugMipNodeDualFeasible` / `debug_dual_feasible`** (`HEkkDual.cpp:75-92`) is MIP-node-only machinery; not relevant to a cold LP solve, and I did not trace it.