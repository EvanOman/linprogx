# PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

# The answer: HiGHS's dual Phase 1 constructs nothing

**We spent the campaign asking "how does HiGHS build a good start cheaply?" The
answer is that it doesn't build one. It swaps two float arrays.**

## What we knew before reading the source (clean-room)

- HiGHS reaches greenbea optimality in ~3,334 pivot-equivalents; linprogx's cold
  dual simplex takes **4,399**.
- HiGHS's dual Phase 1 is **b-invariant** — established behaviourally, then
  proven mathematically by Fenchel duality (`dual_phase1_derivation_2026_07_18.md`).
- We could *reach* 3,334 pivots (and 2,399 from the K7 basis), but **every**
  construction cost **0.145–0.215 s** to build — always more than it saved.
- We named this the **conservation law**: `pivots × µs/pivot ≈ 0.38–0.40 s`
  across every start ever built.

The unanswered question was never *what* a good start looks like. It was *how
HiGHS gets one without paying for it*.

## The mechanism (HEkk.cpp, `HEkk::initialiseBound`)

Dual Phase 1 in HiGHS is **the same dual simplex, on the same matrix, with the
same costs, from the same basis and the same factorization** — with the primal
bound vectors replaced by a synthetic map:

| original bounds | Phase-1 bounds |
|---|---|
| free `(-inf, +inf)` | `[-1000, 1000]` |
| upper-only `(-inf, u]` | `[-1, 0]` |
| lower-only `[l, +inf)` | `[0, 1]` |
| boxed or fixed `[l, u]` | `[0, 0]` |

HiGHS's own comment states the theory:

> *"The dual objective is the sum of products of primal and dual values for
> nonbasic variables. For dual simplex phase 1, the primal bounds are set so
> that when the dual value is feasible, the primal value is set to zero.
> Otherwise the value is +1/-1 according to the required sign of the dual,
> except for free variables... Hence the dual objective is the negation of the
> sum of infeasibilities."*

`HEkkDual::solvePhase1()` (HEkkDual.cpp:573) calls
`initialiseBound(SimplexAlgorithm::kDual, solve_phase)` and then runs the
ordinary iteration loop. On completion, `initialiseBound(kDual, kSolvePhase2)`
restores the true bounds (HEkk.cpp:3576) and Phase 2 continues **from the same
basis**.

### Why it is free

- **No auxiliary LP.** No extra columns, no different matrix, no separate solve.
- **No refactorization.** The basis and its LU carry straight across.
- **No crossover.** Phase 2 resumes on the identical basis.
- **b never appears** in the Phase-1 bounds — which is exactly the b-invariance
  we proved from the outside without being able to explain it.

The entire cost is an **O(n) rewrite of two float arrays.**

### The load-bearing insight

Under the Phase-1 map **every variable is boxed with finite bounds**. Therefore
every nonbasic can *always* be placed on the side matching its reduced-cost
sign, so **dual feasibility is achievable by construction**. That is the whole
trick, and it is why no auxiliary problem is needed.

## What linprogx does instead — and why it costs 4,399 pivots

`_csparse.c` §3 "NONBASIC ASSIGNMENT for dual feasibility" (~line 13871) uses
**big-M artificial bounds**:

> *"when the reduced cost points toward an infinite bound... no dual-feasible
> placement exists. We add an artificial finite bound at distance M from the
> finite bound and place the variable there. At exit, if any artificial bound is
> active, we re-solve with a larger M."*

with `M = 1e5 × max(1, max|finite bound|, max|b_i|)`.

**greenbea is the worst possible instance for this.** Our own clean-room
sifting falsifier measured **3,611 of 3,868 columns (93.4%) as one-sided or
free** — so big-M applies to essentially every column, and the dual simplex must
traverse an enormous artificial box.

The campaign *suspected* this. The knob `LINPROGX_DS_BIGM_FACTOR` exists with
the comment *"probe whether big-M magnitude drives the path explosion on
one-sided-column-heavy instances"*. What it never found was the cheap
alternative.

**Tuning M does not fix it** (measured today):

| `LINPROGX_DS_BIGM_FACTOR` | pivots | status |
|---|---:|---|
| 1e5 (shipped) | 4,399 | optimal |
| 1e4 | 4,402 | optimal |
| 1e3 | 4,344 | optimal |
| 1e2 | — | **iteration_limit** (artificial bound becomes active) |
| 1e1 | — | **iteration_limit** |

The fix is structural, not a constant.

## Why we could not have found this blind

We *did* build the right mathematical object: the homogeneous auxiliary
`min c'x s.t. Ax=0, x ∈ [0,1]` is morally the same construction as HiGHS's
Phase-1 bound map. But we implemented it as a **separate LP solved from
scratch** (0.145–0.215 s) instead of an **in-place bound swap on the live
tableau** (O(n)).

The conservation law was never a law. It was an artefact of assuming Phase 1
must be *constructed* rather than *entered*.

## The change linprogx needs

Replace big-M entirely with an in-place dual Phase 1:

1. Save true bounds.
2. Overwrite bounds with the `{0, ±1, ±1000}` map above.
3. Place every nonbasic on the side matching its reduced-cost sign — always
   possible, since all Phase-1 bounds are finite.
4. Run the existing dual simplex loop until primal feasible w.r.t. Phase-1
   bounds.
5. Restore true bounds, recompute `x_B`, verify dual feasibility against the
   true bounds; if it fails, the LP is dual infeasible (primal unbounded).
6. Continue into the existing Phase 2 loop from the same basis and factorization.

This removes the big-M re-solve loop and its `bigM` scaling entirely.

## Attribution

Mechanism understood by reading HiGHS (MIT, ERGO-Code/HiGHS,
`highs/simplex/HEkk.cpp`, `highs/simplex/HEkkDual.cpp`). **No code was copied.**
The implementation in linprogx is written independently from this algorithm
description. Per `docs/PROVENANCE.md`, any greenbea result following from this
is a **source-informed** result and must never be reported as a clean-room one.

---

# PART 2 — the measured ground truth, and the correction to our central assumption

## HiGHS's actual numbers on greenbea (measured, not inferred)

The campaign's "HiGHS takes ~3,334 pivots" was **inferred** — it was linprogx's
*Phase-2* count starting from a HiGHS-derived basis. HiGHS's own iteration count
was never measured. Measured now via `highspy==1.14.0`:

```
Simplex iterations: DuPh1 1448; DuPh2 1376; PrPh2 12; Total 2836
```

| | iterations | wall (this box) | us/pivot |
|---|---:|---:|---:|
| HiGHS | **2,836** | 421.2 ms | 148.4 |
| linprogx | **4,399** | ~377 ms | **85.7** |

## Two corrections this forces

**1. HiGHS really does win on trajectory: 2,836 vs 4,399, −35.5%.** The inferred
3,334 was in the right region but pessimistic.

**2. linprogx's per-pivot cost is 1.73x BETTER than HiGHS's**, and on this box
linprogx is *faster in absolute wall* (377 ms vs 421 ms). The board loss exists
only on Modal's 4-vCPU containers, which matches the host-dependence measured
in the v3 paired cert (ratio varied 1.16–1.47 across three hosts of the same
class, and HiGHS's own wall varied 54%).

**This reframes the whole target.** Closing the pivot gap is worth **−35.5%**
against a **13.46%** bar — a 2.6x margin. Every clean-room wave was optimising
the slice where we were already ahead.

## What the bound swap alone does NOT do

Simulated end-to-end with the existing diagnostic hooks
(`experiments/phase1_bound_swap_probe.py`), Phase 1 solving `Ax = 0` under the
Phase-1 bound map and Phase 2 warm-started from its basis:

| | Phase 1 | Phase 2 | total |
|---|---:|---:|---:|
| linprogx two-phase (Dantzig) | 2,418 | 2,399 | **4,817** |
| linprogx two-phase (Devex) | 5,603 | 2,239 | 7,842 |
| linprogx two-phase (exact DSE) | 5,198 | 1,883 | 7,081 |
| linprogx big-M baseline | — | — | 4,399 |
| **HiGHS** | **1,448** | **1,376 (+12)** | **2,836** |

Two things stand out:

- The Phase-1 basis is genuinely excellent: Phase 2 from it costs **2,399**
  pivots, which is *exactly* the campaign's recorded "K7 2,399-pivot native
  basis". Independent confirmation that we had reached the right vertex before.
- But our Phase 1 costs 2,418 pivots where HiGHS's costs 1,448, and our Phase 2
  costs 2,399 where HiGHS's costs 1,376. **We are ~1.7x worse in BOTH phases.**

So the bound-swap structure is **necessary but not sufficient**. It removes
big-M and makes Phase 1 free to *enter*, but it does not by itself produce
HiGHS's trajectory.

## The remaining gap is pricing

HiGHS uses **dual steepest edge** throughout. Our recorded verdict that "exact
DSE measured worse" was obtained on the **cold big-M path** — a different
regime. DSE weight *initialisation* is the classic failure mode: from a logical
basis the initial weights are exactly 1, but from a crash basis they must be
computed or approximated, and a poor approximation makes DSE behave worse than
Dantzig. Our sweep above shows exactly that signature: DSE gives the *best*
Phase 2 (1,883, better than Dantzig's 2,399) but a terrible Phase 1 (5,198).

**DSE is helping where its weights are meaningful and hurting where they are
not.** That is an initialisation problem, not a rule problem.

## Implementation target (next wave)

1. Replace big-M with the in-place two-phase structure (bound swap, `Ax=0`
   Phase 1, restore bounds, continue on the same basis and factorization).
2. Implement dual steepest edge with **correct weight initialisation** for the
   starting basis, used in both phases.
3. Target: 2,836 pivots at linprogx's existing 85.7 us/pivot.

At our current per-pivot cost that projects to roughly `2,836 x 85.7us = 243 ms`
against HiGHS's 421 ms on this box — and against the board, a pivot reduction of
35.5% versus the 13.46% required.

## Attribution

Mechanism understood by reading HiGHS (MIT, ERGO-Code/HiGHS). Iteration counts
measured through the public `highspy` API. **No code copied.** Per
`docs/PROVENANCE.md` any resulting greenbea win is **source-informed**.
