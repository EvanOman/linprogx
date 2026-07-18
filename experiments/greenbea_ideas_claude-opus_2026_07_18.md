# greenbea idea classes — claude-opus — 2026-07-18

YOURTAG: `claude-opus`.

## Framing that governs every idea below

Two measured facts reshape the search space before any idea:

1. **The route change is dead on wall-clock, not just on the certificate.**
   The IPM stall doc measures 0.747 / 0.701 / 0.675 s to reach the good
   primal at iteration 58, at ~11.5 ms/iteration. Even a *free* dual
   certificate leaves the IPM at ~0.68 s versus HiGHS 0.266 s. The dual
   residual is still 1.2e-3 (worst violation 4.3e-5 > 2e-5) at iteration 30,
   so you cannot certify early enough to fit a ~0.25 s budget. PDHG is already
   killed. Therefore **the only live route is dual simplex**, and every idea
   here lives or dies on `wall = pivots x us/pivot`.

2. **The trade-against evidence is specifically about *transferred* trajectories.**
   Warm-starts, phase-1 transfer, and IPM crossover densify because they drop
   our solver onto a *foreign* basis manifold whose factors carry more fill.
   Solve-side mechanisms that keep our own pivot sequence unchanged are
   **trajectory-preserving** and are therefore immune to that trade-off. This
   is the load-bearing distinction the campaign has not exploited: you can cut
   `us/pivot` without paying the pivot-count penalty, because you never leave
   your own trajectory. Consequently **no single idea below hits -41% alone**;
   the win is a *stack* of one trajectory-preserving solve-side cut with one
   native (non-transferred) pivot trim. I make that arithmetic explicit per
   idea and in the self-assessment.

Reference arithmetic (local quiet-box, from the dossier): ours 4,399 pivots x
95 us/pivot = 0.418 s; HiGHS 0.24 s; flip target -41% = 0.248 s. Per-pivot
wall shares: pivot-row/PRICE 24.8%, BTRAN 18.9%, FTRAN 17.9%, ratio 14.9%,
reduced-cost update 9.7%, LU update 6.1%, refactor 5.5%. Solve+price+rc terms
that a numeric/layout idea can touch = 17.9+18.9+24.8+9.7 = **71.3%**.

Structural check I ran locally (main-tree venv, read-only, no C build):
raw greenbea `A` has **87% of columns with row-spread < 10% of m** (strong
staircase locality) but col row-spread p99 = 2223 ~ m, i.e. a **dense coupling
border**; RCM cuts AAt bandwidth only 2363 -> 1422; one dominant connected
component. This is a bordered-staircase energy model, and it is *why* BTRAN
`rho` is p50 897/1525 = 59% dense and the priced pivot row is 3,625/3,868 =
94% dense. That density fact is the hinge for ideas 1, 2, and 3.

---

## Idea 1 (the bet) — Mixed-precision DS body with an fp64 certificate tail

**Mechanism.** Run the entire dual-simplex inner loop — BTRAN, PRICE
(`N^T rho`), FTRAN, and the reduced-cost update — in fp32, and keep fp64 only
for (a) the DSE/pricing weights, (b) a single refined recomputation of the
*chosen* pivot element and its row/column before the basis update commits, and
(c) the final certificate. The enabling gift is that `eps = 2e-5` is loose:
fp32 carries ~1.2e-7 relative precision, ~2 orders of margin over the
acceptance gate, so the *bulk* linear algebra does not need fp64. Because the
priced row is 94% dense and `rho` is 59% dense, these are contiguous-ish,
memory-bound kernels that get near-ideal fp32 SIMD/BLAS speedup (~1.8x from
halved memory traffic and doubled vector width), not the ~1.4x that
hyper-sparse gather/scatter would give. Selection quantities (ratios, pricing
scores) are computed in fp32 but the *winner* is re-evaluated in fp64 so a
noisy fp32 ratio cannot commit an unstable pivot. A cheap fp64 residual check
on `B alpha - a_q` every k pivots (k global, e.g. 64) triggers an fp64
refactorization if drift exceeds a fixed floor. The certificate at the end is
recomputed entirely in fp64 against the exact basis, so honest optimality is
never fp32-dependent. This is a global mechanism with one fixed threshold, not
per-problem tuning.

**Wall term + ceiling.** Attacks `us/pivot` on 71.3% of the per-pivot wall.
At 1.8x on those terms: 71.3% -> 39.6%, so us/pivot 95 -> 95*(0.287 +
0.713/1.8) = **64.9 us/pivot**, i.e. **-32% alone** (0.418 -> 0.286 s). Not a
flip by itself. Stacked with a native pivot trim to HiGHS-parity 3,309 (idea 4):
3,309 x 64.9 = **0.215 s = -49% -> FLIP**. Even with no pivot help, at HiGHS's
own 86 us/pivot-equivalent workload this is the cheapest -32% on the board.

**Not a closed axis.** Nearest closed items are the per-pivot kernels (Suhl,
Forrest-Tomlin, dense-U FTRAN, cross-pivot reuse). Every one of those changes
*which arithmetic* is done or *reuses values*; none changes the *precision* of
the arithmetic. Precision is an untouched axis, and it is orthogonal — it
composes with all of them. It is explicitly not the killed "dense-U FTRAN":
the data type changes, the sparse structure does not.

**Falsifier-first probe (cheap).** Before touching the solver, microbench fp32
vs fp64 on greenbea-shaped kernels: (i) triangular solve against the actual
basis factor at pivot ~2000, (ii) `N^T rho` SpMV. Kill idea if measured fp32
speedup on these is < 1.5x. Second, run a *simulation* harness: drive the
existing fp64 DS but round `rho`, the priced row, and the ratios to fp32
before each selection (keeping the fp64 basis exact), and count pivots and
certificate residual. Kill if pivot count rises > 5% or the run fails to
certify at 2e-5 — that would mean fp32 selection noise perturbs the trajectory
enough to erase the gain.

**Risks.** Numerics: fp32 pricing could pick a marginally worse leaving
variable and lengthen the trajectory (the falsifier's job is to catch this).
Certificate: must stay fp64 and be recomputed against the exact basis — no
fp32 value may leak into acceptance. Constraint compliance: the k-pivot fp64
residual gate must fail *closed* (force refactor) so accumulated fp32 error
never silently violates feasibility.

---

## Idea 2 — Bordered-block / Schur-complement basis factorization

**Mechanism.** greenbea is a bordered-staircase model: 87% of columns are
locally supported, tied together by a thin dense border. A standard AMD/RCM
factorization ignores that and lets the border smear fill across `B^{-1}`,
which is exactly why `rho` is 59% dense and the priced row 94% dense — dense
solve vectors are the *symptom* of border-induced fill, not an intrinsic
property. Detect the block-plus-border partition once (hypergraph/net
partition of the columns; RCM alone is insufficient — measured 2363->1422
only), then factor the basis as a bordered-block system: cheap sparse factors
on the local diagonal blocks and a small dense Schur complement for the border
coupling. BTRAN/FTRAN then solve the blocks independently (sparse, local) and
correct through the small Schur factor, so the *block* part of `rho` stays
sparse and only the border rows fill. Maintain the partition across pivots and
route Forrest-Tomlin eta updates into the block they touch, reserving Schur
refactorization for border-touching pivots. This is trajectory-preserving:
same pivot sequence, cheaper factors.

**Wall term + ceiling.** Attacks `us/pivot` on the three solve terms (61.6%)
plus, indirectly, LU/refactor (11.6%). If effective solve-vector density drops
from ~59-94% to ~35%, solve cost scales ~0.6x on 61.6%: us/pivot 95 ->
95*(0.384 + 0.616*0.6) = **71.6, -25% alone** (0.418 -> 0.316 s). Stacked with
idea 4's 3,309 pivots: 3,309 x 71.6 = **0.237 s = -43% -> FLIP** (thin).
Stronger if it composes with idea 1's fp32 on the now-sparser factors.

**Not a closed axis.** Nearest closed item is presolve depth (aggregation
proven to break pivot parity both ways) and the shipped block-row uplook gate.
This proposes *no reductions* and changes *no pivot sequence*: it is a pure
permutation-plus-factorization-structure change to the numeric linear algebra.
The shipped uplook gate is a per-pivot heuristic; this is a global bordered
factorization with a Schur complement, a different object.

**Falsifier-first probe (cheap).** No solver change first. Take the actual
basis at pivots ~500 / ~2000 / ~4000 (dump the column index sets from an
instrumented run, or approximate with the optimal basis), and compute symbolic
LU fill under (a) current ordering vs (b) a block+border ordering from a net
partition of those columns. Kill if fill / factor nnz does not drop >= 25% at
all three snapshots — no fill drop means no solve-density drop means no gain.

**Risks.** High-risk area per the contract (factorization + updates). Dynamic
Forrest-Tomlin updates into a bordered factor are genuinely hard; a border
that grows as pivots progress erases the benefit (the probe's fill-vs-pivot
trend catches this). Numerics: the Schur complement can be ill-conditioned if
the border couples strongly; guard with an fp64 Schur factor and a condition
check that falls back to a full refactor.

---

## Idea 3 — Support-stability-driven adaptive memory reordering (contiguity, not reuse)

**Mechanism.** The 96-99.97% consecutive-pivot support overlap has been read
only as a *value/symbolic reuse* opportunity, and that reading is dead: values
change fully each pivot (different `e_r` per BTRAN), and symbolic reuse is weak
because the vectors are 59-94% dense (little DFS to amortize). But overlap has
a third, untapped payoff: **memory layout**. If the union support of `rho`,
the priced set, and the ratio candidates is ~stable for ~30 pivots, permute
the working copies of `B`, `N`, and the dual vector *once* so that hot support
is **contiguous** in memory, then run the memory-bound solves over dense
contiguous ranges with unit-stride SIMD instead of scattered indirect
gather/scatter. Detect support drift with a cheap running Jaccard on the
nonzero pattern and re-permute only when drift crosses a fixed threshold
(amortized every ~20-30 pivots). Sparse simplex solves are latency-bound on
indirect addressing; contiguity converts scattered loads into streamed loads.

**Wall term + ceiling.** Attacks `us/pivot` on the 71.3% solve+price+rc terms
via cache/SIMD efficiency, ~1.3-1.5x realistic on memory-bound kernels:
71.3/1.4 = 50.9, us/pivot 95 -> 95*(0.287 + 0.509) = **75.6, -20% alone**
(0.418 -> 0.335 s). Its real value is as a **multiplier that stacks under
ideas 1 and 2** (contiguous fp32 is faster than scattered fp32; contiguous
block factors stream better), so its marginal contribution to the stack is
larger than its standalone number.

**Not a closed axis.** Nearest closed item is cross-pivot reuse (killed
because `alpha'_k` needs the old `alpha_k`). That kill is about reusing
*numeric values*; this reuses *neither values nor the symbolic DFS* — it reuses
the *physical memory arrangement* of a support set whose identity is stable
even though its contents are recomputed from scratch every pivot. Different
object, different failure mode.

**Falsifier-first probe (cheap).** Instrument a run to dump the per-pivot
nonzero patterns of `rho` and the priced set; confirm the *union* support over
a 30-pivot window is < ~1.1x the per-pivot support (i.e. drift is genuinely
slow at the window level, not just pairwise). Then microbench a contiguous vs
scattered SpMV/triangular solve at the measured densities. Kill if either the
30-pivot union support blows up past ~1.3x per-pivot (overlap is pairwise-only,
useless for a window layout) or the contiguity speedup is < 1.2x.

**Risks.** The re-permutation cost must stay well under the amortized savings;
a fixed drift threshold that mis-tunes could re-permute too often. Numerics are
unaffected (pure permutation). Constraint compliance unaffected (trajectory
preserved).

---

## Idea 4 — Lagged / perturbed dual pricing *dynamics*, with a HiGHS-edge probe

**Mechanism + probe (this is both).** On identical input HiGHS does 3,309
pivots; our exact Forrest-Goldfarb DSE does 4,675 (worse than our own Dantzig
4,399). So HiGHS's edge is *not* the textbook DSE rule — the closed
"leaving-rules" family tested only *exact, freshly-updated* rules. The
untested hypothesis is that the edge is in the *dynamics*: stale/lagged edge
weights or a tiny bound/cost perturbation that reshapes the trajectory.
Concretely, run two probes on our own DS (staying on our sparse manifold, so
no transfer densification): (P1) update DSE weights only every k pivots
(lagged/approximate reference weights) rather than exactly each pivot; (P2)
apply a fixed global Wolfe-style cost perturbation before phase 2 and remove it
at the end. Both are single-global-knob mechanisms, not per-problem tuning.
The mechanism claim: an *approximate* edge weight can beat both Dantzig and
exact DSE here (as HiGHS's does) while being cheaper per pivot, and because it
is native it does not densify the solves. This is the one idea that attacks the
**pivot** term, and it is the pivot-side partner every solve-side idea needs.

**Wall term + ceiling.** Attacks `pivots`. Best case: reach HiGHS-parity 3,309
(-25% pivots). Alone at 95 us/pivot: 3,309 x 95 = **0.314 s, -25%** — not a
flip, and it may raise us/pivot slightly (lagged weights can be dense to carry).
Its role is to supply the pivot-side factor of the stack; combined with idea 1
it flips (0.215 s). If the probe reveals the edge is elsewhere, its *diagnostic*
value stands: it localizes the 1,090-pivot gap behaviorally without any source.

**Not a closed axis.** Nearest closed item is "leaving rules (family closed:
Dantzig, exact FG-DSE, Devex, rule2-4)." Every entry there is an *exact rule
evaluated fresh each pivot*. Weight *staleness dynamics* (a rule applied to
deliberately lagged weights) and a global cost *perturbation* are different
degrees of freedom — the dossier itself lists "perturbation? phase structure?
ratio-test interplay?" as the *unidentified* machinery. This probes exactly
that unidentified region.

**Falsifier-first probe (cheap).** Run P1 for k in {8,32,128} and P2 at one
fixed perturbation scale; record pivot count *and* solve-vector density
(to confirm no densification). Kill the pivot-reduction claim if neither drops
below ~4,000 pivots, or if any drop is bought by densification that raises
projected wall. On a kill, escalate to instrumenting our phase-1 vs phase-2
pivot split (HiGHS is 1,655/1,633) to localize whether the gap is a phase-1
dual-feasibility-method difference rather than pricing at all.

**Risks.** Lagged weights risk cycling on a degenerate problem — but greenbea
is ~non-degenerate (1/4,399), so anti-cycling risk is low here (note: this is a
greenbea-specific safety argument, not a general one; a global deployment needs
a fallback to exact weights on degeneracy detection). Perturbation must be
removed and the certificate recomputed on the unperturbed problem — the
optimality proof must be perturbation-free.

---

## Idea 5 — Density-adaptive kernel switching (per-solve dense/sparse selection)

**Mechanism.** Because `rho` runs 59% dense and the priced row 94% dense, the
hyper-sparse gather/scatter/SPA machinery is paying indirect-addressing
overhead on vectors that are nearly dense — the sparse representation is the
wrong tool for most greenbea pivots. Add a per-solve density estimate (cheap:
predicted nnz from the previous pivot's result, which the support overlap makes
reliable) and switch that individual solve to a dense contiguous array
representation when predicted density crosses a fixed global threshold (e.g.
40%). Apply it where density is highest and most stable — BTRAN, PRICE, and the
reduced-cost update — leaving FTRAN alone. One global threshold; no per-problem
tuning.

**Wall term + ceiling.** Attacks `us/pivot` on BTRAN+PRICE+rc = 53.4%. Dense
kernels on ~60-94%-dense vectors avoid gather/scatter overhead for maybe
1.2-1.4x on those terms: 53.4/1.3 = 41.1, us/pivot 95 -> 95*(0.466 + 0.411) =
**83.3, -12% alone**. Modest; a stack filler, not a headliner.

**Not a closed axis.** Nearest closed item is "dense-U FTRAN dead in all three
bandwidth regimes." That kill is a *blanket dense FTRAN*. This is (a) explicitly
*not FTRAN* — it targets BTRAN/PRICE/rc where measured density is far higher
than FTRAN's — and (b) *conditional*, switched per-solve on a live density
estimate rather than always-on. The killed experiment does not cover this
conditional, non-FTRAN application.

**Falsifier-first probe (cheap).** Dump the per-solve density distribution
across the trajectory for BTRAN, PRICE, and rc-update, and microbench the
dense-vs-sparse crossover density for each kernel at greenbea sizes. Kill if
the crossover density sits *above* the observed distribution's bulk (i.e. the
solves are not dense enough for dense kernels to win) — this is the honest risk
that the sparse machinery is already near-optimal at 59% density.

**Risks.** Closest of the five to a closed axis; the probe must run first
precisely to avoid re-deriving the dense-U FTRAN kill. Numerics unaffected
(same arithmetic, different container). Memory: dense buffers for wide vectors
cost allocation — reuse a persistent scratch buffer.

---

## Self-assessment — the single bet

**I bet on Idea 1 (mixed-precision DS body).** It is the only idea that is
simultaneously (a) orthogonal to every closed axis — precision is a virgin
degree of freedom — (b) trajectory-preserving, so it is *immune to the
pivots<->us/pivot trade-off* that has sunk every transfer/warm-start attempt,
(c) grounded in two measured gifts the campaign has never spent: the loose
`eps = 2e-5` acceptance gate and the 59-94% solve-vector density that makes
fp32 SIMD near-ideal rather than the ~1.4x hyper-sparse would allow, and (d)
falsifiable in an afternoon by a pure fp32-rounding *simulation* over the
existing fp64 solver, before a single kernel is rewritten. Its honest ceiling
is -32% alone — short of the flip — but the central thesis of this submission
is that **-41% is a stacking problem, not a single-mechanism problem**, and
mixed precision is the highest-EV, lowest-risk *multiplier* in that stack. The
stack I would actually build: Idea 1 (fp32 body, -32% on us/pivot) x Idea 4's
native pivot trim toward 3,309 (-25% on pivots, no densification because it is
native), which multiplies to **~0.215 s, -49%, a clean flip** — with Ideas 2
and 3 held as further solve-side multipliers if the first stack lands thin.
The reason to lead with precision rather than the pivot trim is asymmetry of
risk: the solve-side cut is certain-to-partially-work and cheap to falsify,
whereas the pivot trim is a genuine research question (does an approximate
weight beat exact DSE here as HiGHS's does?). Bank the certain multiplier
first, then chase the pivot gap with the probe that also identifies HiGHS's
unidentified edge as a side effect.
