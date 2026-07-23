# greenbea idea classes — glm-5.2 (2026-07-18)

Independent ideation thread. Evidence base: `greenbea_dossier_2026_07_18.md`
plus the four supporting probe reports (`pivot_gap`, `ipm_stall`,
`warmstart`, `rr_falsifier`). No measurements were run for this submission;
all arithmetic is grounded in the dossier's published numbers. Mechanism
depth is the deliverable.

## Bet summary (ranked)

1. **Active-set prediction from a partial interior trajectory + reduced-LP
   certification.** Route attack. Predict the 83.2% bound-active set from a
   cheap partial IPM (iter ~50, 0.117s measured), fix those vars, solve the
   ~649-var reduced LP cold, reconstruct the dual certificate. Ceiling ~-57%.
2. **Behavioral identification of HiGHS's DSE-adjacent machinery.**
   Measurement probe. The 1,090-pivot simplex-internal gap lives in
   unidentified machinery around their steepest-edge rule. A pivot-trace diff
   isolates WHERE the paths diverge and which lever (perturbation, phase
   interplay, price strategy) owns it. Enables parity at ~3,309 pivots.
3. **Schur-complement constraint partitioning for sparse BTRAN/FTRAN.**
   us/pivot attack. Strip the p99-degree-91.8 dense rows into a Schur block;
   factor the sparse remainder with a leaner symbolic profile. Ceiling ~-11%.
4. **Graph-permutation of constraint rows for BTRAN/FTRAN vector cache
   locality.** us/pivot attack. Symmetric row reordering by overlap-clustering
   so the 96-99.97% consecutive-pivot support hits contiguous cache lines.
   Ceiling ~-7%; combinable with #3.
5. **Subspace-expanded dual simplex (block pivoting on top-k DSE candidates).**
   Pivots attack. Take k entering columns simultaneously via a small ratio
   test on the k-dimensional face. Ceiling uncertain (-10% to -30%); high
   numerical risk.

---

## Idea 1: Active-set prediction from a partial interior trajectory + reduced-LP certification

### Mechanism

The dossier certifies that 83.2% of greenbea's variables are active at a
bound at optimum (3,219 of 3,868). The IPM stall report shows the IPM primal
converges strongly while the dual certificate pins: at iter 50, mu=2.4e-3,
primal residual 2.9e-6, and only 34 columns carry wrong-sign infinite-side
reduced costs (<1% of columns). The warmstart report shows partial IPM to
iter 50 costs 0.117s — cheap, because the stall and adaptive retries that
balloon the full IPM to 0.7s live at iter 58+, after the point we would
extract. The mechanism: run IPM to a global early-stop predicate (primal
residual < 1e-5 OR mu < 1e-2), extract the primal iterate x*, classify each
column as bound-active when |x_j - nearest_bound_j| < tau * (1 + |bound_j|)
with a global tau (e.g. 1e-3), fix those columns at their bounds, and form
the reduced LP on the remaining ~649 free columns: min c_free^T x_free s.t.
A_free x_free = b - A_fixed x_fixed, lo_free <= x_free <= hi_free. Solve
this reduced LP COLD via dual simplex (no warm start, no crossover — a
different, smaller problem). Reconstruct the full dual certificate: y from
the reduced LP's dual multipliers, s_j = c_j - A_j^T y for the fixed
columns, verify s_j sign against the fixed bound for every fixed column to
eps=2e-5. Any fixed column that violates its reduced-cost sign is added to
the free set and the reduced LP is re-solved (column generation in the
dual); this converges because each violation is a certificate miss, not a
guess.

### Wall term attacked + ceiling arithmetic

Attacks the ROUTE — replaces the full 4,399-pivot dual simplex on 3,868 vars
with: partial IPM (0.117s measured at iter 50) + active-set classification
(negligible) + reduced LP cold solve. The reduced LP has 649 structural vars
and 1,525 constraints (overdetermined; ~876 redundant). Cold dual simplex on
it: ~876 Phase-1 pivots to eject redundant-constraint artificials + ~500
Phase-2 pivots on the effective 649-var basis, at ~40 us/pivot (smaller
basis → cheaper BTRAN/FTRAN), = ~55ms. Reconstruct + KKT verify: ~5ms.
Conservative total: 0.117 + 0.060 = 0.177s. Best case with one re-solve for
misclassified columns: ~0.23s. Against the 0.42s local reference, that is
-58% to -45%. Against the 0.24s HiGHS local, 0.18-0.23s is competitive to
winning. The -41% bar is cleared with margin in the base case.

### Why it is NOT a closed axis

The nearest closed axis is #3 (starting bases / IPM-crossover warm starts)
and #6 (IPM route). The distinctions are load-bearing:

- **Not IPM-as-solver (closed #6).** The IPM does not certify and is not
  asked to. It is killed at iter ~50 (global predicate), well before the
  iter-58 dual stall. Its primal is a PREDICTOR, not a certificate. The
  certificate comes from the reduced LP's dual simplex.
- **Not crossover (closed #3).** Crossover maps x* to a basis on the
  ORIGINAL problem and warm-starts simplex from it (super-basic singular,
  Bixby 4,489-5,412, densifies, never beats cold). This idea does NOT
  construct a basis from x* and does NOT warm-start. It fixes the active
  set and solves a DIFFERENT, SMALLER LP cold. The warmstart probe's
  failure mode (transferred basis densifies solves +27.4% us/pivot) cannot
  recur because no basis is transferred.
- **Not presolve (closed #1).** Presolve eliminates structurally forced
  columns; this fixes columns at PREDICTED-OPTIMAL bounds, a dynamic
  active-set decision validated by KKT after the solve. The dossier's
  presolve fixpoint finding (0 propagated tightenings) is about structural
  elimination, not optimal-bound fixing.
- **Not a leaving rule or ratio test (closed #2, #4).**

The dossier's trade-against evidence (transferred trajectories densify)
does not apply: no trajectory is transferred. The active set is a set
membership, not a basis or a density.

### Falsifier-first probe

Cheapest decisive probe: load the existing IPM iterate at iter 50 (the
stall report already captures it), classify columns by |x_j - bound_j| <
1e-3 * (1 + |bound_j|), compare the predicted active set against the
certified optimal active set from the 4,399-pivot DS solve (which the
solver already exports via `LINPROGX_DS_EXPORT_BASIS=1`). Measure
false-positive and false-negative rates. Then form the reduced LP, solve it
cold via the existing DS path, reconstruct the dual, and run the full KKT
check. Kill criteria (any one kills):
- Active-set prediction error > 15% (false positives + false negatives) at
  iter 50 — would make the column-generation fallback dominate.
- Reduced-LP cold solve does not certify to eps=2e-5 after at most 2
  column-generation re-solves.
- Reduced-LP wall > 0.12s (would push total above 0.24s).
- Total pipeline wall > 0.30s.

All four are measurable in one afternoon with existing probes and the
existing DS export hook. No new C code required for the falsifier.

### Risk notes

- **Active-set prediction quality at iter 50.** The stall report shows 34
  wrong-sign columns at iter 50; if those are misclassified as active when
  they should be free (or vice versa), the reduced LP gives the wrong
  answer. The column-generation fallback handles this, but each re-solve
  adds ~55ms. If >3 re-solves are needed, the ceiling collapses.
- **Reduced-LP overdetermination.** 1,525 constraints on 649 vars means
  ~876 redundant constraints. Phase-1 must eject them. If the redundant
  constraints are not cleanly separable (numerical rank ambiguity), Phase-1
  may churn. A rank-revealing QR of A_free could pre-identify them but
  costs ~64ms — more than letting simplex handle it.
- **Certificate compliance.** The reconstructed dual y comes from the
  reduced LP's constraint multipliers. The fixed-column reduced costs
  s_j = c_j - A_j^T y must be checked against the ORIGINAL bounds, not the
  reduced bounds. eps=2e-5 is the fixed gate.
- **Determinism.** The IPM early-stop predicate (primal residual < 1e-5)
  is global and deterministic. No per-problem threshold.
- **Fallback.** If the reduced LP fails to certify, fall back to the
  existing 4,399-pivot DS. No regression risk.

---

## Idea 2: Behavioral identification of HiGHS's DSE-adjacent machinery

### Mechanism

The dossier establishes that on the identical 1,525x3,868 reduction, our
exact Forrest-Goldfarb DSE (4,675 pivots, correct crash-basis gamma, exact
update) does NOT reproduce HiGHS's 3,309, while HiGHS's own DSE flag
(`simplex_dual_edge_weight_strategy=2`) does give 3,309. Our Dantzig
(4,399) crushes HiGHS-Dantzig (12,279). The edge is therefore not the
textbook DSE rule itself but DSE-ADJACENT machinery: phase structure
(HiGHS runs DuPh1 1,655 / DuPh2 1,633 / PrPh2 21 — a real split, vs our
unified "Phase-2 only" loop with 30 artificial ejections), ratio-test
interplay, perturbation, or price strategy (`simplex_price_strategy=3`,
meaning undisclosed). The mechanism is a black-box behavioral probe that
pinpoints where the two solvers' pivot sequences diverge and which
controllable lever owns the divergence, WITHOUT reading HiGHS source.
Concretely: capture HiGHS's per-pivot trace via `log_dev_level=3` +
`simplex_iteration_limit` as a sliding window, and capture our trace via
the existing DS instrumentation. Align the two traces by entering-variable
index at each pivot. Find the FIRST pivot where the entering variables
diverge. If they agree for N pivots then diverge, the divergence point
reveals WHERE the machinery kicks in. Then ablate each candidate lever in
isolation: (a) perturbation — run HiGHS with `simplex_scale_strategy`
variations and look for pivot-count sensitivity (perturbation often ties to
scaling); (b) phase interplay — check whether HiGHS's DuPh1 pivots are
doing dual-feasibility repair that our unified loop defers; (c) price
strategy — the undisclosed `simplex_price_strategy=3` can be behaviorally
fingerprinted by comparing its candidate-ranking against DSE on
overlapping pivots.

### Wall term attacked + ceiling arithmetic

Attacks PIVOTS. If the probe identifies a replicable mechanism and we
match HiGHS's 3,309 pivots at our 95 us/pivot, wall = 3,309 * 95us = 314ms
= -25%. Not enough alone. But combined with ideas #3 and #4 (us/pivot
reductions), 3,309 * (95 - 10.5 - 7)us = 3,309 * 77.5us = 256ms = -39%,
within striking distance of -41%. The probe itself does not reduce wall;
it IDENTIFIES the target for other ideas to attack, and the dossier
explicitly validates this as a submission class.

### Why it is NOT a closed axis

The nearest closed axis is #2 (leaving rules: Dantzig, FG-DSE, Devex,
rule2/3/4). The distinction: those are LEAVING-RULE swaps on our solver.
This is a MEASUREMENT PROBE that identifies what HiGHS does differently
behaviorally. The dossier says "HiGHS's DSE-adjacent machinery is
unidentified" and explicitly invites "a probe design that identifies it
behaviorally, without reading source." The probe does not assume the
machinery is a leaving rule — it might be perturbation, phase structure,
or price strategy, none of which are in the closed leaving-rule family.
The phase-structure angle is distinct from closed axis #3 (starting
bases): #3 is about the BASIS we start from; this is about how Phase-1
and Phase-2 INTERLEAVE during the solve.

### Falsifier-first probe

Run HiGHS on our 1,525x3,868 reduction with `simplex_iteration_limit`
stepped from 1 to 3,309 in chunks (e.g., every 100 pivots), capturing
`getBasis()` at each stop. Capture our DS basis at the same pivot
indices. For each pivot index, compute the symmetric difference of the
basic sets. Plot |basis_symdiff| vs pivot index. Kill criteria:
- If the bases agree for the first ~200 pivots then diverge monotonically,
  the machinery is active from the start (likely perturbation or scaling) —
  this is a FINDING, not a kill.
- If the bases diverge from pivot 1 and never reconverge, the machinery is
  a different crash/Phase-1 structure — this COLLIDES with closed axis #3
  and is a kill.
- If no lever ablation (scale strategy, price strategy) changes HiGHS's
  pivot count by >5%, the machinery is internal and not behaviorally
  isolable through the public option surface — kill.

### Risk notes

- **Behavioral probes can be misleading.** HiGHS may have multiple
  interacting differences; single-variable ablations may show no effect
  even when the combined effect is large.
- **The probe may identify a mechanism that is impractical to replicate**
  (e.g., requires a full perturbation framework). Identifying is not
  implementing.
- **Even if identified, matching 3,309 pivots at 95 us/pivot is -25%,
  short of -41%.** This idea MUST combine with us/pivot ideas to flip.

---

## Idea 3: Schur-complement constraint partitioning for sparse BTRAN/FTRAN

### Mechanism

Greenbea's row-degree distribution is highly skewed: p50=5, p99=91.8. A
small tail of high-degree rows injects dense fill into the basis factor
B = L*U, inflating BTRAN (18.9% of per-pivot) and FTRAN (17.9%). The
mechanism: partition the 1,525 constraints into a "sparse block" (rows
with degree below a global threshold, e.g., 30) and a "dense block" (the
~76-150 high-degree rows). Factor the sparse block's basis submatrix with
Forrest-Tomlin as usual (lean symbolic profile, low fill). Handle the dense
block via a Schur complement: the dense rows' contributions are folded
into a small (d x d, d~76-150) dense Schur matrix that is factorized and
solved directly. BTRAN and FTRAN become: solve against the sparse L*U
(cheap), then apply a dense Schur correction of size d. The Schur update
per pivot is O(d^2) = ~5,800-22,500 flops — negligible against the
23k-nnz sparse solve. The dense rows never enter the sparse factor, so
fill from them is eliminated entirely.

### Wall term attacked + ceiling arithmetic

Attacks US/PIVOT (BTRAN 18.9% + FTRAN 17.9% = 36.8% of per-pivot). If the
dense-row tail owns ~30% of the factor fill (plausible given p99=91.8 vs
p50=5), stripping them cuts BTRAN+FTRAN cost by ~30%. Save: 0.368 * 0.30 *
95us = 10.5 us/pivot. Over 4,399 pivots: 46ms = -11% of the 0.42s wall.
Composes with idea #4 (cache locality) and idea #2 (pivot parity):
3,309 * (95 - 10.5 - 7)us = 256ms = -39%.

### Why it is NOT a closed axis

The nearest closed axis is #5 (per-pivot kernels: "dense-U FTRAN dead in
all three bandwidth regimes" and "block-row uplook gate SHIPPED"). The
distinctions: "dense-U FTRAN" was about a DENSE representation of the U
factor (a storage-layout choice that died). This is about PARTITIONING THE
CONSTRAINTS by row degree and using a Schur complement for the dense
subset — a different algebraic structure, not a dense storage of U. "Block-
row uplook gate" is a specific optimization within the existing FT factor;
the Schur complement is a structurally separate dense block that never
enters FT. The existing `min_degree_prototype.py` operates on the IPM's
normal equations ADA^T, not on the dual simplex basis B — a different
matrix. The supernodal roadmap is for the IPM factorization path, not the
DS basis.

### Falsifier-first probe

Instrument the existing DS factor to report, per refactorization: (a) the
row-degree histogram of the current basis columns, (b) the L*U fill count,
(c) the fill attributable to the top-5% highest-degree rows (by removing
them and re-factorizing symbolally). Kill criteria:
- If removing the top-5% dense rows does not reduce symbolic fill by >15%,
  the dense tail does not own enough fill to justify the Schur overhead.
  Kill.
- If the Schur block size d > 200, the O(d^2) per-pivot correction exceeds
  ~3ms and erodes the gain. Kill.
- If the partition is unstable across refactorizations (the dense-row set
  churns by >30% between refactors), the Schur matrix must be rebuilt
  frequently and the overhead dominates. Kill.

### Risk notes

- **Schur update correctness under FT.** The FT update modifies B by
  rank-1 per pivot; the Schur complement must be updated consistently.
  Standard Schur-complement simplex implementations (Fourer, Bixby)
  handle this, but the interaction with our shipped FT needs care.
- **Numerical conditioning.** The dense Schur block may be ill-conditioned
  if the dense rows are nearly linearly dependent. A condition check on
  the Schur matrix is needed; fallback to full FT on conditioning failure.
- **Certificate compliance.** The Schur-complement BTRAN/FTRAN must
  produce the same y and alpha vectors as the full FT, to machine
  precision. The certificate is unchanged; only the solve path differs.

---

## Idea 4: Graph-permutation of constraint rows for BTRAN/FTRAN vector cache locality

### Mechanism

The dossier reports 96-99.97% consecutive-pivot support overlap in DS.
The within-pivot reuse is shipped; the cross-pivot reuse was killed for a
SPECIFIC algebra (alpha'_k needs old alpha_k, so the FTRAN result cannot
be carried across pivots). That kill is about FLOP reuse, not memory
locality. This idea attacks the MEMORY LAYOUT of the BTRAN/FTRAN vectors:
the sparse right-hand-sides (c_B for BTRAN, the entering column for FTRAN)
and the solution vectors (y, alpha) are scattered across memory in
presolve-inherited row order. A one-time symmetric row permutation of A
(and correspondingly of B's row/column indexing) by a graph-clustering on
the row-overlap graph — edge weight = Jaccard of structural row supports —
places rows that co-occur in the same BTRAN/FTRAN supports into contiguous
memory. The scatter/gather in BTRAN and FTRAN then hits contiguous cache
lines and TLB entries instead of strided access. No FLOP is eliminated;
the same operations execute against warmer cache.

### Wall term attacked + ceiling arithmetic

Attacks US/PIVOT (BTRAN 18.9% + FTRAN 17.9% = 36.8%). If ~40% of BTRAN/FTRAN
wall is cache-miss overhead (plausible for 0.39% density with strided
scatter/gather), and the permutation halves that miss overhead, save:
0.368 * 0.40 * 0.50 * 95us = 7 us/pivot. Over 4,399 pivots: 31ms = -7%.
Composes with #3: combined us/pivot reduction ~17.5us, giving 3,309 *
77.5us = 256ms = -39% with idea #2.

### Why it is NOT a closed axis

The nearest closed axis is #5 (cross-pivot reuse KILLED on algebra) and
"block-row uplook gate SHIPPED." The distinction: the cross-pivot kill was
about reusing ALPHA VALUES across pivots (an algebraic harvest that fails
because the basis changed). This is about reordering MEMORY so the same
FLOPs hit warmer cache — zero algebra changes, zero value reuse. "Block-row
uplook gate" is a decision gate within the FT update; this is a static
problem-representation reordering applied once before the solve. The
`min_degree_prototype.py` operates on the IPM's ADA^T, not on the DS basis
vectors.

### Falsifier-first probe

Instrument BTRAN and FTRAN with `perf stat -e cache-misses,L2-cache-misses`
on the current solver. Then apply a one-time symmetric Metis-style
permutation of A's rows (via `scipy.sparse.csgraph` on the row-overlap
graph) and re-run. Kill criteria:
- If L2 cache-miss rate does not drop >30% on greenbea specifically, the
  access pattern is not the bottleneck. Kill.
- If wall improvement < 3% after permutation, the cache-miss reduction
  does not translate to wall (possibly because the FT factor's own layout
  dominates). Kill.
- If the permutation degrades another fixture by >5% (global mechanism
  must not regress), kill.

### Risk notes

- **Interaction with FT.** FT's pivot order is dynamic (determined by
  leaving variables); a static row permutation may be partially
  overridden. The permutation must be applied to A's row ordering BEFORE
  basis selection and respected by FT's pivot candidates.
- **Interaction with Suhl search (shipped).** The Suhl bounded-pivot
  search assumes a particular column ordering; the row permutation must
  not break its invariants.
- **The ceiling is modest (-7%) and alone cannot flip.** This idea is a
  combinable contributor, not a standalone fix.

---

## Idea 5: Subspace-expanded dual simplex (block pivoting on top-k DSE candidates)

### Mechanism

Dual simplex selects one leaving and one entering variable per pivot.
Block pivoting selects k entering columns simultaneously: from the DSE
pricing, take the top-k candidates by edge weight, form the k-dimensional
entering submatrix, and solve a small k-variable ratio test on the dual
feasibility face. If the k-pivot is accepted, k variables move in one
step. The mechanism: after DSE pricing produces the ranked candidate list
(already computed), instead of taking only the top-1, evaluate whether the
top-k (k=2 or 3, global) can jointly enter without dual-infeasibility. The
ratio test generalizes from a 1-D breakpoint walk to a k-D face walk:
compute the minimum step that maintains dual feasibility for all k entering
variables, accepting the bound-flip subset that stays feasible. This is
dual block pivoting, distinct from primal "multiple pricing" (which only
pre-prices, still pivoting one at a time).

### Wall term attacked + ceiling arithmetic

Attacks PIVOTS. If k=2 and 50% of mega-pivots succeed (both columns
enter), effective pivots drop from 4,399 to ~2,930. But a mega-pivot costs
more than a single pivot: the k-D ratio test and the joint update cost
~1.5x a single pivot (shared BTRAN, slightly more expensive FTRAN for k
columns). Wall = 2,930 * 1.5 * 95us = 417ms = -1%. If k=3 and 60% succeed:
effective pivots ~1,950, cost ~2x, wall = 1,950 * 2 * 95us = 371ms =
-12%. The arithmetic is unfavorable unless the mega-pivot cost is strongly
sub-linear (shared work factor < 1.3x for k=2). Best case with very
favorable sharing (1.2x for k=2, 70% success): 2,640 * 1.2 * 95us = 301ms
= -28%. Still short of -41% alone.

### Why it is NOT a closed axis

The nearest closed axis is #2 (leaving rules: Dantzig, FG-DSE, Devex,
rule2/3/4). The distinction: those all select ONE entering variable per
pivot (a leaving-rule family). Block pivoting selects MULTIPLE entering
variables per pivot — a pivot STRATEGY, not a leaving rule. The "block-row
uplook gate SHIPPED" is about the LU update's internal block structure,
not about block pivot selection. No block-pivoting probe appears in
`experiments/`.

### Falsifier-first probe

Implement k=2 mega-pivot: after DSE pricing, take the top-2 candidates,
form the 2-column submatrix, solve the 2-D ratio test (the breakpoint set
is the union of both columns' breakpoints, intersected with the dual-
feasibility-maintaining region). Measure: mega-pivot success rate, pivot
reduction, numerical stability (refactorization frequency). Kill criteria:
- If mega-pivot success rate < 30%, the 2-D ratio test rarely accepts both
  columns and the overhead is wasted. Kill.
- If refactorization frequency increases > 50% (numerical instability from
  joint updates), the mega-pivots corrupt the factor. Kill.
- If mega-pivot cost > 1.5x single-pivot cost, the sharing is insufficient
  and the wall arithmetic turns negative. Kill.

### Risk notes

- **Dual block pivoting is not well-established in the literature.** Primal
  multiple pricing is standard; dual block pivoting is a research
  direction. The 2-D ratio test is the hard part — maintaining dual
  feasibility for two simultaneous entering variables requires solving a
  small LP on the face.
- **Numerical stability.** Joint updates may amplify cancellation. The
  refactorization cadence (validated at current size) may need
  re-validation.
- **Ceiling arithmetic is unfavorable** unless the shared-work factor is
  strongly sub-linear. This idea is the highest-risk, most uncertain
  contributor.

---

## Self-assessment: which single idea I would bet on

I bet on **Idea 1: active-set prediction from a partial interior trajectory
+ reduced-LP certification.** It is the only idea whose ceiling (-57%
conservative, -58% base) clears the -41% bar with margin AS A STANDALONE
mechanism, without requiring the combinable success of two or three other
ideas. The key insight that makes it viable despite the closed IPM and
crossover axes: the dossier's own warmstart probe measured partial IPM to
iter 50 at 0.117s — cheap, because the 0.7s full-IPM cost is dominated by
the iter-58+ stall machinery that lives AFTER the extraction point. The
IPM primal at iter 50 has mu=2.4e-3 and only 34 wrong-sign columns (<1%),
which is strong enough for active-set prediction (not certification —
prediction). The mechanism sidesteps the trade-against evidence entirely:
no basis is transferred, no density is imported, a different and smaller
LP is solved cold. The falsifier is cheap (one afternoon, existing probes,
existing DS export hook, no new C code) and decisive on the one assumption
that matters — active-set prediction quality at iter 50. If the falsifier
shows >15% prediction error, the idea dies cleanly and I pivot to the
combination path (#2 + #3 + #4, projected -39%, within reach of -41% with
any additional us/pivot work). The risk notes are honest: the column-
generation fallback for misclassified columns could balloon, and the
overdetermined reduced LP needs Phase-1 work. But the floor (fall back to
existing 4,399-pivot DS) is a regression-free safety net.
