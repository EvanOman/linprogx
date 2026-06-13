# Sparse PDHG Performance Handoff

**Date:** Wednesday, June 10, 2026 (parallel-experiment update)
**Primary worktree:** `/home/evan/dev/linprogx`
**Primary branch:** `sparse-support`
**Supersedes:** the May 18, 2026 handoff (column equilibration / tuned polish era)

## June 12 Update 19: O(m^2) Pattern Sort Fixed — ken_18 Setup 10.8s -> 0.03s

Phase-timing chol_setup (new SETUP_MARK prints under
LINPROGX_CHOL_DEBUG) on ken_18 (m=105k) found the one-time setup cost
dominated by the permuted-pattern build: the counting sort scanned all
m buckets per column — O(m^2), 1.1e10 iterations, 10.8 s — exactly the
case the code's own comment warned about ("redo with a sort-free
approach if it ever shows up in profiles"). Replaced with per-column
sorts (insertion for <= 64 entries, qsort above). Assembly phase:
**10.79 s -> 0.03 s**; MD ordering (2.1 s) is now the only meaningful
setup cost on ken-scale instances. Results are bit-identical (sorted
pattern is sorted; ken_18's objective matches to 11 digits).

ken_18's remaining time is per-iteration (m=105k triangular solves and
assembly scatter at ~250-600 ms/iter depending on load); the suite
table keeps the coherent v3 run and will refresh wholesale on the next
full official run.

## June 12 Update 18: Threaded Tail GEMM — pilot87 1.8x, cre_b 1.8x More

The dense-tail trailing update now runs on the existing thread pool
(rows partitioned in 4-aligned chunks, so every output element is
computed wholly by one thread in the same loop order — bit-identical
at any thread count, verified by test and by byte-identical suite
objectives). solve_eq_box_ipm gains a threads kwarg (default 0 = auto
= min(4, cores)); the auto default is safe precisely because the
kernel is deterministic.

Official suite v3 (timings only — results byte-identical to v2):
cre_b 27.6 -> **15.3 s** (now beats Clarabel's 18.7), cre_d 38.1 ->
24.0 s, pilot87 23.8 -> **13.2 s**, maros_r7 9.9 -> 7.4 s, ken_18
23.7 -> 20.6 s, osa_60 -> 14.5 s. Worst suite row is now pds_20 at
34.6 s (PDHG-iteration-bound). Coverage 23/24, 8 fastest-of-three,
every optimum certificate-backed. 135 tests.

## June 12 Update 17: Dense-Tail Factorization LANDED — maros 1.4x, pilot87 1.6x, cre_b 1.5x

Implemented per Update 16, with two lessons the measurements forced:

1. **Entry density is the wrong break-even metric.** The first cut
   selected the tail by pattern density >= 25% and made everything
   SLOWER (pilot87 4x worse): dense processing costs t^3/3 regardless
   of pattern while sparse costs sum(colnnz^2), so 25% entry density
   means ~16x flop inflation. The selection is now flop-based:
   largest tail with t^3/3 <= 3x the tail's sum of squared column
   counts (the kernel's measured speed advantage with margin).
2. **The kernel needs FMA, surgically.** Plain -O3 has no AVX2/FMA;
   adding -march=native globally broke the PDHG bit-identity guarantee
   (FMA contraction everywhere). The tail kernels alone carry
   __attribute__((target("avx2,fma"))) — PDHG math stays bit-identical,
   the tail runs at full speed.

Integration is minimal by design: the dense tail writes its results
back into the same CSC factor storage, so chol_solve / SMW / matvec /
refinement are untouched. Tail rows keep their sparse prefix
processing; the leftover up-looking accumulator values at tail
positions are exactly the Schur-corrected tail entries.

The factor's changed summation order shifted cre_d's degenerate
endgame off its certified trajectory (presolved IPM fails, raw IPM
certifies in 75 iterations). Rather than re-roll the dice, auto now
**retries the IPM on the unpresolved problem when the presolved run
fails certification** — two independent trajectory tickets, soundness
unchanged (certificate-gated), and the fallback chain continues to
PDHG as before. cre_d costs 38 s (failed presolved run + successful
raw run) instead of being lost.

Official suite (full re-run): maros_r7 13.9 -> 9.9 s, pilot87 37.7 ->
23.8 s, cre_b 41.6 -> 27.6 s, ken_18 25.8 -> 23.7 s, cre_d 27.6 ->
38.1 s (the robustness cost). Coverage 23/24 and 8 fastest-of-three
held. Known pre-existing corner (documented, untouched): when every
column exceeds the dense-column threshold the SMW path solves against
M_s = delta*I and loses ~10 digits — the tail never activates there
(diagonal-only pattern fails the flop test by construction).

Tests: dense-tail residual checks vs dense reference at sizes that
activate the path (134 total).

## June 12 Update 16: Dense-Tail Factorization — Designed and Sized

Measured flop concentration in the factor (numeric Cholesky of the
AMD-permuted normal equations, pattern thresholded):
- cre_b: the trailing **10% of columns hold 57% of factor flops at 99%
  density** (tail-20%: 97% of flops at 47% density).
- pilot87: tail-30% holds 46% at 100% density; tail-50%: 95% at 69%.
- maros_r7: flatter — tail-20% holds 14% at 100%; tail-30%: 29% at 79%.

With the blocked dense kernel's measured 4.2 Gflop/s vs the sparse
scalar up-looking path's 0.93 (4.5x), the expected wins are ~2x cre_b,
~1.5x pilot87, ~1.2x maros_r7. Design (classic dense-tail, no public
solver source consulted):

1. **Tail selection (symbolic)**: from the column counts, pick the
   largest tail size t whose predicted block density
   sum(colnnz[s:]) / (t(t+1)/2) >= 0.25 (dense processing at 4.5x
   breaks even at ~22% density), capped at t <= 2048 for memory
   (t^2 doubles).
2. **Numeric**: up-looking unchanged for rows < s. For tail rows,
   compute only the prefix part L[i, :s] sparsely (ereach pattern,
   skip cols >= s). Then T = G[s:, s:] + delta*I minus Lp Lp^T
   accumulated by column: transpose the tail rows' prefix parts to
   CSC, rank-1 scatter per prefix column (same flops as today, scalar)
   — then blocked dense Cholesky of T (4x4 register-tiled GEMM update,
   NB=96, as benchmarked in Update 15).
3. **Solves**: forward/backward split at s — sparse prefix as today,
   dense triangular solves on the tail (t^2, negligible).
4. chol_matvec (refinement) and the SMW dense-column path are
   unaffected (they sit above chol_solve).

Validation plan: factor correctness vs the existing path on small
random SPD systems (bitwise is NOT expected — different summation
order — so compare residuals); IPM end-to-end on the fixture + suite
quality sweep; per-instance refactor timers before/after.

## June 12 Update 15: Stall-Gated Early Acceptance — Suite-Wide IPM Speedups

Component timers (debug-only, behind the debug kwarg) settled the
factorization question: **refactorization is 87-94% of IPM runtime** on
every slow instance, running at ~0.93 Gflop/s scalar. A blocked dense
Cholesky benchmark (`/tmp/chol_bench.c`-style, register-tiled 4x4
GEMM update) measured 4.2 vs 1.8 Gflop/s on this machine — so a
supernodal rebuild is worth ~3-4x on factor time and remains the big
open subsystem.

But the cheaper lever landed first: the debug traces showed instances
satisfying the exit path's relaxed acceptance bars long before the
strict tolerance — pilot87 met them at iteration ~50 of 149, osa_60 at
~57 of 199. The IPM now checks the SAME relaxed acceptance
(pres <= 1e-6, dres <= 5e-6, mu <= 1e-6, certified gap <= 1e-5) inside
the loop, gated on a mu-stall test (mu shrinking less than 4x per 10
iterations — a healthy Mehrotra run does far better, so fast convergers
keep polishing to strict tolerance and lose nothing). When the raw
certificate fails, the min-norm dual cleanup (now extracted into
ipm_dual_cleanup, shared with the exit path) is attempted in-flight,
rate-limited to once per 16 iterations.

Full official suite re-run (loaded machine; all rows refreshed):
- osa_60 33.1 -> **15.6 s** (57 iters) — now FASTEST of three (HiGHS
  24.1, Clarabel 27.5).
- osa_14 4.0 -> 1.7 s; cre_d 31.5 -> 27.6 s (77 iters); cre_a 0.42 ->
  0.17 s (45 iters); 80bau3b 0.67 -> 0.23 s; d2q06c 1.5 -> 0.97 s.
- pilot87 unchanged (its mid-run cleanup attempts fail — the dual face
  is greenbea-like; it still needs the full 150-iteration tail).
- Quality: all rel deltas <= 1.2e-5, every optimum certificate-backed.
- **Fastest-of-three count: 5 -> 8** (fit2p, ken_07, ken_11, osa_30,
  osa_60, qap12, qap15, truss).

Tests: cre_a fixture now asserts the early exit (45 iterations vs 199);
132 tests green.

## June 12 Update 14: NaN Bail-Out — cre_b 2.8x, ken_18 3.9x Faster

Profiling the slow IPM instances with the debug trace found pure waste:
on cre_b the iterate goes **NaN at iteration ~75** (late Newton
overflow on the ill-conditioned endgame, exactly what the best-iterate
snapshot exists for) and the IPM then spins on NaN for 120+ more
iterations — ~65% of its runtime doing literally nothing. ken_18 had
the same pattern. The fix is one guard: if mu/pres/dres goes
non-finite, break immediately; the exit path (best iterate, relaxed
acceptance, dual polish, min-norm cleanup) takes over as before.

Official rows (identical objectives and residuals to the last digit —
the returned point is the same best iterate):
- cre_b: 136.0 s / 199 iters -> **48.2 s / 67 iters**
- ken_18: 100.1 s / ~120 iters -> **25.4 s / 40 iters**
- pilot87: 37.4 -> 36.6 s (no NaN spin; genuinely iteration-bound)
- maros_r7: unchanged 14.2 s (no NaN; factor-bound — 5.7e8 flops x ~45
  iterations on a scalar up-looking factor. The remaining runtime gaps
  vs HiGHS on maros_r7/pilot87/cre_b-class instances need either a
  supernodal/blocked numeric factorization (the big subsystem) or
  fewer IPM iterations.)

Tests: finite-iterates property test on ill-conditioned random LPs
(the user-visible contract the bail protects); 132 total.

## June 12 Update 13: osa_60 SOLVED — Predicted-Fill Early Abort (23/24, Coverage Tie)

osa_60 was never a throughput problem. Threading the PDHG kernels was
built and measured first (persistent pthread pool, disjoint-output jobs
with canonical-order reductions — bit-identical at any thread count,
verified): ~14% at 2 threads on the loaded 12-core box, saturated by
memory bandwidth, nowhere near the 2.3x needed. The `threads` kwarg is
kept (default 1, 0 = auto), with a bit-identity test; the serial path
keeps the direct fused loops so it pays no scratch-array tax.

The real finding: osa_60's IPM factor costs only **2.8e7 flops** — 25x
UNDER the 7e8 cap — but the minimum-degree ordering on its dense-ish
graph (nnz(AAT)/m^2 ~ 1%, same as qap15) exceeded the 1.5e9-op ordering
budget, so the IPM never got to measure the factor. nnz(AAT) density
does NOT discriminate filling from non-filling graphs (osa_60 0.0096 vs
qap15 0.0094); only the ordering itself reveals fill.

Fix: **abort the ordering on predicted factor cost, not ordering
effort**. Each eliminated pivot with external degree d adds ~d^2 to the
factor flops; min_degree_impl now accumulates that running prediction
and aborts once it passes 4x the flops cap (margin because approximate
degrees overestimate; the exact post-ordering check still does the fine
gating). Fill-explosive graphs (qap) now abort in milliseconds instead
of burning the budget; low-fill graphs (osa) complete no matter how
much ordering work they take. The ordering ops budget becomes a pure
time guard, raised to 1e10 (~20 s at this machine's throughput).

Official results: **osa_60 optimal 33.1 s** (ipm, 199 iters, one dual
cleanup round, rel delta 5.5e-6, rel residual 8.7e-8) — was timeout;
HiGHS 24.1 s, Clarabel 27.5 s. qap12/qap15 unchanged (0.44 s / 1.07 s
— early abort). pds_10 4.99 s and pds_20 40.7 s (identical iteration
counts; the deltas are one completed-then-rejected ordering attempt
plus machine load).

**Coverage: 23/24 — equal with HiGHS and Clarabel.** Each solver misses
exactly one instance: HiGHS times out on qap15, Clarabel reports
DualInfeasible on ken_18, linprogx declines to certify greenbea (where
Clarabel certifies a point that is wrong by 1.3e-3).

## June 12 Update 12: greenbea Fully Characterized — Dual Repair Is the Hard Core

greenbea's IPM is not diverging; the pace watchdog kills it mid-flight.
With the watchdog disabled (`debug=True` kwarg added to
`solve_eq_box_ipm` for this work), the run converges beautifully:
**rel 9.3e-7 objective, raw residual 1.2e-10, in 1.9 s / ~300
iterations** — steps lengthen and mu accelerates right where iter-60
used to bail. mu bottoms at 1.7e-9; running 500 or 1000 iterations
returns the identical best iterate.

What still blocks coverage is the certificate: 101 wrong-signed reduced
costs (up to 2e-4 relative) on infinite-bound columns. Three repair
approaches measured, all negative:
1. **Union-set min-norm cleanup** (the stage that certifies cre_a/b):
   diverges — zeroing 101 violations creates 213, then 467; the union
   blows past the 512 cap in three rounds.
2. **Protected least squares** (violators pinned to margin + 1,044
   near-feasible inf-bound columns pinned to current values): union
   grows slower (648 after 9 rounds) and the would-be gap falls to
   1e-7, but ~100 fresh violations surface every round. The protection
   is soft; LS residual spills onto protected columns.
3. **Agmon-Motzkin-Schoenberg projection** (most-violated-constraint,
   exact projections): stalls at violation 4e-4 within 5k projections
   and stays there for 200k — near-antiparallel active constraints make
   the geometric rate hopeless.

Conclusion: greenbea's dual face requires a genuine linear-feasibility
solve with curvature handling (the repair LP itself, or a dual-side
active-set method). The watchdog is left unchanged — without a
certificate at the end, letting the IPM run just adds ~2-6 s before the
same PDHG fallback. If the dual repair is ever cracked, re-visit the
watchdog with a progress-aware rule (e.g. bail at iter 60 only if
mu_60 > 0.25 * mu_40), which is what greenbea needs to reach its
converged point. Coverage stays 22/24; greenbea remains the one
instance where honesty (no certificate, no claim) costs us vs Clarabel
certifying a 1.3e-3-wrong point.

## June 12 Update 11: Min-Norm Dual Cleanup — cre_a and cre_b SOLVED

The crossover line paid off, but not the way it was headed. The Tapia
probe (`experiments/tapia_probe.py`) was a clean no-go — the IPM exit
point identifies only 974 basic columns (vs 931 from the PDHG stall;
singular-basis abort at pivot 623), confirming no point-based indicator
can identify cre_a's ~2,500 degenerate basis members. But measuring the
IPM exit directly revealed the real opening: at max_iter=200 the IPM
reaches rel 1.4e-6 / pres 4.3e-6 on cre_a in 0.4 s with only **42**
violating reduced costs (and 56 on cre_b after a 138 s run).

With |S| that small, a min-norm dual correction solving
`A_S' delta = r_S - margin` (Gram matrix A_S'A_S, dense Cholesky,
min-norm via delta = A_S w) zeroes the violations while barely moving
the other reduced costs. Iterating with a cumulative union set (zeroed
columns can push neighbors slightly negative; the union with a +margin
target converges where one-shot flip-flops) certifies cre_a in 2 rounds
and cre_b in 3.

Production: implemented at the IPM exit in `_csparse.c` as a third
stage after relaxed acceptance and dual polish, gated on pres <= 1e-6,
|S| <= 512, <= 5 rounds. Soundness: any y yields a valid Lagrangian
bound, so the stage can gain certificates but never fake one; if the
certificate still fails, y is restored untouched. Result key
`dual_cleanup_rounds` exposes the stage for tests.

Results (official suite worker, eps=2e-5 untuned):
- **cre_a: optimal 0.42 s** (ipm, rel delta 1.9e-7) — was
  iteration_limit. HiGHS 0.10 s, Clarabel 0.15 s.
- **cre_b: optimal ~139 s** (ipm, rel 2.7e-6, residual 3.75e-6) — was
  iteration_limit. HiGHS 2.11 s, Clarabel 18.66 s. The time is almost
  entirely the 200-iteration IPM run; the cleanup itself is ~0.1 s.
- greenbea unchanged: its IPM exits via the pace watchdog at iter 60
  with pres ~1e+2 — an IPM convergence problem upstream of any
  certificate work, still open.
- Coverage: **22/24** (HiGHS and Clarabel 23/24).

Tests: `tests/test_dual_cleanup.py` (6 new; 128 total) — cre_a vendored
as a 47 KB fixture (`tests/data/lp_cre_a.mat`, public LPnetlib data)
asserting the cleanup fires and an independent raw-space audit of the
returned dual; idle-on-clean-problem check; degenerate-LP
never-fake-optimality property vs scipy.

## June 12 Update 10: Dual-Simplex Probe — Basis Identification Is the Bottleneck

The warm-started dual simplex (`experiments/dual_simplex_prototype.py`)
does NOT improve on the primal probe, and the negative result is more
valuable than the hoped-for positive one. Setup: same pivoted-QR warm
basis, every nonbasic column placed on its reduced-cost-sign-matching
bound (dual feasible by construction), temporary expanded bounds where
the matching side is infinite, dual ratio-test pivots. Warm-starting
from the solver's actual stall dual (only ~59 violating reduced costs)
instead of a min-norm least-squares dual improved the start (949 vs
1,578 initial primal violations) but not the shape of the run: ~8,650
pivots to reach zero primal violations at objective 2.3588e7 (3e-4
relative from optimum), with 1,166 columns stuck on artificial expanded
bounds that five 100x enlargements could not release.

The lesson: the "59 violating reduced costs" intuition was misleading.
Dual feasibility at a BASIC solution requires r_B = 0 exactly, and
re-deriving y from a warm basis that is ~1,500 members wrong (cre_a's
interior supplies only 931 of 3,516 basis columns; the rest must be
picked from thousands of degenerate pinned columns with r ~ 0)
scrambles the signs of ~1,600 columns no matter how good the stall dual
was. Both probes converge to the optimum region in 5-20k pivots because
both are paying for the same wrong basis.

A competitive production crossover therefore needs, in order:
1. **Better basis identification** — e.g. Tapia-style indicators
   (x_j/z_j ratios) from the IPM's late iterates, which cleanly
   separate basic from nonbasic in a way a PDHG stall point cannot.
   This is the next cheap probe: cre_a routes through the IPM first, so
   the iterates are already available.
2. **Basis-update machinery** (product-form class) to amortize
   factorizations — only worth building if (1) gets the pivot count
   into the hundreds.

## June 12 Update 9: Crossover Probes — Repair Fails, Warm Simplex Works

Two prototypes for the cre/greenbea endgame (the +3 coverage gap), both
in `experiments/` (`crossover_prototype.py`,
`revised_simplex_prototype.py`):

1. **Naive basis repair** (alternate exact least-squares solves with
   violation-driven partition swaps): does not converge. cre_a's stall
   point pins 6,337 columns but the vertex needs 3,516 basic — 2,605
   basis members sit exactly at bounds (degenerate), the min-norm dual
   then shows ~1,700 spurious sign violations, and both bulk and capped
   paired swaps oscillate. The earlier "34-59 violating coordinates"
   measured certificate violations at the polished y; the *partition*
   ambiguity is far wider.
2. **Warm-started revised simplex** (bounded-variable, scipy splu
   refactored per pivot, pivoted-QR warm basis + Big-M artificials):
   **reaches the published cre_a optimum** — violations 1,804 → 1, zero
   active artificials, objective matching to 7 digits. Cost at
   prototype quality: ~20k pivots, ~8 ms/pivot (~230 s). The warm basis
   is dual-informed but primal-infeasible, causing a Big-M excursion
   (objective spikes to 1e16 before descending), which is where most of
   the pivot count goes.

Conclusion: the endgame is reachable with basis machinery we can build
dependency-free (the C Cholesky solves square basis systems via
B B^T u = rhs, x = B^T u — no LU needed). The promising production
shape is a **warm-started dual simplex**: the stall point is nearly
dual-feasible (only ~59 violating reduced costs at small magnitude), so
dual pivots repairing those should number in the hundreds, not 20k.
That probe is next. Also fixed en route: the existing tableau simplex
times out (>300 s) on cre_a cold — it cannot serve as the crossover
engine at this scale.

## June 12 Update 8: PDHG Kernel — Fused Dual Pass, Branchless Clamp

Per-iteration cost baselined (presolved instances, 2000 iterations,
checks disabled): pds_20 2.292 ms/iter, osa_60 10.206 ms/iter. Three
candidates were implemented and measured:

- **Fused matvec + dual trial pass** (one sweep over the rows computes
  ax_trial, gradient, y_trial, dy_sq, interaction): kept.
- **Branchless primal clamp** — fmax/fmin against the ±INF-filled scaled
  bounds replaces the bound_kind switch, letting the trial loop
  vectorize: kept.
- **2-way unrolled accumulators** in both matvecs: measured *slower* on
  osa_60 (8.843 vs 8.710 ms/iter without) and the changed summation
  order perturbed FP trajectories enough to knock CYCLE off its
  convergence path (plateau-exit at residual 3.3e-5 instead of
  converging) — the CYCLE guardrail test caught it. Reverted.

The two kept changes preserve summation order exactly, so all
trajectories are bit-identical to the previous kernel; the full test
suite (122) passes unchanged. Result: pds_20 2.115 ms/iter (-7.7%),
osa_60 8.710 ms/iter (-14.7%). Commit `c422f4e`. This is the realistic
ceiling for single-threaded kernel micro-optimization — the loops are
memory-bound gathers; the remaining ~2.3x osa_60 needs would require
threading or an algorithmic change.

A suite re-run on the four PDHG-routed instances reproduced every
objective, residual, and iteration count to the last digit (confirming
bit-identity end to end), but wall clocks came out 25-35% slower than
the recorded table because the machine was under load (loadavg ~5.8,
tailscaled + headless Chrome). The kernel win is paired-measured
(back-to-back on identical load), so the official table runtimes are
retained from the prior quiet-machine run; refresh them with the next
full suite run.

## June 12 Update 7: osa_60 Characterized — Solvable, Budget-Bound

With an unconstrained budget, osa_60 solves: optimal in 486s at residual
1.9e-5 (under the 2e-5 bar) and relative objective error 2.1e-7, after
82,304 PDHG iterations. It is a throughput question, not a capability
gap: the suite's 180s protocol budget is what it misses. A ~2.7x PDHG
kernel speedup would bring it inside; kernel throughput (currently
~1-2 Gflop/s on the matvec pair) is the one remaining broad,
non-problem-specific lever and also benefits pds_10/pds_20/qap
instances. All four remaining misses are now fully measured and
characterized: three structural-dual research items plus one
throughput item.

## June 12 Update 6: Compensated Summation — Killed by Measurement

Before implementing double-double residual accumulation for the cre
endgame, the certificate violations were measured directly: cre_a has 59
violating reduced costs with worst relative magnitude 8.3e-5, cre_b has
34 at 3.9e-4 -- four to five orders above the 1e-9 certificate tolerance.
Extra floating-point digits cannot close that; the duals genuinely have
not converged on a few dozen degenerate coordinates. The cre/greenbea
endgame is finite-termination / crossover territory, recorded as the
open research item it is.

## June 12 Update 5: Staged Precision (final round of this push)

Once mu < 1e-7 the regularization floor drops 1e-10 -> 1e-12 with a
second refinement pass covering the conditioning: fit2p converges to the
strict KKT test in 21 iterations (0.55s -> 0.11s, faster than BOTH HiGHS
and Clarabel), 80bau3b reaches 9.2e-12 relative. The sound dual polish
(weighted-least-squares y from the final factor on the uncertified exit
path) is in place but does not recover cre_a/cre_b: their stalls are
structural. FINAL official scoreboard of this push: 20/24 all
certificate-backed, fastest-of-three on 5, vs HiGHS/Clarabel 23/24.
Remaining open research items: cre_a/cre_b/greenbea IPM endgame (needs
genuinely deeper dual convergence -- candidates: quad-precision residual
accumulation, neighborhood-following with separate step lengths done
right), and osa_60 (certified first-order finish beyond 180s).

## June 12 Update 4: Lagrangian-Certified Acceptance

The relaxed-acceptance gap is now a true Lagrangian bound built from the
actual reduced costs r = c - A'y split onto bounds (pinned zero-width-box
columns absorb r; a reduced cost pointing at an infinite bound makes the
iterate uncertifiable). Found via 80bau3b: its IPM was fully converged but
blocked by a phantom gap (pinned columns omitted from the dual objective);
the naive pinned fix then wrongly certified greenbea's 1.3e-3-off point
(which Clarabel also certifies). The sound certificate keeps 80bau3b
(0.59s, 7.5e-9 relative) and rejects greenbea; cre_a/cre_b lose their
certificates (slightly negative reduced costs on unbounded variables at
their stall points) and are reported honestly. Official suite: 20/24, all
optima certificate-backed. Next idea for cre_a/cre_b: dual polish at the
best iterate (recompute y from the final factorization via weighted least
squares; any y yields a valid bound, so trying a better y is sound and
costs one solve).

## June 12 Update 3: Dense-Column Splitting (fit2p solved)

Sherman-Morrison-Woodbury treatment of dense columns in the normal
equations: columns with nnz > max(64, m/8) are excluded from the sparse
factor and handled as U = A_d sqrt(D_d), W = M_s^-1 U, and a dense
Cholesky of I + U'W (k <= 256). fit2p (25 dense columns, previously a
fully dense 9-Gflop factor) moved from unsolved to optimal in 0.59s,
faster than HiGHS. Official suite: 21/24, fastest-of-three on 5,
ties/beats Clarabel on 11 of 21.

Process lesson recorded: a silent patch no-op plus a grep that masked a
compile error let several validations run against a stale .so (caught by
profiling: the hotspots matched the old code path). All bulk edits now
assert their replacements, and rebuilds are verified by .so mtime.

## June 12 Update 2: Gondzio Correctors — Tried and Reverted

Multiple centrality corrections (enlarged-step trial products clamped into
[0.1, 10] x sigma mu, corrector solve reusing the factorization, accept on
step improvement) were implemented and benchmarked. Per-instance iteration
counts improved where trajectories were already healthy (pilot87 164->125,
cre_d 78->53, maros_r7 18->13, CYCLE 28->22) but the suite NET was
negative: cre_b tipped from a 152s solve into timeout (corrector overhead
without iteration gains there), and several endgames lost 2-4 digits of
final residual quality (osa_30 1e-4 -> 6e-3) because corrections perturb
the delicate late trajectory that the relaxed acceptance then snapshots.
Gating corrections by mu and tightening acceptance made things worse, not
better. REVERTED in full. Lesson: with the current single-step-length
formulation, the correctors' benefit is not separable from their endgame
noise; revisit only together with separate primal/dual step scaling and a
fully gap-certified acceptance path.

## June 12 Update: Approximate Degrees, Refinement, Calibrated Caps

- Approximate-degree ordering (element residuals computed once per
  elimination): 10-60x faster than exact MD at equal or better fill
  (ken_18 49s -> 1.7s, DFL001 0.14s with 9% better fill). Routing ceiling
  raised to 50k rows; ken_18 now solves via the IPM (~148s) where Clarabel
  fails (DualInfeasible).
- One step of iterative refinement on every Newton solve (residual via the
  assembled permuted matrix). Residuals on long solves improve up to six
  orders of magnitude.
- Pace watchdog (bail at iteration 60 if mu has not dropped 1e4x) and a
  throughput-calibrated factor-flops cap (7e8): tractable-but-slow factors
  (pds_10, 9.5e8 flops) go to PDHG where they solve in seconds.
- Official suite: 20/24 solved. osa_30 flipped to the IPM and now beats
  HiGHS (4.7s vs 7.1s). 80bau3b regressed into an IPM trajectory stall
  under the new ordering (refinement did not save it) and its PDHG
  fallback also fails: a robustness target alongside greenbea. The
  IPM-trajectory fragility on these two instances is the clearest signal
  that the endgame needs higher-order correctors (Gondzio) or adaptive
  step safeguards.
- Remaining named fixes: fit2p (dense-column splitting), osa_60
  (certified first-order endgame; ends at residual 4.9e-5 vs the 2e-5
  bar), greenbea/80bau3b (IPM endgame robustness).

## June 11 Update 2: Cost-Based Routing

Routing to the IPM is now decided by measured factorization cost, not row
count: auto attempts the IPM for reduced problems up to 16k rows; the C
side aborts via a minimum-degree work budget (1.5e9 scan ops, 4x for
m <= 3000) or the 1e9 factor-flops cap, and falls back to PDHG. Suite
moved from 17/24 to 20/24 (stocfor3 0.82s, cre_d 27s, cre_b 117s now
solve; ken_11/ken_13/osa_14 flipped to the faster IPM route). Fastest-of-
three on 4 instances; ties/beats Clarabel on 10 of 20 solves. Remaining
misses and their named fixes: ken_18 (approximate-degree ordering),
fit2p (dense-column splitting in the normal equations), osa_60 (certified
first-order endgame or longer budget), greenbea (stalls on both paths).

## June 11 Update: LPnetlib Suite Round

A 24-instance LPnetlib sweep (`experiments/suite_bench.py`, results in
`assets/lpnetlib_suite.md`) drove a major IPM hardening round:

- **Mehrotra least-squares starting point** (dominant fix), mu-proportional
  regularization (floor 1e-10; free columns stay at fixed 1e-8 — shrinking
  them overflows their Newton block), slack/dual positivity floors,
  best-iterate snapshots with a finiteness guard, and a relaxed acceptance
  for stalled runs that REQUIRES an explicit primal-dual gap <= 1e-5
  (small mu with an infeasible dual hid a 1.3e-3 objective error on
  greenbea — Clarabel certifies that same wrong point).
- **Density guards**: Python-side dense-column check and a C-side factor
  flops cap (1e9, measured as sum of squared L column counts after
  symbolic analysis) route clique-forming instances (fit2p, qap12) to
  PDHG. qap12: 84s IPM -> 4.6s PDHG (HiGHS needs 96.7s).
- Suite scoreboard: linprogx 17/24 solved (HiGHS 23, Clarabel 23),
  fastest-of-three on 3, quality 1.8e-12..2.2e-5 relative where solved.
  CYCLE improved again: 28 IPM iterations, 0.106s.
- Useful external references found this round: Mittelmann benchmarks
  (plato.asu.edu, MPS format — needs a parser, future work), published
  Netlib optima CSV (github.com/SkyLiu0/NETLIB, Gurobi 1e-8), Clarabel
  benchmarks repo (oxfordcontrol/ClarabelBenchmarks).
- **Next frontier**: the 7 unsolved instances are PDHG tails — cre_b/cre_d,
  fit2p, ken_18 end at residuals 9e-5..1e-3 (just above the 2e-5 bar) with
  excellent objectives; greenbea/stocfor3 stall further out; osa_60 needs
  more than the 180s budget. A PDHG endgame that certifies these
  near-misses (better cleanup, or longer adaptive budgets) is the highest
  value remaining work.

## June 10 Update: Parallel CYCLE-gap Experiments

Three experiments ran in parallel worktrees and were integrated:

1. **Doubleton-row presolve (WIN, integrated as `src/linprogx/presolve.py`).**
   Dependency-free presolve: empty rows, cascading singleton rows, and
   doubleton-row substitution `x_p = (b_i - d*x_q)/a` with bound mapping and
   postsolve replay. Fill limit `max_fill=5` is critical (fill 2 and 10 both
   fail to help; conditioning, not size, is what matters). Wired into
   `SparseSolver` behind `presolve=True` (default). CYCLE: removes 388 rows /
   360 cols and converges via FULL KKT at 36k iterations, delta 2.8e-6.
   DFL001: removes 15/15, 27.9k iterations, delta 0.16.
2. **Plateau early-exit (integrated, dormant insurance).** Ring buffer of
   best-seen relative KKT per eval; if <2% improvement over the last 80 evals
   and the best iterate is within 50x tol primal residual, adopt the best
   iterate and exit. `plateau_window`/`plateau_threshold` kwargs; result dict
   reports `plateau_exit`. With presolve active it never fires on the
   benchmarks; it exists for degenerate shapes presolve cannot fix.
3. **eval_interval 64 -> 40 (reverted; interaction lesson).** A clear win on
   the UNPRESOLVED CYCLE (full KKT at 39.9k iters) but a loss on both
   presolved problems (CYCLE plateau-exits prematurely at delta 1.1e-3,
   DFL001 regresses to 35k iters). The restart trajectory is chaotic in
   these parameters; tune them only jointly with presolve. The
   `eval_interval_override`, `restart_*` and `debug` kwargs from this
   experiment were kept as tooling. The diagnosis stands: long restart
   epochs let omega drift fatally (e.g. 0.027 -> 0.004) when the sufficient
   criterion is unreachable and the artificial restart is ~17k iters away.

**Current committed results:** DFL001 5.34s delta 1.6e-1 (HiGHS 6.38s,
Clarabel 8.06s — linprogx fastest); CYCLE 1.39s delta 2.8e-6 via full KKT
(HiGHS 0.18s, Clarabel 0.22s — still ~7x slower, expected simplex territory).

### Second round: remaining CYCLE ideas, all exhausted

A follow-up round swept every remaining identified idea on the PRESOLVED
problems. Every knob is already at its optimum and every dynamics variant
regressed; the conclusion is that ~36k iterations is what this restarted
PDHG costs on CYCLE, and further gains need a different algorithm class
(e.g. a simplex/crossover endgame), not more tuning.

- Restart constant sweeps (sufficient 0.1-0.4, necessary 0.7-0.9,
  artificial 0.2-0.55, eval_interval 48-128): the defaults
  (0.2/0.8/0.36/64) are the optimum of a chaotic landscape; most neighbors
  fail outright. rs=0.10 saves ~2% (noise).
- max_fill sweep (3-12): 3-5 give identical reductions and the best result;
  6+ degrade quality and can break convergence.
- omega seeds (0.01-10): best non-default (0.1) saves ~6%, within noise.
- The debug trajectory shows where CYCLE's iterations go: ~60% of the run
  (iters ~3k-25k) is a second omega down-spiral to 3.5e-5 before omega
  climbs back to its correct level ~2e-2, after which convergence is
  explosive (KKT 3.8e-3 -> 8.4e-5 in ~500 iterations). Two targeted fixes
  for that spiral BOTH regressed:
  - replacing the movement update with residual-balance steps when the KKT
    error is lopsided (CYCLE infeasible at 50k; the early omega descent
    needs the movement signal),
  - clamping the per-restart movement update to a factor 4 (CYCLE
    infeasible at 50k, DFL001 +7% iterations).
- Duplicate-row removal in presolve: finds ZERO rows on either benchmark
  after the singleton/doubleton cascade (the 13 raw duplicates get consumed
  by it) and only adds presolve time. Implemented, measured, removed.

### Third round: a different algorithm class (June 10, continued)

Tried and negative:

- **Active-set crossover from the PDHG point** (3 prototype variants in
  scipy/lsqr: primal-proximity faces, dual-reduced-cost faces, combined).
  On these degenerate problems the predicted face never stabilizes
  (CYCLE: free count < row count -> inconsistent least squares, pres ~1e1+;
  DFL001: dual residual stuck ~0.6). Doing crossover properly requires
  basis management (rank-revealing LU, pushes, ratio tests) — i.e. building
  simplex itself. Not a quick add-on.
- **Halpern anchoring** (blend accepted step toward the cycle anchor with
  weight 1/(k+2), replacing within-cycle averaging): CYCLE infeasible at
  50k, DFL001 +46% iterations. Reverted; averaging + restarts wins again.

Validated and promising:

- **The C result dict now exposes the dual vector `y`** (original units).
- **Mehrotra predictor-corrector IPM prototype**
  (`experiments/ipm_prototype.py`, scipy splu on the regularized normal
  equations, Ruiz + cost scaling, native boxes, zero-width boxes pinned):
  - CYCLE: **34 iterations, delta 1.7e-7, true residual 1.8e-11, 2.4s in
    Python** — HiGHS-class accuracy; a C port should land ~0.2-0.5s.
  - DFL001: normal equations fill badly (98s in Python, wobbly tail,
    delta 2.6e4) — IPM is the wrong tool there, PDHG already beats HiGHS.
  - Conclusion: portfolio architecture (IPM for small/degenerate, PDHG for
    large sparse), which is standard practice, not overfitting.

**The C IPM is now BUILT and SHIPPED (June 10, late session):**

1. Exact minimum-degree ordering in C (`_csparse.min_degree`): quotient
   graph with element absorption, stamp-marked exact degrees,
   lazy-deletion heap. Fill beats SuperLU MMD on CYCLE (59218 vs 63086).
2. `CholContext` sparse Cholesky of `A D A' + delta I`: pattern +
   ordering + etree + ereach symbolic once per problem; per-iteration
   assembly (precomputed scatter map) + up-looking refactorization with
   dynamic pivot boosts + triangular solves. Verified to machine
   precision against scipy (`CSRMatrix.normal_equations_solve`).
3. `CSRMatrix.solve_eq_box_ipm`: Mehrotra predictor-corrector mirroring
   the validated prototype — Ruiz (10 inf passes) + cost scaling,
   zero-width boxes pinned (bound_kind 4, H = 1e16), interior start with
   duals sized to |c|, affine + corrector via the shared Newton solver,
   0.995 step factors, relative-residual + mu termination.
4. `SparseSolver(algorithm="ipm"|"auto")`: auto routes reduced problems
   with <= 4000 rows (AUTO_IPM_MAX_ROWS) to the IPM, larger ones to
   PDHG, and falls back to PDHG if the IPM does not reach optimal.
   Both benchmarks use `algorithm="auto"`.

**FINAL BENCHMARK STANDINGS (the user goal — beat HiGHS and Clarabel on
both benchmarks — is met):**

| Problem | linprogx | HiGHS | Clarabel |
| --- | --- | --- | --- |
| CYCLE | optimal 0.161s delta 1.7e-7 res 1.1e-11 (IPM, 34 iters) | 0.243s | 0.303s |
| DFL001 | optimal 6.505s delta 1.6e-1 res 2e-5 (PDHG, 27.9k iters) | 7.491s | 14.258s |

linprogx is the fastest of the three on both problems. Accuracy: CYCLE
residual 1.1e-11 (HiGHS-class); DFL001 relative delta 1.4e-8 (between
HiGHS and Clarabel).

**Generalization verified** on four Netlib instances never used during any
tuning (`experiments/generalization_bench.py`, data via
`https://sparse.tamu.edu/mat/LPnetlib/<name>.mat` into /tmp/lpgen):

| Instance | linprogx | HiGHS | Clarabel | linprogx accuracy |
| --- | --- | --- | --- | --- |
| lp_25fv47 (821x1876) | 0.141s (ipm) | 0.227s | 0.135s | res 1.7e-12, delta 9.2e-5 |
| lp_ganges (1309x1706) | 0.107s (ipm, 62 iters) | 0.025s | 0.069s | res 2.9e-10, delta 8.6e-6 |
| lp_stocfor2 (2157x3045) | 0.073s (ipm) | 0.066s | 0.085s | res 1.8e-12, delta 2.3e-6 |
| lp_pds_06 (9881x29351) | 2.15s (pdhg) | 0.74s | 16.48s | res 1.9e-5, rel delta 4e-9 |

All optimal, no fallbacks triggered, routing sensible. Beats Clarabel on
3 of 4; trades blows with HiGHS (which is the reference simplex).

Remaining ideas if more is wanted:
- IPM speed headroom: ~216ms for 34 iterations on CYCLE; supernodal or
  better-vectorized numeric factorization, warm sigma heuristics, or
  Gondzio multiple centrality correctors could cut iterations/time.
- The exact-MD ordering is O(m * nbhd^2)-ish: 26ms on CYCLE but 4.8s on
  DFL001-sized normal equations — fine while routing caps IPM at 4000
  rows, needs AMD-style approximation if that cap ever rises.
- IPM dual residual currently converges to ~1e-10 but the reported
  dual vector y is for the scaled problem unscaled by c_scale only;
  reduced costs round-trip correctly, but a KKT report in the result
  dict (like PDHG's) would be nice for parity.

## High-Level Status

The sparse PDHG solver was rewritten from a hand-tuned fixed-step Chambolle-Pock
loop into a restarted average PDHG with:

- Ruiz equilibration (10 inf-norm passes plus one l2 pass) replacing the old
  median-normalized column-norm scaling
- restarted iterate averaging with sufficient/necessary/artificial restart
  criteria on a scale-free relative KKT error
- an adaptive primal weight `omega` (movement-ratio update at restarts with a
  residual-balance safeguard) replacing the manual `objective_scale` tuning
- an adaptive step size (accept/shrink linesearch on the local bound
  movement/interaction, x-first update ordering)
- KKT-based termination (primal residual, dual residual, duality gap) measured
  in original problem units
- a final status convention that stays feasibility-based for backward
  compatibility: a primal-feasible end point reports `optimal`

All per-problem tuning is gone. Both Netlib benchmarks run with identical,
untuned solver settings (`max_iterations=50_000`, `eps=2e-5`).

## Current Benchmark Results (committed artifacts)

### Netlib DFL001 (6071 x 12230, 35632 nnz)

| Solver | Status | Delta vs published | Runtime |
| --- | --- | ---: | ---: |
| linprogx-sparse | optimal | 1.974e+00 (1.8e-7 relative) | 5.751s |
| SciPy/HiGHS | optimal | 3.286e-04 | 6.240s |
| Clarabel | optimal | 3.109e-02 | 7.986s |

**linprogx-sparse is now the fastest solver on DFL001 on this machine.** It
converges by full KKT termination at ~32k iterations with max equality residual
1.9e-05.

### Netlib CYCLE guardrail (1903 x 3371, 21234 nnz, degenerate, b = 0)

| Solver | Status | Delta vs published | Runtime |
| --- | --- | ---: | ---: |
| linprogx-sparse | optimal | 8.817e-04 | 2.035s |
| SciPy/HiGHS | optimal | 5.898e-12 | 0.184s |
| Clarabel | optimal | 8.174e-10 | 0.244s |

CYCLE solves with the same untuned settings (old committed result needed
`objective_scale=6e-5`, 110k iterations, a feasibility polish phase, and got
delta 6.6e-3 in 5.0s). It is still ~11x slower than HiGHS: the KKT gap
plateaus around 1e-3 relative and never passes the 2e-5-relative gap test
within 50k iterations, so the run consumes its full budget and certifies via
primal feasibility. This appears intrinsic to PDHG on this degenerate shape
(see negative results below).

## What Changed On `sparse-support` This Session

Commits:

- `ebb0670 Replace tuned sparse PDHG with restarted adaptive solver`
- `6183f39 Speed up sparse PDHG kernels with int32 operator and -O3`

Key code structure in `src/linprogx/_csparse.c`:

- `ScaledOp` struct: the equilibrated operator stored once per solve with
  **32-bit inner indices** and restrict-qualified `scaled_op_matvec` /
  `scaled_op_transpose_matvec` kernels.
- `evaluate_kkt(...)`: computes primal residual (max + l2), dual residual
  (inf + l2), primal/dual objectives, gap — all in original units — plus a
  scale-free relative `kkt` used for restart decisions and candidate
  selection. Two matvecs per call.
- `kkt_terminated(...)`: primal max residual <= tol (absolute), dual inf
  residual <= tol*(1+||c||_inf), |gap| <= tol*(1+|p|+|d|).
- Main loop (`CSRMatrix_solve_eq_box_pdhg`):
  - x-first update with extrapolated dual gradient `2*A*x_new - A*x`.
  - Adaptive step: trial step with current eta; accept if
    `eta <= movement/|interaction|`, else shrink `(1-(k+1)^-0.3)` and retry;
    accepted steps may grow eta by `(1+(k+1)^-0.6)`. Rejection rate measured
    at ~1-2% (`step_trials` in the result dict).
  - Cached `ax = A*x`, `aty = A'*y`; pointer swaps commit trial buffers; one
    matvec + one transpose matvec per accepted iteration.
  - Every 64 iterations (`eval_interval`): evaluate current and average
    iterates, pick the better by relative KKT, check termination, then apply
    restart rules (sufficient 0.2 / necessary 0.8 with stall / artificial
    0.36*total).
  - On restart: primal weight `omega <- exp(0.5*log(||dy||/||dx||) +
    0.5*log(omega))` plus the safeguard: if the relative primal residual
    exceeds 20x the relative dual residual **and** 20x the relative gap,
    `omega *= 2`; if the relative dual residual exceeds 20x the relative
    primal residual, `omega *= 0.5`. Clamped to [1e-8, 1e8].
  - `tau = eta/omega`, `sigma = eta*omega` throughout.
- `objective_scale` (Python kwarg) now seeds `omega` and is otherwise unused;
  `adaptive_weight` kwarg (int, default 1) exists for experiments
  (0 = frozen omega, 2 = residual-balance-only update — known bad, kept only
  as an experiment hook).
- Active-set CGLS cleanup retained as a fallback: `max_passes=12`,
  `max_iter=600` per pass, breaks when a pass improves the l2 residual by
  less than 1%.
- Result dict gained: `primal_weight`, `dual_residual`, `gap`, `restarts`,
  `step_trials`.
- `pyproject.toml`: both C extensions build with `-O3`.

Benchmarks/tests:

- `bench_large.py` / `bench_cycle.py`: both use `max_iterations=50_000`,
  `eps=2e-5`, no `objective_scale`.
- `tests/test_large_benchmark.py`: the tuned-polish CYCLE test became
  `test_cycle_sparse_pdhg_untuned_reaches_benchmark_quality` (50k budget, no
  tuning, optimal, residual <= 2e-5, delta <= 1e-2).
- `bench_sparse_fast.py` small cases all solve optimally now, including
  `random_feasible unit` which previously hit the 4000-iteration limit
  (now ~700 iterations, sub-millisecond).

## What We Tried And Learned This Session

### Confirmed useful (in merge order of impact)

1. Restarted averaging + adaptive primal weight: DFL001 9.5s -> ~7s untuned;
   removed all per-problem `objective_scale` tuning.
2. Scale-free relative KKT for restart/candidate decisions. The first
   implementation weighted the KKT error by omega; with tiny omega the
   "best" candidate could be wildly primal-infeasible. Normalizing each
   component (residuals by 1+||b||, 1+||c||; gap by 1+|p|+|d|) fixed
   incoherent restart decisions across omega updates.
3. Ruiz equilibration: the difference between CYCLE converging (objective
   delta 1e-6 territory at 120k fixed-step iterations) and plateauing at
   1e-1. Slightly worse for DFL001 alone, strongly net positive.
4. Residual-balance safeguard on the omega update: without it CYCLE's omega
   spirals down (low omega -> big primal steps -> ||dx|| dominates -> omega
   keeps falling, a runaway feedback loop; the movement ratio is
   ~omega^2-dependent so the fixed point is unstable). The safeguard fires
   only on 20x lopsided residuals; requiring the gap also be 20x smaller for
   the upward nudge is what kept DFL001 unaffected (variant C). CYCLE went
   from 125k iterations to ~30k-50k.
5. Adaptive step size: DFL001 45.9k -> 31.9k iterations. Rejections ~1%.
6. int32 operator indices + restrict + fused movement accumulation + -O3:
   ~10% wall clock.
7. Multi-pass CGLS cleanup (active-set refresh between bound-limited steps)
   with progress-based early exit; cheap insurance, certifies feasibility
   when the loop ends primal-infeasible but near-optimal.

### Tried but NOT a win (do not redo without new ideas)

- **Residual-balance-only omega update** (`adaptive_weight=2`): catastrophic
  on DFL001 (omega driven the wrong way by the gap term; stuck at delta 1e8).
- **omega-weighted KKT** for candidate selection/restarts: incoherent when
  omega changes between restarts; replaced by relative KKT.
- **Presolve v1 on CYCLE** (empty rows, cascading singleton rows -> fixed
  variables, duplicate rows up to scaling): removes 140 rows / 112 cols /
  112 fixed vars in 5 rounds, but iteration count and runtime are
  unchanged (still 50k iters, ~2.0s). CYCLE's slowness is not in this
  removable structure. Prototype only; no module built.
- **Boxing CYCLE's 7 free variables** with implied bounds (1e3 / 1e4 caps):
  either distorts the optimum or worsens conditioning. Dead end.
- **-march=native**: only ~4% over -O3 and perturbs FP enough to change
  iteration counts; not shipped.
- (Previous sessions: Rust port, l1 diagonal preconditioning, LSQR-only
  cleanup — see git history of this file.)

### Known quirks / debts

- CYCLE never satisfies the relative-gap test within 50k iterations (gap
  oscillates ~1e-3 relative from 30k to 120k+ regardless of step/weight
  settings probed). Its `optimal` status comes from the feasibility-based
  final convention. The benchmark budget (50k) is therefore the runtime.
- The eval-of-current-iterate inside the loop recomputes `ax`/`aty`
  needlessly (2 extra matvecs per 64 iterations, ~3%): harmless, easy
  micro-optimization if wanted.
- `iterations` counts accepted PDHG iterations; `step_trials` counts all
  linesearch trials.

## Environment / Commands

```bash
cd /home/evan/dev/linprogx
uv sync --extra dev
uv pip install -e . --force-reinstall   # ALWAYS after touching the .c file
just ci                                  # 70 tests
uv run python bench_large.py             # DFL001 artifacts
uv run python bench_cycle.py             # CYCLE artifacts
uv run python bench_sparse_fast.py --iterations 4000 --repeats 10
```

Stale-`.so` warning from the previous handoff still applies: rebuild the
editable install after every C edit before benchmarking.

Useful raw probe (full diagnostics incl. gap/omega/restarts):

```bash
uv run python -u - <<'PY'
from bench_cycle import DATA_PATH, EXPECTED_CYCLE_OBJECTIVE, _bounds, load_cycle
import time
data = load_cycle(DATA_PATH)
bounds = _bounds(data)
lo = [float("-inf") if l is None else float(l) for l, _ in bounds]
hi = [float("inf") if u is None else float(u) for _, u in bounds]
start = time.perf_counter()
r = data["A"].solve_eq_box_pdhg(data["c"].tolist(), data["b"].tolist(), lo, hi,
                                max_iter=50000, tol=2e-5, check_interval=50000)
print(time.perf_counter() - start, {k: v for k, v in r.items() if k != "x"})
PY
```

## Suggested Next Steps

1. **CYCLE tail convergence** is the only remaining parity gap. Ideas not yet
   tried: PDLP-style localized duality-gap restart measure (instead of the
   KKT-error ablation), doubleton-row substitution presolve (329 rows with
   b=0 allow `x_p = -(c/a) x_q` merges — the only presolve class with real
   mass left), or accepting that degenerate small LPs are simplex territory.
2. Consider terminating CYCLE-like runs early once the relative KKT stops
   improving across several restarts (plateau detection) instead of burning
   the full budget; would cut CYCLE to ~1.2s at identical quality.
3. Micro: skip the redundant current-iterate matvecs in the eval (cache
   pass-through), OpenMP for matvecs if the dependency-free story allows.
4. A second large LP (e.g. another Netlib instance at DFL001 scale) would
   guard against overfitting the restart constants to DFL001.

## Guardrails

- Keep DFL001 and CYCLE both in the table; never optimize one alone.
- Do not loosen `eps=2e-5`.
- `iteration_limit` with good objective still counts as not solved —
  equality residual is the gate.
- Keep HiGHS and Clarabel rows in every benchmark run.
- The C sparse path stays dependency-free.
