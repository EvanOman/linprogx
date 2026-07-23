# Creative attack — six mandates (2026-07-21)

Each agent executes ONE mandate. The dossier
(experiments/creative_attack_dossier_2026_07_21.md) binds.

## C1 — Replace the factorization data structure
The chase itself is the enemy. Design + falsify an alternative factor
representation for B (1,525 rows, factor nnz ~23-30k): candidates —
fully-dense fp64 factor (2.3M flops/solve but pure contiguous TRSV
streams: measure whether streamed O(n^2) beats latency-bound O(nnz);
note K3 swept SPARSE storage densely, never a truly dense-STORED factor),
register-blocked/ELLPACK-style formats, split row/col-major halves per
solve direction. Probe: implement the most promising as a standalone
factor+solve microbenchmark first (correctness vs the existing LU on
captured bases from the trajectory), THEN in-loop if the microbench beats
the current solves by >=30%. Kill below that.

## C2 — Overlap BTRAN and FTRAN
Within one pivot, BTRAN (rho) and FTRAN (alpha) are independent solves —
36.8% of wall, never overlapped. Probe: run them on two threads
(persistent pool, the a2 lesson — no per-call spawn) or interleave their
chases on one core (two independent dependency chains hide each other's
latency — this differs from LS-B, which interleaved WITHIN one solve).
Deterministic, byte-identical results required (they write disjoint
outputs). Kill if combined solve wall improves <12%.

## C3 — Trajectory shaping by scaling
HiGHS's scale_strategy=4 changed ITS pivot counts on greenbea. Our
equilibration is fixed 10-pass Ruiz. Probe (global rules only): sweep
scaling FAMILIES (Curtis-Reid, geometric mean, pow-2 rounding, combined)
as a global option; measure OUR DS pivots + wall on greenbea AND the DS
sentinels + oracle equality. The 26th-settled kill was Ruiz EARLY-EXIT
(pass count); scaling ALGORITHM was never swept. Kill if no family cuts
greenbea wall >=10% with sentinels clean.

## C4 — Cost perturbation with exact recovery
Perturb c by tiny structured amounts (global rule; e.g. scaled by column
norms, deterministic seed), solve the perturbed LP (possibly many fewer
pivots — degeneracy/near-ties in pricing may be stalling us in ways the
leaving-rule sweep could not see), then RECOVER exactness: fix the final
basis, re-solve reduced costs with true c, verify optimality of that
basis for the true problem at eps=2e-5 (if not optimal, continue DS from
there with true c — count all recovery pivots). Probe: perturbation-size
sweep, pivots + recovery + wall. Kill if best total wall improvement
<10% or recovery ever fails certificates.

## C5 — PDHG-approximated auxiliary
The pipeline route needs the auxiliary basis for <=0.05s (K7: exact
simplex costs 0.157-0.215s). The auxiliary is HOMOGENEOUS (min c'x,
Ax=0, unit boxes) and we only need a GOOD BASIS GUESS, not exact optimum:
run our existing sparse PDHG on the auxiliary for a bounded iteration
budget, crossover its approximate support to a basis (Bixby-style from
the iterate — the G2 machinery exists), warm-start the main DS via the
injection hook. Sweep budget; measure end-to-end pipeline (PDHG-aux +
crossover + warm DS + certificate) vs 0.37s baseline. Kill if end-to-end
never beats baseline by >=10%.

## C6 — Dual-of-the-dual: primal simplex route
Never certified on greenbea: OUR primal simplex family (the dense
two-phase exists; a sparse primal variant could reuse the DS kernels
with roles swapped). HiGHS-primal did 13,809 pivots on our reduction —
but that is THEIR primal. Cheap probe FIRST: run our existing dense
two-phase simplex path on presolved greenbea (bounded time) — pivots,
wall, status. If within 2x of flip-relevant, assess what a sparse primal
using existing kernels would cost. Kill quickly if the pivot count is
wildly high (>8,000) — report and stop; this is the cheapest mandate,
budget accordingly.
