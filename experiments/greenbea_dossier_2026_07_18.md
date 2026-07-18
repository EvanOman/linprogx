# greenbea dossier — ideation evidence pack (2026-07-18)

The campaign's last standing loss. This document is the complete evidence
base for generating NEW idea classes. Everything in "Closed axes" is
measured dead — proposals on those axes are rejected on sight.

## The cell

- LPnetlib lp_greenbea. Board verdict ~1.69-1.74 across all v3 waves
  (median-of-hosts, AWS us-west-2): our 0.62-0.64s vs HiGHS 0.36-0.38s.
  Local quiet-box: ours ~0.42s vs HiGHS ~0.24s. A flip needs ~-41%.
- Certified route: dual simplex (Dantzig leaving), 4,399 pivots,
  ~95 us/pivot end-to-end. HiGHS: 2,836 pivots, ~86 us/pivot.
- Our per-pivot DS wall split (local census): pivot-row 24.8%, BTRAN
  18.9%, FTRAN 17.9%, ratio test 14.9%, reduced-cost update 9.7%,
  LU update 6.1%, refactorization 5.5%.

## Structure (post-presolve, current stack)

- Presolved shape 1,525 x 3,868 x 23,274 (raw 2,392 x 5,598 x 31,070).
  Presolve removes 867 rows / 1,730 cols (forcing columns 1,102,
  doubletons 438, fixed 113, singleton rows 47, col singletons 23,
  dup cols 7).
- 338 remaining singleton columns, ALL with non-redundant bounds
  (190 slack-like [0,inf), 148 boxed, 0 free). The problem is already
  at a bound-propagation fixpoint: eliminating all 338 under ranged-row
  semantics yields 10 redundant rows, 0 propagated tightenings, 0 fixes.
- Row degree p50/p99 = 5/91.8; col degree p50/p99 = 6/17. Density 0.39%.
- CORRECTED 2026-07-18 (probe_activeset): 61.6% of variables are truly
  nonbasic-at-bound at optimum (2,381/3,868); the earlier 83.2% figure
  counted 681 degenerate-BASIC columns sitting at bounds by value. The
  IPM primal cannot distinguish the two classes (measured prediction
  precision ceiling ~0.76). ~1/4,399 degenerate pivots.
- DS support density: rho p50 897, alpha p50 3,625, ratio-candidates
  p50 182; consecutive-pivot support overlap 96-99.97%.

## The HiGHS gap, decomposed (all measured, runtime-behavioral)

- HiGHS presolve-on raw: 2,836 pivots (DuPh1 1,448 / DuPh2 1,376).
- HiGHS presolve-off on OUR reduction: 3,309 pivots (DuPh1 1,655 /
  DuPh2 1,633). So: 473 pivots are presolve-geometry, 1,090 are
  simplex-internal.
- Geometry is non-transferable BOTH ways: our DS on HiGHS's reduction
  does 5,222 pivots (823 WORSE); our aggregation reaching HiGHS's shape
  (936 rows) makes our DS do MORE pivots (+24% at target shape; best
  -7% at intermediate fill cap).
- On identical input, our Dantzig (4,399) crushes HiGHS-Dantzig (12,279)
  and our exact Forrest-Goldfarb DSE (4,675, correct crash-basis gamma,
  exact update) does NOT reproduce HiGHS-DSE's 3,309. Their edge lives
  in DSE-adjacent machinery we have not identified (phase structure,
  ratio test interplay, perturbation?), not the textbook rule.
- HiGHS crash strategies 0-9: all 3,309 (no effect). Their bound-flip
  ratio test (longest-step BFRT) on ours: -101 pivots only.

## Closed axes (do NOT propose these)

1. Presolve depth of any kind: aggregation (shape parity != pivot
   parity, proven bidirectionally), bounded-singleton/ranged rows
   (propagation fixpoint), parallel/dominated columns (7 exact dups
   only; 3.3% ceiling), fixpoint re-staging (shipped; helps others).
2. Leaving rules: Dantzig(4,399), exact FG-DSE(4,675), Devex(6,807),
   rule2(11,948), rule3(15,188), rule4(fails). Family closed.
3. Starting bases: HiGHS Phase-1 transfer -> 3,529 pivots but FLAT wall
   (densifies solves 88.8 -> 113.1 us/pivot); IPM-crossover warm starts
   (super-basic: singular; Bixby from iterate: 4,489-5,412, several
   dual_infeasible, never beats cold). Even 2,836 pivots at transferred
   density projects 0.321s vs HiGHS 0.266s: pivot parity and per-pivot
   parity TRADE AGAINST each other on every transferred trajectory.
4. Ratio test: longest-step BFRT -101; Harris is shipped.
5. Per-pivot kernels: within-pivot support reuse SHIPPED; cross-pivot
   reuse KILLED on algebra (alpha'_k needs old alpha_k); dense-U FTRAN
   dead in all three bandwidth regimes; Suhl bounded pivot search
   SHIPPED (-96% search); Forrest-Tomlin SHIPPED; refactor cadence
   validated at the current size; block-row uplook gate SHIPPED.
6. IPM route: stalls with a PINNED dual-certificate failure — primal
   nearly converges (residual 7.9e-10, mu 3e-9 at iter 58) but nine
   one-sided columns stay dual-sign infeasible (floor 1.8e-6),
   certificate gap inf; Newton direction then goes non-finite (now
   guarded). Adaptive primal-dual regularization cannot move the
   pinned columns (199 iters, obj rel err 1.2e-3). Row-space
   regularization fails outright.
7. PDHG route: never certified competitive on this size/class.

## Constraints (inviolable)

- Never read any solver's source code (papers/textbooks fine).
- No network access in workers. No per-problem tuning (global
  mechanisms/thresholds only). eps=2e-5 fixed. Certificate-backed
  optimality only. Honest reporting; falsifier-first with kill criteria.

## What a winning idea must explain

Any proposal must say which term it attacks in:
wall = pivots x us/pivot (DS route), or an entirely different route
that certifies. It must be compatible with the trade-against evidence
(transferred trajectories densify), and must not be a re-skin of a
closed axis. Ceiling arithmetic required: greenbea needs -41%.
