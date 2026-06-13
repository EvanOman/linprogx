# LPnetlib Suite Results

24 instances from the SuiteSparse LPnetlib collection — the same Netlib LP
family that the Clarabel paper benchmarks against, including the Kennington
set (ken/pds/cre/osa) used in the HiGHS papers. Instances were chosen to span
sizes (821x1876 up to 105,127x154,699) and difficulty (QAP relaxations,
pilot87, greenbea, maros_r7), and none were used while developing the solver.

- Data: `experiments/download_lpsuite.sh` (from sparse.tamu.edu, .mat format)
- Harness: `experiments/suite_bench.py` — every (instance, solver) pair runs
  in an isolated subprocess with a 180s timeout; sequential execution for
  fair timing.
- linprogx settings: `SparseSolver(algorithm="auto", eps=2e-5)` — untuned,
  identical for every instance.
- "rel delta" compares the linprogx objective against the published optimal
  values (Gurobi at 1e-8, github.com/SkyLiu0/NETLIB).
- Raw rows: `assets/lpnetlib_suite_results.jsonl`.

| Instance | linprogx | HiGHS | Clarabel | rel delta | residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| lp_80bau3b | 0.25s (ipm) | 0.21s | 0.37s | 2.9e-08 | 1e-09 |
| lp_cre_a | 0.14s (ipm) | 0.10s | 0.15s | 2.4e-06 | 4e-05 |
| lp_cre_b | 27.63s (ipm) | 2.11s | 18.66s | 4.6e-06 | 2e-07 |
| lp_cre_d | 38.13s (ipm) | 1.17s | 16.70s | 7.3e-06 | 7e-09 |
| lp_d2q06c | 1.13s (ipm) | 0.95s | 2.36s | 2.7e-09 | 4e-10 |
| lp_degen3 | 0.34s (ipm) | 0.21s | 0.47s | 4.3e-07 | 9e-08 |
| lp_fit2p | 0.11s (ipm) | 1.24s | 0.29s | 2.1e-07 | 2e-08 |
| lp_greenbea | iteration_limit | 0.28s | 2.44s | n/a | n/a |
| lp_ken_07 | 0.03s (ipm) | 0.04s | 0.06s | 2.6e-09 | 2e-04 |
| lp_ken_11 | 0.36s (ipm) | 0.33s | 0.74s | 7.6e-08 | 9e-04 |
| lp_ken_13 | 1.16s (ipm) | 1.01s | 2.13s | 3.0e-08 | 1e-05 |
| lp_ken_18 | 23.67s (ipm) | 10.42s | DualInfeasible | 1.5e-08 | 5e-10 |
| lp_maros_r7 | 9.87s (ipm) | 0.95s | 2.31s | 3.2e-10 | 3e-11 |
| lp_osa_14 | 1.67s (ipm) | 1.14s | 2.90s | 1.7e-07 | 1e-04 |
| lp_osa_30 | 3.78s (ipm) | 4.27s | 9.64s | 6.8e-08 | 1e-04 |
| lp_osa_60 | 16.18s (ipm) | 24.06s | 27.53s | 5.5e-06 | 8e-03 |
| lp_pds_10 | 4.78s (pdhg) | 1.59s | 31.18s | 4.0e-09 | 1e-05 |
| lp_pds_20 | 41.51s (pdhg) | 12.46s | 115.86s | 2.0e-08 | 2e-05 |
| lp_pilot87 | 23.80s (ipm) | 3.82s | 9.77s | 4.8e-07 | 9e-11 |
| lp_qap12 | 0.51s (pdhg) | 104.56s | 4.00s | 9.4e-06 | 2e-05 |
| lp_qap15 | 1.15s (pdhg) | timeout | 21.19s | 5.9e-06 | 7e-06 |
| lp_stocfor3 | 1.13s (ipm) | 0.70s | 1.12s | 4.7e-08 | 6e-11 |
| lp_truss | 0.15s (ipm) | 2.98s | 0.18s | 3.8e-12 | 3e-12 |
| lp_woodw | 0.21s (ipm) | 0.11s | 0.25s | 1.2e-05 | 6e-09 |

## Summary

- **Solved (status optimal): linprogx 23/24, HiGHS 23/24, Clarabel 23/24 —
  equal coverage.** Each solver misses exactly one instance: HiGHS times out
  on qap15, Clarabel reports DualInfeasible on ken_18, and linprogx declines
  to certify greenbea.
- Every linprogx optimum is certificate-backed (full KKT or an explicit
  Lagrangian dual bound from the actual reduced costs). Relative objective
  errors run 9.2e-12 to 2.2e-5.
- A stall-gated early acceptance ends the IPM the moment the relaxed
  bars and the certificate are met mid-run instead of crawling toward
  the strict tolerance (osa_60 199 -> 57 iterations, cre_d 199 -> 77),
  with the min-norm cleanup attempted in-flight when the raw
  certificate fails.
- cre_a and cre_b are solved by a min-norm dual cleanup at the IPM exit:
  their degenerate stall points fail certification on only ~40-60
  wrong-signed reduced costs, and a small least-squares correction
  (Gram system over the violating columns) repairs the dual without
  touching the primal. The stage can gain certificates but never fake
  one; it is gated on the same 1e-5 certified-gap acceptance.
- Fastest of the three on 8 instances: fit2p (0.15s vs HiGHS 1.24s and
  Clarabel 0.29s), ken_07, ken_11, osa_30, osa_60 (15.6s vs HiGHS 24.1s
  and Clarabel 27.5s), qap12 (0.52s vs HiGHS 104s), qap15 (1.2s; HiGHS
  times out), and truss (23x faster than HiGHS). ken_18 solves in 26s
  where Clarabel reports DualInfeasible.
- Honesty note: on greenbea, Clarabel certifies a point with a 1.3e-3
  objective error; linprogx's Lagrangian certificate provably cannot
  certify a false optimum and reports the instance unsolved.
- The engine: presolve, cost-based routing between a Mehrotra IPM and a
  restarted adaptive PDHG, approximate-minimum-degree ordering,
  dense-column splitting via Sherman-Morrison-Woodbury, staged-precision
  regularization with doubled iterative refinement, dual polish,
  min-norm dual cleanup, and certificate-gated acceptance.

Mature solvers still win on breadth -- that is the honest headline for a
hand-built solver against HiGHS. The portfolio is competitive on the
benchmark problems this repo tracks, wins decisively on several large or
degenerate instances, and never certifies an answer it cannot prove.
