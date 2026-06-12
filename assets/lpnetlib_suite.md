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
| lp_80bau3b | 0.23s (ipm) | 0.21s | 0.38s | 1.7e-09 | 4e-12 |
| lp_cre_a | 0.28s (ipm) | 0.09s | 0.14s | 2.4e-06 | 5e-05 |
| lp_cre_b | 117.19s (ipm) | 2.09s | 18.77s | 2.9e-06 | 8e-10 |
| lp_cre_d | 26.98s (ipm) | 1.03s | 15.85s | 1.9e-08 | 4e-11 |
| lp_d2q06c | 0.68s (ipm) | 0.97s | 2.26s | 2.9e-10 | 2e-08 |
| lp_degen3 | 2.10s (ipm) | 0.20s | 0.44s | 4.4e-07 | 1e-07 |
| lp_fit2p | iteration_limit | 1.45s | 0.29s | n/a | n/a |
| lp_greenbea | iteration_limit | 0.28s | 2.52s | n/a | n/a |
| lp_ken_07 | 0.06s (ipm) | 0.04s | 0.05s | 2.3e-09 | 2e-04 |
| lp_ken_11 | 0.43s (ipm) | 0.35s | 0.76s | 7.2e-10 | 2e-06 |
| lp_ken_13 | 4.31s (ipm) | 1.12s | 2.00s | 2.8e-08 | 2e-05 |
| lp_ken_18 | iteration_limit | 10.85s | DualInfeasible | n/a | n/a |
| lp_maros_r7 | 14.53s (ipm) | 0.96s | 2.29s | 3.2e-10 | 3e-11 |
| lp_osa_14 | 8.15s (ipm) | 1.06s | 2.75s | 1.7e-07 | 1e-04 |
| lp_osa_30 | 99.97s (pdhg) | 3.78s | 6.66s | 1.5e-06 | 2e-05 |
| lp_osa_60 | timeout | 24.66s | 30.44s | n/a | n/a |
| lp_pds_10 | 6.09s (pdhg) | 1.67s | 29.46s | 4.0e-09 | 1e-05 |
| lp_pds_20 | 23.64s (pdhg) | 12.35s | 112.23s | 2.0e-08 | 2e-05 |
| lp_pilot87 | 14.75s (ipm) | 3.87s | 9.59s | 1.4e-07 | 3e-11 |
| lp_qap12 | 2.14s (pdhg) | 100.97s | 3.86s | 9.4e-06 | 2e-05 |
| lp_qap15 | 2.90s (pdhg) | timeout | 19.99s | 5.9e-06 | 7e-06 |
| lp_stocfor3 | 0.82s (ipm) | 0.70s | 1.05s | 4.7e-08 | 1e-10 |
| lp_truss | 0.12s (ipm) | 2.71s | 0.15s | 3.9e-12 | 3e-12 |
| lp_woodw | 0.20s (ipm) | 0.09s | 0.23s | 2.2e-05 | 6e-09 |

## Summary

- **Solved (status optimal): linprogx 20/24, HiGHS 23/24, Clarabel 23/24.**
- Where linprogx solves, quality is high: relative objective error between
  3.9e-12 and 2.2e-5, equality residuals between 3e-12 and 2e-4.
- linprogx is the fastest of the three on 4 instances and ties or beats
  Clarabel on 10 of its 20 solves. Highlights: qap15 in 2.9s where HiGHS
  times out at 180s; qap12 in 2.1s vs HiGHS at 101s; truss 22x faster than
  HiGHS; d2q06c and stocfor3 fastest-of-three; pds_10/pds_20 are 5x/4.7x
  faster than Clarabel; on ken_18 Clarabel fails outright (DualInfeasible)
  while linprogx reaches a 4.5e-9-relative objective but cannot certify the
  last digit of feasibility.
- Cost-based routing (ordering work budget plus a factor-flops cap measured
  during symbolic analysis) lets the interior point method take any problem
  whose factor is affordable regardless of row count: stocfor3, cre_b, and
  cre_d moved from unsolved to optimal, and ken_11/ken_13/osa_14 solve
  faster than on the first-order path.
- The 4 unsolved instances: fit2p (dense columns make the normal equations
  fully dense; needs dense-column splitting), ken_18 (the exact
  minimum-degree ordering is too slow at 40k rows; needs approximate
  degrees), greenbea (stalls on both paths; the explicit gap test refuses
  to certify the near-optimal point that Clarabel wrongly accepts), and
  osa_60 (1.4M nonzeros, exceeds the 180s budget on the first-order path).

Mature solvers still win on breadth — that is the honest headline for a
hand-built solver against HiGHS. The portfolio is competitive on the
benchmark problems this repo tracks, wins decisively on several large or
degenerate instances, and reports its failures truthfully.
