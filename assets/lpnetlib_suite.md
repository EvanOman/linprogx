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
| lp_80bau3b | 0.28s (ipm) | 0.17s | 0.29s | 1.7e-09 | 4e-12 |
| lp_cre_a | 0.29s (ipm) | 0.09s | 0.13s | 2.4e-06 | 5e-05 |
| lp_cre_b | iteration_limit | 1.88s | 16.02s | n/a | n/a |
| lp_cre_d | iteration_limit | 1.02s | 13.90s | n/a | n/a |
| lp_d2q06c | 0.63s (ipm) | 0.91s | 2.03s | 2.9e-10 | 2e-08 |
| lp_degen3 | 3.99s (ipm) | 0.21s | 0.40s | 4.4e-07 | 1e-07 |
| lp_fit2p | iteration_limit | 1.17s | 0.24s | n/a | n/a |
| lp_greenbea | iteration_limit | 0.25s | 2.12s | n/a | n/a |
| lp_ken_07 | 0.06s (ipm) | 0.04s | 0.05s | 2.3e-09 | 2e-04 |
| lp_ken_11 | 1.15s (pdhg) | 0.28s | 0.59s | 1.8e-12 | 5e-06 |
| lp_ken_13 | 4.35s (pdhg) | 0.86s | 1.65s | 1.8e-10 | 2e-05 |
| lp_ken_18 | iteration_limit | 8.31s | DualInfeasible | n/a | n/a |
| lp_maros_r7 | 12.74s (ipm) | 0.80s | 1.64s | 3.2e-10 | 3e-11 |
| lp_osa_14 | 7.43s (ipm) | 1.02s | 1.88s | 1.7e-07 | 1e-04 |
| lp_osa_30 | 66.49s (pdhg) | 3.54s | 5.00s | 1.5e-06 | 2e-05 |
| lp_osa_60 | timeout | 17.38s | 22.47s | n/a | n/a |
| lp_pds_10 | 2.70s (pdhg) | 1.25s | 24.20s | 4.0e-09 | 1e-05 |
| lp_pds_20 | 17.16s (pdhg) | 10.29s | 92.29s | 2.0e-08 | 2e-05 |
| lp_pilot87 | 35.79s (ipm) | 3.57s | 8.16s | 1.4e-07 | 3e-11 |
| lp_qap12 | 4.56s (pdhg) | 96.74s | 3.02s | 9.4e-06 | 2e-05 |
| lp_qap15 | 0.65s (pdhg) | timeout | 14.44s | 5.9e-06 | 7e-06 |
| lp_stocfor3 | iteration_limit | 0.55s | 0.92s | n/a | n/a |
| lp_truss | 0.12s (ipm) | 2.72s | 0.17s | 3.9e-12 | 3e-12 |
| lp_woodw | 0.15s (ipm) | 0.09s | 0.21s | 2.2e-05 | 6e-09 |

## Summary

- **Solved (status optimal): linprogx 17/24, HiGHS 23/24, Clarabel 23/24.**
- Where linprogx solves, quality is high: relative objective error between
  1.8e-12 and 2.2e-5, equality residuals between 3e-12 and 2e-4.
- linprogx is the fastest of the three on 3 instances and ties or beats
  Clarabel on 7 of its 17 solves. Highlights: qap15 in 0.65s where HiGHS
  times out at 180s; qap12 in 4.6s vs HiGHS at 96.7s; truss 22x faster than
  HiGHS; pds_10/pds_20 are 9x/5x faster than Clarabel; on ken_18 Clarabel
  fails outright (DualInfeasible) while linprogx reaches a 4.5e-9-relative
  objective but cannot certify the last digit of feasibility.
- The 7 unsolved instances are first-order (PDHG) tails: cre_b, cre_d,
  fit2p, and ken_18 end with residuals between 9e-5 and 1e-3 (just above the
  2e-5 bar) and excellent objectives; greenbea and stocfor3 stall further
  out; osa_60 (1.4M nonzeros) exceeds the 180s budget.
- One correctness finding: on greenbea, an interior point iterate with a
  small barrier mu but a 1.8e-6-infeasible dual hides a 1.3e-3 objective
  error. Clarabel reports that point as Solved; linprogx's acceptance test
  requires an explicit primal-dual gap bound and reports the instance
  unsolved instead.

Mature solvers still win on breadth — that is the honest headline for a
hand-built solver against HiGHS. The portfolio is competitive on the
benchmark problems this repo tracks, wins decisively on several large or
degenerate instances, and reports its failures truthfully.
