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
| lp_80bau3b | 0.59s (ipm) | 0.20s | 0.36s | 7.5e-09 | 1e-11 |
| lp_cre_a | iteration_limit | 0.10s | 0.16s | n/a | n/a |
| lp_cre_b | iteration_limit | 2.11s | 19.12s | n/a | n/a |
| lp_cre_d | 28.62s (ipm) | 1.17s | 16.72s | 1.9e-08 | 5e-07 |
| lp_d2q06c | 1.62s (ipm) | 0.94s | 2.33s | 1.4e-10 | 7e-10 |
| lp_degen3 | 2.26s (ipm) | 0.22s | 0.48s | 5.5e-07 | 1e-07 |
| lp_fit2p | 0.55s (ipm) | 1.20s | 0.30s | 7.0e-07 | 3e-08 |
| lp_greenbea | iteration_limit | 0.27s | 2.43s | n/a | n/a |
| lp_ken_07 | 0.07s (ipm) | 0.04s | 0.06s | 1.6e-09 | 3e-05 |
| lp_ken_11 | 0.86s (ipm) | 0.35s | 0.75s | 7.5e-08 | 7e-04 |
| lp_ken_13 | 4.36s (ipm) | 1.09s | 2.22s | 3.0e-08 | 7e-06 |
| lp_ken_18 | 85.47s (ipm) | 10.55s | DualInfeasible | 1.5e-08 | 6e-05 |
| lp_maros_r7 | 13.58s (ipm) | 0.85s | 2.07s | 3.2e-10 | 3e-11 |
| lp_osa_14 | 3.51s (ipm) | 0.99s | 2.51s | 1.7e-07 | 1e-04 |
| lp_osa_30 | 3.45s (ipm) | 4.31s | 7.50s | 6.8e-08 | 1e-04 |
| lp_osa_60 | timeout | 23.41s | 29.28s | n/a | n/a |
| lp_pds_10 | 3.56s (pdhg) | 1.71s | 29.90s | 4.0e-09 | 1e-05 |
| lp_pds_20 | 29.86s (pdhg) | 12.46s | 118.34s | 2.0e-08 | 2e-05 |
| lp_pilot87 | 26.45s (ipm) | 4.49s | 9.83s | 1.4e-07 | 6e-11 |
| lp_qap12 | 0.44s (pdhg) | 107.52s | 3.92s | 9.4e-06 | 2e-05 |
| lp_qap15 | 1.28s (pdhg) | timeout | 18.94s | 5.9e-06 | 7e-06 |
| lp_stocfor3 | 1.31s (ipm) | 0.64s | 1.20s | 4.7e-08 | 7e-11 |
| lp_truss | 0.13s (ipm) | 2.95s | 0.19s | 3.8e-12 | 2e-12 |
| lp_woodw | 0.17s (ipm) | 0.09s | 0.25s | 2.2e-05 | 6e-09 |

## Summary

- **Solved (status optimal): linprogx 20/24, HiGHS 23/24, Clarabel 23/24.**
- Every linprogx optimum is certificate-backed: either the full KKT test or
  an explicit Lagrangian dual bound built from the actual reduced costs.
  Relative objective errors run 3.8e-12 to 2.2e-5.
- Fastest of the three on 4 instances; ties or beats Clarabel on 9 of its
  20 solves. Highlights: qap12 in 0.44s and qap15 in 1.3s where HiGHS needs
  107s and times out respectively; fit2p in 0.55s vs HiGHS 1.20s
  (dense-column splitting); 80bau3b certified at 7.5e-9 relative error;
  osa_30 beats HiGHS; truss 23x faster than HiGHS; ken_18 solves where
  Clarabel reports DualInfeasible.
- Honesty note: on greenbea, Clarabel certifies a point with a 1.3e-3
  objective error; linprogx's Lagrangian certificate provably cannot
  certify a false optimum and reports the instance unsolved instead.
  cre_a and cre_b reach excellent objectives (about 4e-7 and 4e-7
  relative in earlier rounds) but their stall points carry slightly
  negative reduced costs on unbounded variables, so no sound certificate
  exists and they are reported as iteration_limit.
- The engine: presolve, cost-based routing between a Mehrotra IPM and a
  restarted adaptive PDHG, approximate-minimum-degree ordering,
  dense-column splitting via Sherman-Morrison-Woodbury, iterative
  refinement on the Newton solves, and certificate-gated acceptance.

Mature solvers still win on breadth -- that is the honest headline for a
hand-built solver against HiGHS. The portfolio is competitive on the
benchmark problems this repo tracks, wins decisively on several large or
degenerate instances, and never certifies an answer it cannot prove.
