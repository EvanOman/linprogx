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
| lp_80bau3b | iteration_limit | 0.22s | 0.42s | n/a | n/a |
| lp_cre_a | 0.38s (ipm) | 0.12s | 0.17s | 3.9e-07 | 5e-05 |
| lp_cre_b | 154.23s (ipm) | 2.46s | 23.02s | 4.4e-07 | 2e-08 |
| lp_cre_d | 35.29s (ipm) | 1.38s | 19.74s | 1.9e-08 | 5e-07 |
| lp_d2q06c | 0.84s (ipm) | 1.03s | 2.69s | 1.4e-10 | 7e-10 |
| lp_degen3 | 4.64s (ipm) | 0.30s | 0.56s | 5.5e-07 | 1e-07 |
| lp_fit2p | 0.59s (ipm) | 1.42s | 0.38s | 7.0e-07 | 3e-08 |
| lp_greenbea | iteration_limit | 0.31s | 2.65s | n/a | n/a |
| lp_ken_07 | 0.09s (ipm) | 0.06s | 0.07s | 1.6e-09 | 3e-05 |
| lp_ken_11 | 1.07s (ipm) | 0.41s | 1.17s | 7.5e-08 | 7e-04 |
| lp_ken_13 | 4.61s (ipm) | 1.16s | 3.23s | 3.0e-08 | 7e-06 |
| lp_ken_18 | 139.52s (ipm) | 12.85s | DualInfeasible | 1.5e-08 | 6e-05 |
| lp_maros_r7 | 14.30s (ipm) | 0.92s | 2.86s | 3.2e-10 | 3e-11 |
| lp_osa_14 | 5.56s (ipm) | 1.22s | 4.35s | 1.7e-07 | 1e-04 |
| lp_osa_30 | 4.46s (ipm) | 5.45s | 10.12s | 6.8e-08 | 1e-04 |
| lp_osa_60 | timeout | 33.15s | 38.45s | n/a | n/a |
| lp_pds_10 | 3.99s (pdhg) | 1.99s | 37.29s | 4.0e-09 | 1e-05 |
| lp_pds_20 | 47.75s (pdhg) | 14.48s | 152.83s | 2.0e-08 | 2e-05 |
| lp_pilot87 | 33.29s (ipm) | 3.96s | 12.28s | 1.4e-07 | 6e-11 |
| lp_qap12 | 0.44s (pdhg) | 110.03s | 5.32s | 9.4e-06 | 2e-05 |
| lp_qap15 | 1.43s (pdhg) | timeout | 26.33s | 5.9e-06 | 7e-06 |
| lp_stocfor3 | 1.35s (ipm) | 0.84s | 1.63s | 4.7e-08 | 7e-11 |
| lp_truss | 0.19s (ipm) | 3.31s | 0.26s | 3.8e-12 | 2e-12 |
| lp_woodw | 0.24s (ipm) | 0.12s | 0.29s | 2.2e-05 | 6e-09 |

## Summary

- **Solved (status optimal): linprogx 21/24, HiGHS 23/24, Clarabel 23/24.**
- Where linprogx solves, quality is high: relative objective error between
  3.8e-12 and 2.2e-5.
- linprogx is the fastest of the three on 5 instances and ties or beats
  Clarabel on 11 of its 21 solves. Highlights: qap12 in 0.44s and qap15 in
  1.4s where HiGHS needs 110s and times out respectively; fit2p in 0.59s vs
  HiGHS 1.42s (dense-column splitting); osa_30 via the IPM in 4.5s vs HiGHS
  5.5s; truss 17x faster than HiGHS; pds_20 3.2x faster than Clarabel;
  ken_18 solves where Clarabel reports DualInfeasible.
- The engine: presolve, cost-based routing between a Mehrotra IPM and a
  restarted adaptive PDHG, approximate-minimum-degree ordering, dense-column
  splitting via Sherman-Morrison-Woodbury, iterative refinement on the
  Newton solves, and gap-certified acceptance.
- The 3 unsolved instances: greenbea and 80bau3b (IPM endgame stalls; the
  explicit gap test refuses to certify near-optimal points -- on greenbea
  Clarabel certifies a point with a 1.3e-3 objective error), and osa_60
  (1.4M nonzeros, exceeds the 180s budget on the first-order path at
  residual ~5e-5).

Mature solvers still win on breadth -- that is the honest headline for a
hand-built solver against HiGHS. The portfolio is competitive on the
benchmark problems this repo tracks, wins decisively on several large or
degenerate instances, and reports its failures truthfully.
