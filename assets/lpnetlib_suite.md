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
| lp_80bau3b | iteration_limit | 0.23s | 0.47s | n/a | n/a |
| lp_cre_a | 0.33s (ipm) | 0.11s | 0.15s | 3.9e-07 | 5e-05 |
| lp_cre_b | 152.13s (ipm) | 2.23s | 22.61s | 4.4e-07 | 2e-08 |
| lp_cre_d | 33.84s (ipm) | 1.19s | 19.43s | 1.9e-08 | 5e-07 |
| lp_d2q06c | 1.76s (ipm) | 1.02s | 2.65s | 1.4e-10 | 7e-10 |
| lp_degen3 | 2.61s (ipm) | 0.23s | 0.55s | 5.5e-07 | 1e-07 |
| lp_fit2p | iteration_limit | 1.58s | 0.31s | n/a | n/a |
| lp_greenbea | iteration_limit | 0.30s | 2.97s | n/a | n/a |
| lp_ken_07 | 0.08s (ipm) | 0.04s | 0.07s | 1.6e-09 | 3e-05 |
| lp_ken_11 | 0.99s (ipm) | 0.39s | 0.98s | 7.5e-08 | 7e-04 |
| lp_ken_13 | 4.78s (ipm) | 1.31s | 3.40s | 3.0e-08 | 7e-06 |
| lp_ken_18 | 147.80s (ipm) | 14.52s | DualInfeasible | 1.5e-08 | 6e-05 |
| lp_maros_r7 | 12.65s (ipm) | 0.83s | 2.83s | 3.2e-10 | 3e-11 |
| lp_osa_14 | 5.35s (ipm) | 1.33s | 4.47s | 1.7e-07 | 1e-04 |
| lp_osa_30 | 4.66s (ipm) | 7.08s | 10.91s | 6.8e-08 | 1e-04 |
| lp_osa_60 | timeout | 34.52s | 41.34s | n/a | n/a |
| lp_pds_10 | 3.62s (pdhg) | 1.87s | 34.99s | 4.0e-09 | 1e-05 |
| lp_pds_20 | 56.14s (pdhg) | 14.84s | 160.81s | 2.0e-08 | 2e-05 |
| lp_pilot87 | 32.22s (ipm) | 4.38s | 14.68s | 1.4e-07 | 6e-11 |
| lp_qap12 | 0.44s (pdhg) | 108.06s | 5.61s | 9.4e-06 | 2e-05 |
| lp_qap15 | 1.26s (pdhg) | timeout | 24.82s | 5.9e-06 | 7e-06 |
| lp_stocfor3 | 1.00s (ipm) | 0.83s | 1.66s | 4.7e-08 | 7e-11 |
| lp_truss | 0.22s (ipm) | 3.03s | 0.22s | 3.8e-12 | 2e-12 |
| lp_woodw | 0.21s (ipm) | 0.12s | 0.24s | 2.2e-05 | 6e-09 |

## Summary

- **Solved (status optimal): linprogx 20/24, HiGHS 23/24, Clarabel 23/24.**
- Where linprogx solves, quality is high: relative objective error between
  3.8e-12 and 2.2e-5.
- Highlights: qap12 in 0.44s and qap15 in 1.3s where HiGHS needs 108s and
  times out respectively; osa_30 via the IPM in 4.7s vs HiGHS 7.1s; truss
  14x faster than HiGHS; pds_20 2.9x faster than Clarabel; ken_18 solves
  (148s, residual 6e-5 relative-accepted) where Clarabel reports
  DualInfeasible.
- The portfolio routes by measured factorization cost: an approximate-
  degree ordering (10-60x faster than exact minimum degree at equal or
  better fill) plus an ordering work budget and a throughput-calibrated
  factor-flops cap decide between the interior point method and PDHG, with
  automatic fallback. Newton solves use one step of iterative refinement.
- The 4 unsolved instances: fit2p (dense columns make the normal equations
  fully dense; needs dense-column splitting), greenbea and 80bau3b (IPM
  trajectory stalls; the explicit gap test refuses to certify near-optimal
  points -- on greenbea Clarabel certifies a point with a 1.3e-3 objective
  error), and osa_60 (1.4M nonzeros, exceeds the 180s budget on the
  first-order path at residual 4.9e-5).

Mature solvers still win on breadth -- that is the honest headline for a
hand-built solver against HiGHS. The portfolio is competitive on the
benchmark problems this repo tracks, wins decisively on several large or
degenerate instances, and reports its failures truthfully.
