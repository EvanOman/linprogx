
| Instance | linprogx | HiGHS | Clarabel | lx delta vs HiGHS | lx residual | lx route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lp_80bau3b | 0.29s | 0.17s | 0.31s | 8.50e-02 | 8.5e-09 | ipm |
| lp_cre_a | 0.11s | 0.08s | 0.12s | 5.75e+01 | 3.9e-06 | ipm |
| lp_cre_b | 5.80s | 1.87s | 16.34s | 6.96e+00 | 6.4e-08 | ipm |
| lp_cre_d | 4.77s | 1.01s | 14.40s | 5.12e-01 | 2.0e-09 | ipm |
| lp_d2q06c | 0.33s | 0.85s | 2.03s | 3.07e-04 | 6.2e-10 | ipm |
| lp_degen3 | 0.18s | 0.20s | 0.41s | 3.57e-04 | 2.0e-06 | ipm |
| lp_fit2p | 0.08s | 1.46s | 0.24s | 3.50e-01 | 2.9e-07 | ipm |
| lp_greenbea | 3.03s | 0.24s | 2.13s | 2.16e-04 | 3.1e-07 | simplex |
| lp_ken_07 | 0.02s | 0.04s | 0.05s | 3.67e+03 | 4.8e-07 | ipm |
| lp_ken_11 | 0.20s | 0.30s | 0.58s | 4.49e+04 | 2.0e-05 | ipm |
| lp_ken_13 | 0.57s | 0.85s | 1.73s | 4.50e+04 | 2.6e-07 | ipm |
| lp_ken_18 | 7.23s | 8.08s | DualInfeasible | 5.12e+05 | 1.5e-05 | ipm |
| lp_maros_r7 | 2.16s | 0.91s | 1.92s | 6.85e+00 | 8.1e-06 | ipm |
| lp_osa_14 | 1.05s | 1.00s | 2.02s | 1.68e-01 | 6.5e-10 | ipm |
| lp_osa_30 | 1.66s | 3.71s | 5.43s | 2.42e+01 | 2.1e-09 | ipm |
| lp_osa_60 | 6.21s | 17.60s | 24.07s | 2.13e+01 | 5.1e-09 | ipm |
| lp_pds_10 | 2.90s | 1.31s | 25.70s | 1.07e+02 | 1.3e-05 | pdhg |
| lp_pds_20 | 15.39s | 10.30s | 102.57s | 4.79e+02 | 1.8e-05 | pdhg |
| lp_pilot87 | 3.98s | 3.93s | 8.42s | 1.47e-04 | 6.5e-11 | ipm |
| lp_qap12 | 1.59s | 101.54s | 3.08s | 3.10e-03 | 1.9e-05 | pdhg |
| lp_qap15 | 0.68s | timeout | 15.88s | n/a | 7.0e-06 | pdhg |
| lp_stocfor3 | 0.66s | 0.54s | 0.99s | 1.86e-03 | 5.2e-11 | ipm |
| lp_truss | 0.11s | 2.65s | 0.15s | 6.48e+00 | 2.4e-09 | ipm |
| lp_woodw | 0.16s | 0.09s | 0.22s | 2.88e-05 | 5.5e-09 | ipm |

## Summary (run of 2026-07-04 late, build 3a1932c: BFRT + dtrsv tail solves)

- **Coverage: linprogx 24/24 — exceeds HiGHS (23/24, qap15 timeout) and
  Clarabel (23/24, ken_18 DualInfeasible).**
- **Aggregate suite wall: linprogx 59.2s vs HiGHS 337.7s (incl. 180s
  qap15 timeout) vs Clarabel 229.6s.**
- **Head-to-head: 12 linprogx wins on this run; with the 5-run paired
  protocol osa_14 (5/5) and degen3 (4/5) are established linprogx wins
  -> 13-11. pilot87 sits at parity (3.98 vs 3.93 here; paired ratios
  0.93-1.05) pending a quiet-box certification.**
- greenbea (dual-simplex route with the new longest-step BFRT ratio
  test): 3.03s, from 21s at the route's birth.
- Every result certificate-backed at the public eps=2e-5.
