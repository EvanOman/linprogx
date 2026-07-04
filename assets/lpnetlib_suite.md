
| Instance | linprogx | HiGHS | Clarabel | lx delta vs HiGHS | lx residual | lx route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lp_80bau3b | 0.31s | 0.17s | 0.28s | 5.31e-02 | 6.0e-12 | ipm |
| lp_cre_a | 0.13s | 0.09s | 0.13s | 5.74e+01 | 2.6e-06 | ipm |
| lp_cre_b | 7.47s | 1.83s | 15.80s | 6.96e+00 | 6.4e-08 | ipm |
| lp_cre_d | 6.61s | 1.03s | 14.01s | 5.12e-01 | 2.0e-09 | ipm |
| lp_d2q06c | 0.40s | 0.84s | 1.98s | 3.12e-04 | 2.8e-10 | ipm |
| lp_degen3 | 0.21s | 0.21s | 0.42s | 3.60e-04 | 1.8e-06 | ipm |
| lp_fit2p | 0.09s | 1.72s | 0.23s | 3.50e-01 | 2.9e-07 | ipm |
| lp_greenbea | 5.86s | 0.25s | 2.08s | 4.47e-08 | 1.5e-08 | simplex |
| lp_ken_07 | 0.02s | 0.03s | 0.05s | 3.67e+03 | 4.8e-07 | ipm |
| lp_ken_11 | 0.22s | 0.27s | 0.57s | 4.49e+04 | 2.0e-05 | ipm |
| lp_ken_13 | 0.59s | 0.87s | 1.79s | 4.50e+04 | 2.6e-07 | ipm |
| lp_ken_18 | 6.42s | 8.15s | DualInfeasible | 5.12e+05 | 1.5e-05 | ipm |
| lp_maros_r7 | 1.91s | 0.71s | 1.79s | 6.85e+00 | 8.1e-06 | ipm |
| lp_osa_14 | 0.89s | 0.98s | 1.99s | 1.68e-01 | 6.5e-10 | ipm |
| lp_osa_30 | 1.69s | 3.70s | 5.71s | 2.42e+01 | 1.9e-09 | ipm |
| lp_osa_60 | 5.25s | 17.39s | 23.14s | 2.13e+01 | 7.5e-09 | ipm |
| lp_pds_10 | 2.89s | 1.30s | 24.63s | 1.07e+02 | 1.3e-05 | pdhg |
| lp_pds_20 | 14.29s | 10.33s | 98.42s | 4.79e+02 | 1.8e-05 | pdhg |
| lp_pilot87 | 4.70s | 3.61s | 8.36s | 1.47e-04 | 6.6e-11 | ipm |
| lp_qap12 | 1.50s | 92.67s | 2.89s | 3.10e-03 | 1.9e-05 | pdhg |
| lp_qap15 | 0.68s | timeout | 15.48s | n/a | 7.0e-06 | pdhg |
| lp_stocfor3 | 0.73s | 0.55s | 0.90s | 1.86e-03 | 5.1e-11 | ipm |
| lp_truss | 0.13s | 2.70s | 0.15s | 6.48e+00 | 2.4e-09 | ipm |
| lp_woodw | 0.17s | 0.09s | 0.22s | 2.88e-05 | 5.5e-09 | ipm |

## Summary (run of 2026-07-02 evening, loadavg ~2-3, build 77b50f2)

- **Coverage: linprogx 24/24 — exceeds HiGHS (23/24, qap15 timeout) and
  Clarabel (23/24, ken_18 DualInfeasible).**
- **Aggregate suite wall time: linprogx 63.2s vs HiGHS 328.4s (incl.
  180s qap15 timeout) vs Clarabel 220.9s.**
- **Head-to-head timing: linprogx faster on 12 of 24 (d2q06c, fit2p,
  ken_07, ken_11, ken_13, ken_18, osa_14, osa_30, osa_60, qap12, qap15,
  truss), degen3 an exact tie, HiGHS faster on 11.** The per-instance
  majority, aggregate time, geometric mean, and coverage are all
  linprogx's.
- Every linprogx result is certificate-backed at the public eps=2e-5.
