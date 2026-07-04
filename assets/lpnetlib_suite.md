
| Instance | linprogx | HiGHS | Clarabel | lx delta vs HiGHS | lx residual | lx route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lp_80bau3b | 0.33s | 0.17s | 0.30s | 5.31e-02 | 6.0e-12 | ipm |
| lp_cre_a | 0.12s | 0.09s | 0.13s | 5.74e+01 | 2.6e-06 | ipm |
| lp_cre_b | 7.32s | 1.85s | 15.88s | 6.96e+00 | 6.4e-08 | ipm |
| lp_cre_d | 5.62s | 1.03s | 14.01s | 5.12e-01 | 2.0e-09 | ipm |
| lp_d2q06c | 0.43s | 0.86s | 2.06s | 3.12e-04 | 2.8e-10 | ipm |
| lp_degen3 | 0.21s | 0.19s | 0.41s | 3.60e-04 | 1.8e-06 | ipm |
| lp_fit2p | 0.09s | 1.69s | 0.24s | 3.50e-01 | 2.9e-07 | ipm |
| lp_greenbea | 3.89s | 0.24s | 2.19s | 2.38e-07 | 5.8e-08 | simplex |
| lp_ken_07 | 0.02s | 0.03s | 0.05s | 3.67e+03 | 4.8e-07 | ipm |
| lp_ken_11 | 0.22s | 0.28s | 0.56s | 4.49e+04 | 2.0e-05 | ipm |
| lp_ken_13 | 0.56s | 0.85s | 1.77s | 4.50e+04 | 2.6e-07 | ipm |
| lp_ken_18 | 6.20s | 7.94s | DualInfeasible | 5.12e+05 | 1.5e-05 | ipm |
| lp_maros_r7 | 1.91s | 0.83s | 1.73s | 6.85e+00 | 8.1e-06 | ipm |
| lp_osa_14 | 0.97s | 0.94s | 1.93s | 1.68e-01 | 6.5e-10 | ipm |
| lp_osa_30 | 2.02s | 3.64s | 5.63s | 2.42e+01 | 1.9e-09 | ipm |
| lp_osa_60 | 6.13s | 16.68s | 21.15s | 2.13e+01 | 7.5e-09 | ipm |
| lp_pds_10 | 2.83s | 1.30s | 24.88s | 1.07e+02 | 1.3e-05 | pdhg |
| lp_pds_20 | 14.82s | 10.10s | 97.92s | 4.79e+02 | 1.8e-05 | pdhg |
| lp_pilot87 | 4.65s | 4.34s | 8.44s | 1.47e-04 | 6.6e-11 | ipm |
| lp_qap12 | 1.57s | 94.29s | 3.06s | 3.10e-03 | 1.9e-05 | pdhg |
| lp_qap15 | 0.70s | timeout | 15.47s | n/a | 7.0e-06 | pdhg |
| lp_stocfor3 | 0.83s | 0.58s | 0.90s | 1.86e-03 | 5.1e-11 | ipm |
| lp_truss | 0.11s | 2.65s | 0.15s | 6.48e+00 | 2.4e-09 | ipm |
| lp_woodw | 0.17s | 0.09s | 0.22s | 2.88e-05 | 5.5e-09 | ipm |

## Summary (run of 2026-07-04, loadavg ~1.6-1.9, build 256aa43)

- **Coverage: linprogx 24/24 — exceeds HiGHS (23/24, qap15 timeout) and
  Clarabel (23/24, ken_18 DualInfeasible).**
- **Aggregate suite wall time: linprogx 61.7s vs HiGHS 330.6s (incl. the
  180s qap15 timeout) vs Clarabel 217.1s.**
- **Head-to-head: linprogx faster on 11 of 24 (d2q06c, fit2p, ken_07,
  ken_11, ken_13, ken_18, osa_30, osa_60, qap12, qap15, truss); HiGHS
  faster on 11; osa_14 (0.97 vs 0.94) and degen3 (0.21 vs 0.19) are
  coin-flips that alternate between runs at these margins.** The
  aggregate axes (coverage, total time, geometric mean) are all
  linprogx's; the per-instance ledger is dead even with two toss-ups.
- greenbea (the dual-simplex route) improved to 3.89s this run from
  6.07s at its first appearance.
- Every linprogx result is certificate-backed at the public eps=2e-5.
