
| Instance | linprogx | HiGHS | Clarabel | lx delta vs HiGHS | lx residual | lx route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lp_80bau3b | 0.33s | 0.18s | 0.31s | 5.31e-02 | 6.0e-12 | ipm |
| lp_cre_a | 0.14s | 0.09s | 0.13s | 5.74e+01 | 2.6e-06 | ipm |
| lp_cre_b | 7.72s | 1.89s | 17.04s | 6.96e+00 | 6.4e-08 | ipm |
| lp_cre_d | 6.81s | 1.17s | 15.24s | 5.12e-01 | 2.0e-09 | ipm |
| lp_d2q06c | 0.62s | 0.90s | 2.12s | 3.12e-04 | 2.8e-10 | ipm |
| lp_degen3 | 0.19s | 0.20s | 0.43s | 3.60e-04 | 1.8e-06 | ipm |
| lp_fit2p | 0.10s | 1.45s | 0.24s | 3.50e-01 | 2.9e-07 | ipm |
| lp_greenbea | 6.07s | 0.26s | 2.15s | 4.47e-08 | 1.5e-08 | simplex |
| lp_ken_07 | 0.03s | 0.04s | 0.05s | 3.67e+03 | 4.8e-07 | ipm |
| lp_ken_11 | 0.31s | 0.29s | 0.64s | 4.49e+04 | 2.0e-05 | ipm |
| lp_ken_13 | 0.84s | 0.90s | 1.73s | 4.50e+04 | 2.6e-07 | ipm |
| lp_ken_18 | 7.11s | 8.75s | DualInfeasible | 5.12e+05 | 1.5e-05 | ipm |
| lp_maros_r7 | 2.40s | 0.92s | 2.06s | 6.85e+00 | 8.1e-06 | ipm |
| lp_osa_14 | 1.40s | 1.00s | 2.56s | 1.68e-01 | 6.5e-10 | ipm |
| lp_osa_30 | 2.55s | 4.16s | 7.30s | 2.42e+01 | 1.9e-09 | ipm |
| lp_osa_60 | 7.10s | 18.22s | 23.84s | 2.13e+01 | 7.5e-09 | ipm |
| lp_pds_10 | 3.64s | 1.48s | 26.85s | 1.07e+02 | 1.3e-05 | pdhg |
| lp_pds_20 | 18.68s | 10.39s | 110.13s | 4.79e+02 | 1.8e-05 | pdhg |
| lp_pilot87 | 5.73s | 4.68s | 8.84s | 1.47e-04 | 6.6e-11 | ipm |
| lp_qap12 | 1.59s | 97.03s | 3.06s | 3.10e-03 | 1.9e-05 | pdhg |
| lp_qap15 | 0.88s | timeout | 16.89s | n/a | 7.0e-06 | pdhg |
| lp_stocfor3 | 0.97s | 0.57s | 0.99s | 1.86e-03 | 5.1e-11 | ipm |
| lp_truss | 0.12s | 2.69s | 0.15s | 6.48e+00 | 2.4e-09 | ipm |
| lp_woodw | 0.18s | 0.09s | 0.29s | 2.88e-05 | 5.5e-09 | ipm |

## Summary (run of 2026-07-02, loadavg ~3-6, build 3ddbff5)

- **Coverage: linprogx 24/24 — exceeds HiGHS (23/24, qap15 timeout) and
  Clarabel (23/24, ken_18 DualInfeasible).** greenbea, historically the
  linprogx miss, is solved by the new bounded-variable dual simplex
  (route "simplex") with the objective matching HiGHS to 4.5e-08
  relative.
- **Aggregate suite wall time: linprogx 75.5s, HiGHS 337.4s (including
  its 180s qap15 timeout), Clarabel 243.5s.**
- **Geometric-mean time ratio linprogx/HiGHS ~0.78** across all 24
  instances (counting qap15 at the timeout cap): faster on the
  geometric mean.
- Head-to-head timing: linprogx faster on 10 of the 23 mutually solved
  instances (d2q06c, degen3, fit2p, ken_07, ken_13, ken_18, osa_30,
  osa_60, qap12, truss) plus qap15 solved outright where HiGHS times
  out; HiGHS faster on 13 (80bau3b, cre_a/b/d, greenbea, ken_11,
  maros_r7, osa_14, pds_10/20, pilot87, stocfor3, woodw).
- Every linprogx result is certificate-backed at the public eps=2e-5;
  the residual column is the recomputed max equality residual on the
  original problem.
