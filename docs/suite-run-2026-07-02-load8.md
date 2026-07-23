| Instance | linprogx | HiGHS | Clarabel | lx delta vs HiGHS | lx residual | lx route |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lp_80bau3b | 0.38s | 0.18s | 0.29s | 5.31e-02 | 6.0e-12 | ipm |
| lp_cre_a | 0.14s | 0.09s | 0.13s | 5.74e+01 | 2.6e-06 | ipm |
| lp_cre_b | 7.53s | 1.85s | 16.17s | 6.96e+00 | 6.4e-08 | ipm |
| lp_cre_d | 7.17s | 1.02s | 14.32s | 5.12e-01 | 2.0e-09 | ipm |
| lp_d2q06c | 0.50s | 0.87s | 2.11s | 3.12e-04 | 2.8e-10 | ipm |
| lp_degen3 | 0.24s | 0.20s | 0.42s | 3.60e-04 | 1.8e-06 | ipm |
| lp_fit2p | 0.11s | 1.81s | 0.24s | 3.50e-01 | 2.9e-07 | ipm |
| lp_greenbea | 18.57s | 0.25s | 2.21s | 1.19e-07 | 6.1e-08 | simplex |
| lp_ken_07 | 0.03s | 0.04s | 0.05s | 3.67e+03 | 4.8e-07 | ipm |
| lp_ken_11 | 0.33s | 0.28s | 0.65s | 4.49e+04 | 2.0e-05 | ipm |
| lp_ken_13 | 0.84s | 0.86s | 1.85s | 4.50e+04 | 2.6e-07 | ipm |
| lp_ken_18 | 8.13s | 8.57s | DualInfeasible | 5.12e+05 | 1.5e-05 | ipm |
| lp_maros_r7 | 2.22s | 0.80s | 1.83s | 6.85e+00 | 8.1e-06 | ipm |
| lp_osa_14 | 1.64s | 0.98s | 1.98s | 1.68e-01 | 4.6e-10 | ipm |
| lp_osa_30 | 3.96s | 4.02s | 6.00s | 2.42e+01 | 1.8e-09 | ipm |
| lp_osa_60 | 15.90s | 18.93s | 22.66s | 2.13e+01 | 6.2e-09 | ipm |
| lp_pds_10 | 3.92s | 1.29s | 25.64s | 1.07e+02 | 1.3e-05 | pdhg |
| lp_pds_20 | 19.74s | 10.22s | 101.94s | 4.79e+02 | 1.8e-05 | pdhg |
| lp_pilot87 | 5.65s | 3.56s | 8.54s | 1.47e-04 | 6.6e-11 | ipm |
| lp_qap12 | 1.99s | 99.32s | 3.07s | 3.10e-03 | 1.9e-05 | pdhg |
| lp_qap15 | 0.97s | timeout | 15.21s | n/a | 7.0e-06 | pdhg |
| lp_stocfor3 | 1.01s | 0.56s | 0.93s | 1.86e-03 | 5.1e-11 | ipm |
| lp_truss | 0.14s | 2.71s | 0.15s | 6.48e+00 | 2.4e-09 | ipm |
| lp_woodw | 0.20s | 0.09s | 0.28s | 2.88e-05 | 5.5e-09 | ipm |

NOTE: paired 3-solver run under loadavg ~7-8 (NOT the quiet-box official
run — treat ratios as reliable, absolute times as ~1.3x inflated).
Build: perf-supernodal-simplex 6315509. Coverage: linprogx 24/24,
HiGHS 23/24 (qap15 timeout), Clarabel 23/24 (ken_18 DualInfeasible).
linprogx head-to-head timing wins vs HiGHS: d2q06c, fit2p, ken_07,
ken_13, ken_18, osa_30, osa_60, qap12, truss (9), plus qap15 solved
where HiGHS times out. Aggregate suite wall time: linprogx ~101s vs
HiGHS ~338s (incl. 180s timeout).
