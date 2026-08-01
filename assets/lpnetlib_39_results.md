# LPnetlib 39-case benchmark

Times are medians of one solve on each of three independent Modal hosts. The paired column is the stricter three-host, seven-pairs-per-host result for the 15 newly added cases.

| Instance | linprogx | HiGHS | Clarabel | vs HiGHS | vs Clarabel | Route | Paired lx/HiGHS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| lp_25fv47 | 0.837s | 0.235s | 0.148s | 0.28x | 0.18x | simplex | 3.531x |
| lp_80bau3b | 0.206s | 0.278s | 0.483s | 1.35x | 2.34x | ipm | — |
| lp_agg2 | 0.010s | 0.027s | 0.064s | 2.65x | 6.19x | simplex | 0.364x |
| lp_agg3 | 0.011s | 0.028s | 0.059s | 2.60x | 5.56x | simplex | 0.372x |
| lp_bnl2 | 0.281s | 0.118s | 1.756s | 0.42x | 6.24x | ipm | 2.677x |
| lp_cre_a | 0.151s | 0.152s | 0.224s | 1.01x | 1.49x | ipm | — |
| lp_cre_b | 1.660s | 2.562s | 23.664s | 1.54x | 14.25x | ipm | — |
| lp_cre_d | 1.428s | 1.489s | 21.065s | 1.04x | 14.75x | ipm | — |
| lp_cycle | 0.111s | 0.262s | 0.316s | 2.36x | 2.86x | simplex | 0.423x |
| lp_d2q06c | 0.428s | 1.171s | 5.430s | 2.73x | 12.68x | ipm | — |
| lp_degen2 | 0.110s | 0.029s | 0.025s | 0.27x | 0.23x | simplex | 3.620x |
| lp_degen3 | 0.261s | 0.272s | 1.067s | 1.04x | 4.09x | ipm | — |
| lp_fffff800 | 0.015s | 0.026s | 0.103s | 1.71x | 6.81x | simplex | 0.574x |
| lp_fit1p | 0.016s | 0.062s | 0.044s | 3.86x | 2.72x | ipm | 0.257x |
| lp_fit2p | 0.114s | 1.343s | 0.408s | 11.76x | 3.57x | ipm | — |
| lp_ganges | 0.035s | 0.032s | 0.088s | 0.93x | 2.54x | ipm | 1.071x |
| lp_greenbea | 0.373s | 0.375s | 4.332s | 1.00x | 11.60x | simplex | — |
| lp_greenbeb | 0.879s | 0.637s | 6.521s | 0.73x | 7.42x | simplex | 1.376x |
| lp_israel | 0.007s | 0.014s | 0.012s | 1.97x | 1.63x | simplex | 0.512x |
| lp_ken_07 | 0.030s | 0.072s | 0.095s | 2.38x | 3.11x | ipm | — |
| lp_ken_11 | 0.283s | 0.446s | 0.905s | 1.57x | 3.20x | ipm | — |
| lp_ken_13 | 0.667s | 1.305s | 2.299s | 1.96x | 3.45x | ipm | — |
| lp_ken_18 | 5.782s | 12.893s | incomplete | 2.23x | n/a | ipm | — |
| lp_maros_r7 | 0.727s | 1.010s | 3.353s | 1.39x | 4.61x | ipm | — |
| lp_osa_14 | 1.049s | 1.319s | 2.846s | 1.26x | 2.71x | ipm | — |
| lp_osa_30 | 1.807s | 4.737s | 6.948s | 2.62x | 3.85x | ipm | — |
| lp_osa_60 | 5.678s | 21.364s | 29.659s | 3.76x | 5.22x | ipm | — |
| lp_pds_10 | 1.670s | 1.819s | 34.874s | 1.09x | 20.88x | pdhg | — |
| lp_pds_20 | 6.362s | 13.254s | 126.421s | 2.08x | 19.87x | pdhg | — |
| lp_pilot | 0.956s | 1.439s | 10.908s | 1.51x | 11.42x | ipm | 0.721x |
| lp_pilot87 | 3.888s | 4.718s | 20.849s | 1.21x | 5.36x | ipm | — |
| lp_qap12 | 2.126s | 120.774s | 3.745s | 56.80x | 1.76x | pdhg | — |
| lp_qap15 | 0.780s | incomplete | 17.233s | n/a | 22.09x | pdhg | — |
| lp_sierra | 0.114s | 0.043s | 0.070s | 0.38x | 0.62x | simplex | 2.737x |
| lp_stocfor2 | 0.067s | 0.089s | 0.100s | 1.32x | 1.48x | ipm | 0.759x |
| lp_stocfor3 | 0.783s | 0.830s | 1.444s | 1.06x | 1.85x | ipm | — |
| lp_truss | 0.158s | 3.574s | 0.254s | 22.57x | 1.60x | ipm | — |
| lp_tuff | 0.007s | 0.018s | 0.028s | 2.62x | 3.94x | simplex | 0.372x |
| lp_woodw | 0.132s | 0.144s | 0.346s | 1.09x | 2.62x | ipm | — |

Ratios in the `vs` columns are competitor time divided by linprogx time, so values above 1.0 favor linprogx. `†` marks a solver with incomplete three-host coverage; its displayed time is the median of successful hosts only.
