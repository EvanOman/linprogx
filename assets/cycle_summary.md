| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | iteration_limit | -4.574252 | 6.521e-01 | 2.205s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG hit the iteration limit; max equality residual 5.484e+00; objective scale 0.06 |
| SciPy/HiGHS | optimal | -5.226393 | 5.898e-12 | 0.185s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | -5.226393 | 8.174e-10 | 0.345s | Clarabel status: Solved; max equality residual 7.276e-12 |
