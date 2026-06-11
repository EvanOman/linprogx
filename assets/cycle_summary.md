| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | -5.226393 | 1.682e-07 | 0.161s | C CSR matrix with native-c-sparse-ipm; equality+bounds; native sparse IPM converged; max equality residual 1.091e-11; presolve removed 388 rows and 360 cols |
| SciPy/HiGHS | optimal | -5.226393 | 5.898e-12 | 0.243s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | -5.226393 | 8.174e-10 | 0.303s | Clarabel status: Solved; max equality residual 7.276e-12 |
