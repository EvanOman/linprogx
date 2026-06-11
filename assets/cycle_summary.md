| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | -5.226393 | 5.326e-08 | 0.106s | C CSR matrix with native-c-sparse-ipm; equality+bounds; native sparse IPM converged; max equality residual 2.717e-11; presolve removed 388 rows and 360 cols |
| SciPy/HiGHS | optimal | -5.226393 | 5.898e-12 | 0.187s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | -5.226393 | 8.174e-10 | 0.211s | Clarabel status: Solved; max equality residual 7.276e-12 |
