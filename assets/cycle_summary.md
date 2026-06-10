| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | -5.226396 | 2.835e-06 | 1.389s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG converged; max equality residual 4.404e-06; objective scale 1; presolve removed 388 rows and 360 cols |
| SciPy/HiGHS | optimal | -5.226393 | 5.898e-12 | 0.180s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | -5.226393 | 8.174e-10 | 0.218s | Clarabel status: Solved; max equality residual 7.276e-12 |
