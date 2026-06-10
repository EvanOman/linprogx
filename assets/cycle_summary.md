| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | -5.225511 | 8.817e-04 | 2.035s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG converged; max equality residual 1.797e-05; objective scale 1 |
| SciPy/HiGHS | optimal | -5.226393 | 5.898e-12 | 0.184s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | -5.226393 | 8.174e-10 | 0.244s | Clarabel status: Solved; max equality residual 7.276e-12 |
