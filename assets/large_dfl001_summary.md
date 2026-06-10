| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266396.207350 | 1.604e-01 | 5.336s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG converged; max equality residual 1.961e-05; objective scale 4.96e+05; presolve removed 15 rows and 15 cols |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.378s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 8.057s | Clarabel status: Solved; objective_scale=100; max equality residual 1.074e-11 |
