| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266394.072934 | 1.974e+00 | 6.408s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG converged; max equality residual 1.914e-05; objective scale 4.96e+05 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.261s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 8.294s | Clarabel status: Solved; objective_scale=100; max equality residual 1.074e-11 |
