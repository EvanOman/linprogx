| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266398.367904 | 2.321e+00 | 9.467s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG converged; max equality residual 1.867e-05; objective scale 1.5e+04 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 5.939s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 6.879s | Clarabel status: Solved; objective_scale=100; max equality residual 1.074e-11 |
