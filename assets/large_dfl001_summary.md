| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266396.071554 | 2.455e-02 | 24.042s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG converged; max equality residual 1.577e-05; objective scale 1.5e+04 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.453s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 8.363s | Clarabel status: Solved; objective_scale=100; max equality residual 1.074e-11 |
