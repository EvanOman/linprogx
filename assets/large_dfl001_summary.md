| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266396.413545 | 3.665e-01 | 33.619s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG, objective_scale=5e4; native sparse PDHG converged; max equality residual 1.911e-05 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.198s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 7.366s | Clarabel status: Solved; objective_scale=100; max equality residual 1.074e-11 |
