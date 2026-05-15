| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266396.050738 | 3.738e-03 | 53.701s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG; native sparse PDHG converged; max equality residual 1.131e-05; objective scale 5e+03 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.753s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 10.180s | Clarabel status: Solved; objective_scale=100; max equality residual 1.074e-11 |
