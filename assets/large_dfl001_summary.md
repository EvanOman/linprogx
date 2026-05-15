| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266396.413545 | 3.665e-01 | 34.359s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG, objective_scale=5e4; native sparse PDHG converged; max equality residual 1.911e-05 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.419s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | reported_dual_infeasible | n/a | n/a | 0.340s | Clarabel status: DualInfeasible |
