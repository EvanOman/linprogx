| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx | skipped | n/a | n/a | n/a | Dense Python tableau skipped; raw A alone would materialize 74,248,330 coefficients before slacks/artificials. |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.266s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | reported_dual_infeasible | n/a | n/a | 0.366s | Clarabel status: DualInfeasible |
