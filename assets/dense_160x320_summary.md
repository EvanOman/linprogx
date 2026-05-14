| Solver | Status | Objective | Delta vs linprogx/expected | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx | optimal | 237.053663 | 2.274e-13 | 0.596s | 168 simplex iterations |
| SciPy/HiGHS | optimal | 237.053663 | 2.842e-13 | 0.030s | Open-source sparse/dense LP baseline |
| Clarabel | optimal | 237.053663 | 1.951e-10 | 0.101s | Open-source conic interior-point baseline |
