# linprogx

[![CI](https://github.com/EvanOman/linprogx/actions/workflows/ci.yml/badge.svg)](https://github.com/EvanOman/linprogx/actions/workflows/ci.yml)
![coverage](assets/coverage.svg)

`linprogx` is a small linear programming solver written from scratch in Python, with an optional C extension for the tableau operations that run in the tight loop.

It solves continuous linear programs with maximization or minimization objectives, `<=`, `>=`, and equality constraints, finite or infinite variable bounds, and free variables. The core algorithm is a two-phase primal simplex method: phase I builds a feasible basis with artificial variables, and phase II optimizes the requested objective.

Reference link saved with the project: https://www.linkedin.com/posts/antonvorobets_if-you-think-ai-can-write-advanced-analytics-share-7460672846042710016-z_ya?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAk_CVIBr27sXDGNG8kKqOPnWZAdJrVOA7Q

## TL;DR

This repo is a compact LP solver built as a benchmarkable artifact: a from-scratch two-phase simplex implementation, a small C accelerator, Python and CLI interfaces, 16 hand-authored LP examples, 8 standardized Klee-Minty stress cases, and test-time correctness checks against SciPy/HiGHS and Clarabel.

The point is not to beat mature solvers on real sparse production models. It is to make the mechanics visible, keep the dependency-free runtime small, and show reproducible comparisons against serious open-source baselines.

## What It Does

- Solves dense small-to-medium LPs without NumPy or SciPy.
- Returns primal values, objective value, slacks, basis names, reduced costs, and shadow-price estimates.
- Provides both a direct matrix API and a small modeling interface.
- Ships a JSON CLI for quick experiments.
- Compiles `linprogx._cfast` for in-place tableau pivots and dot products, with a pure-Python fallback.

This is intended as an inspectable educational and experimental solver, not a replacement for HiGHS, CLP, Gurobi, CPLEX, or Mosek on large production optimization models.

## Install

```bash
git clone git@github.com:EvanOman/linprogx.git
cd linprogx
uv sync --extra dev
```

## Python API

```python
from linprogx import solve, solve_canonical

result = solve(
    c=[3, 2],
    A=[[1, 1], [1, 0], [0, 1]],
    b=[4, 2, 3],
    objective="max",
)

print(result.status)
print(result.objective_value)
print(result.x)
```

Canonical minimization form is available directly:

```python
# min c^T x
# subject to Ax = b and Gx <= h
result = solve_canonical(
    c=[1, 2],
    A=[[1, 1]],
    b=[3],
    G=[[-1, 0], [0, -1], [1, 0]],
    h=[0, 0, 2],
)

print(result.status)
print(result.objective_value)
print(result.x)
```

`solve_canonical()` treats variables as free by default because the statement `min c^T x` subject to `Ax=b`, `Gx<=h` does not include `x >= 0`. Encode nonnegativity as rows in `G` or pass explicit `bounds`.

Sparse matrices use a C-backed compressed sparse row type:

```python
from linprogx import SparseLPProblem, csr_matrix, solve_sparse

A_eq = csr_matrix(
    1,
    2,
    indptr=[0, 2],
    indices=[0, 1],
    data=[1.0, 1.0],
)

result = solve_sparse(
    SparseLPProblem(
        c=[1.0, 2.0],
        A_eq=A_eq,
        b_eq=[3.0],
        objective="min",
        bounds=[(0.0, None), (0.0, None)],
    )
)

print(result.solution.status)
print(result.solution.objective_value)
```

The sparse API is dependency-free: it uses `linprogx`'s C CSR representation and a native sparse two-phase simplex implementation. The current sparse pivot strategy is early-stage and intended for small-to-medium sparse LPs; mature sparse solvers remain much stronger on large Netlib-scale models.

## Modeling API

```python
from linprogx import Model

model = Model("factory")
chairs = model.variable("chairs", upper=3)
tables = model.variable("tables")

model.maximize({chairs: 5, tables: 4})
model.add_constraint({chairs: 2, tables: 1}, "<=", 8, name="wood")
model.add_constraint({chairs: 1, tables: 2}, "<=", 8, name="labor")

solution = model.solve()
print(solution.objective_value, solution.x)
```

## CLI

Create `problem.json`:

```json
{
  "c": [3, 2],
  "A": [[1, 1], [1, 0], [0, 1]],
  "b": [4, 2, 3],
  "objective": "max"
}
```

Run:

```bash
uv run linprogx problem.json --pretty
```

## Sample Problems And Solver Comparison

The repository includes a deterministic sample suite in `src/linprogx/samples.py`:

- `product_mix`
- `diet_minimum`
- `transportation_2x3`
- `blending`
- `assignment_relaxation`
- `free_variable_balance`
- `portfolio_allocation`
- `ad_campaign`
- `workforce_cover`
- `knapsack_relaxation`
- `degenerate_multiple_optima`
- `redundant_constraints`
- `negative_rhs_normalization`
- `lower_bound_shift`
- `infeasible_window`
- `unbounded_ray`

It also includes a standardized stress suite based on Klee-Minty cubes in dimensions 3 through 10. Klee-Minty is the classic LP construction used to show exponential worst-case behavior for Dantzig-style simplex pivoting. Netlib's LP collection is the canonical real-world benchmark set, but its compressed MPS problems are mostly large sparse models; this project keeps the in-repo standardized suite dense and small enough for the educational tableau solver.

References:

- Netlib LP collection: https://www.netlib.org/lp/data/
- Klee-Minty cube background: https://en.wikipedia.org/wiki/Klee%E2%80%93Minty_cube

`tests/test_samples_compare.py` solves every sample and standardized stress case with `linprogx`, solves the same model with SciPy/HiGHS and Clarabel, and asserts matching statuses and objective values. Clarabel is an interior-point conic solver, so its objective comparison uses absolute and relative tolerances.

Run the benchmark:

```bash
uv run python bench.py
```

Run only the standardized Klee-Minty suite:

```bash
LINPROGX_BENCH_SUITE=standard uv run python bench.py
```

Regenerate the README plots:

```bash
just plots
```

## Runtime Performance Summary

The benchmark compares `linprogx` against SciPy/HiGHS and Clarabel on 24 dense LPs: 16 hand-authored examples plus 8 Klee-Minty stress cases. The current plots were generated with 50 repeats per solver/problem.

| Metric | Value |
| --- | ---: |
| Cases compared | 24 |
| SciPy/HiGHS max objective delta | 0.00e+00 |
| Clarabel max objective delta | 3.67e-03 |
| Median SciPy/HiGHS runtime ratio vs linprogx | 25.64x |
| Median Clarabel runtime ratio vs linprogx | 1.06x |
| Fastest measured row | lower_bound_shift / scipy-highs |
| Slowest measured row | assignment_relaxation / scipy-highs |

On these tiny dense examples, `linprogx` is usually faster than SciPy/HiGHS because the benchmark is dominated by setup overhead rather than numerical linear algebra. Clarabel is much closer on runtime. Objective agreement is exact against SciPy/HiGHS in this run; Clarabel's small deltas are expected from an interior-point conic solve.

![Runtime on sample LPs](assets/perf_runtime_samples.png)

![Runtime on Klee-Minty LPs](assets/perf_runtime_klee_minty.png)

![Runtime ratio against linprogx](assets/perf_speed_ratios.png)

![Objective deltas against linprogx](assets/perf_objective_delta.png)

## Large Online Benchmark

There are two larger benchmark paths:

- A dense generated LP that `linprogx`, SciPy/HiGHS, and Clarabel all solve.
- Netlib `DFL001`, a much larger sparse online LP that now runs through `linprogx`'s dependency-free C CSR sparse frontend and native sparse PDHG solver.

### Dense 160x320 Benchmark

This is the fair large comparison for the current solver. The benchmark uses a deterministic dense LP with 160 variables, 320 dense inequality rows, 51,200 dense coefficients, variable bounds `0 <= x <= 1`, and a known optimum at the all-ones point.

Run it:

```bash
just dense-bench
```

Current local result:

| Solver | Status | Objective | Delta vs linprogx/expected | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx | optimal | 237.053663 | 2.274e-13 | 0.596s | 168 simplex iterations |
| SciPy/HiGHS | optimal | 237.053663 | 2.842e-13 | 0.030s | Open-source sparse/dense LP baseline |
| Clarabel | optimal | 237.053663 | 1.951e-10 | 0.101s | Open-source conic interior-point baseline |

![Dense 160x320 runtime](assets/dense_160x320_runtime.png)

The result is the expected shape: `linprogx` is correct and usable on a larger dense case, but mature compiled solvers are substantially faster. This is the benchmark to watch as the tableau implementation improves.

### Sparse Netlib DFL001

The repo also includes Netlib `DFL001`, loaded from the SuiteSparse Matrix Collection. It is a real-world airline schedule planning / fleet assignment model with 6,071 equality rows, 12,230 variables, and 35,632 sparse matrix nonzeros.

Source files:

- Data: `benchmark_data/netlib_dfl001/lp_dfl001.mat`
- Metadata: `benchmark_data/netlib_dfl001/README.md`
- Source URL: https://sparse.tamu.edu/mat/LPnetlib/lp_dfl001.mat
- Reference page: https://www.cise.ufl.edu/research/sparse/matrices/LPnetlib/lp_dfl001.html

Run the large benchmark:

```bash
just large-bench
```

This path uses a C-backed compressed sparse row matrix type in `linprogx._csparse` and a dependency-free sparse primal-dual hybrid gradient path for equality-plus-bounds LPs. The older sparse simplex path remains available for small exact sparse LPs; DFL001 uses the C-native PDHG path because the Python sparse tableau pivot loop is not competitive at Netlib scale. SciPy/HiGHS and Clarabel remain in the table as external comparison baselines.

Current local result:

| Solver | Status | Objective | Delta vs published | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| linprogx-sparse | optimal | 11266396.413545 | 3.665e-01 | 33.619s | C CSR matrix with native-c-sparse-pdhg; equality+bounds PDHG, objective_scale=5e4; native sparse PDHG converged; max equality residual 1.911e-05 |
| SciPy/HiGHS | optimal | 11266396.046671 | 3.286e-04 | 6.198s | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| Clarabel | optimal | 11266396.078090 | 3.109e-02 | 7.366s | Clarabel status: Solved; objective_scale=100; max equality residual 1.074e-11 |

On this Netlib-scale sparse case, `linprogx-sparse` now reaches the published objective to about `3.3e-8` relative error with a max equality residual below the configured `2e-5` tolerance. HiGHS and Clarabel both solve the same feasible model more accurately; HiGHS is also about 5x faster than `linprogx-sparse`, which is the right expectation for a mature production LP solver.

![Large Netlib DFL001 runtime](assets/large_dfl001_runtime.png)

Example local run on the included samples:

```text
sample                   solver       status        obj delta  linprogx ms  solver ms
--------------------------------------------------------------------------------------------
product_mix              scipy-highs  optimal        0.00e+00        0.042      1.555
product_mix              clarabel     optimal        4.85e-10        0.041      0.045
transportation_2x3       scipy-highs  optimal        0.00e+00        0.098      1.625
transportation_2x3       clarabel     optimal        1.69e-06        0.101      0.086
knapsack_relaxation      scipy-highs  optimal        0.00e+00        0.059      1.332
knapsack_relaxation      clarabel     optimal        1.04e-08        0.048      0.049
infeasible_window        scipy-highs  infeasible          n/a        0.021      1.240
infeasible_window        clarabel     infeasible          n/a        0.021      0.044
unbounded_ray            scipy-highs  unbounded           n/a        0.004      1.119
unbounded_ray            clarabel     unbounded           n/a        0.004      0.032
```

Standardized Klee-Minty summary, dimensions 3 through 10:

```text
sample                   solver       status        obj delta  linprogx ms  solver ms
--------------------------------------------------------------------------------------------
klee_minty_3d            scipy-highs  optimal        0.00e+00        0.074      1.333
klee_minty_3d            clarabel     optimal        2.23e-09        0.049      0.063
klee_minty_4d            scipy-highs  optimal        0.00e+00        0.076      1.342
klee_minty_4d            clarabel     optimal        2.25e-07        0.073      0.063
klee_minty_5d            scipy-highs  optimal        0.00e+00        0.098      1.338
klee_minty_5d            clarabel     optimal        7.77e-08        0.088      0.078
klee_minty_6d            scipy-highs  optimal        0.00e+00        0.115      1.358
klee_minty_6d            clarabel     optimal        3.86e-06        0.120      0.084
klee_minty_7d            scipy-highs  optimal        0.00e+00        0.148      1.376
klee_minty_7d            clarabel     optimal        7.49e-05        0.152      0.106
klee_minty_8d            scipy-highs  optimal        0.00e+00        0.175      1.368
klee_minty_8d            clarabel     optimal        1.06e-04        0.187      0.117
klee_minty_9d            scipy-highs  optimal        0.00e+00        0.218      1.349
klee_minty_9d            clarabel     optimal        1.18e-03        0.230      0.118
klee_minty_10d           scipy-highs  optimal        0.00e+00        0.273      1.466
klee_minty_10d           clarabel     optimal        3.67e-03        0.288      0.157
```

The default benchmark prints all 24 included cases. These timings are for tiny dense models where Python call overhead dominates. SciPy/HiGHS and Clarabel are the right baselines for larger models.

## Build Provenance

Minimal recreation of the transcript that produced this repository:

```text
User: write a linear programming solver from scratch in python w/ all the bells and whistles. make a nice interface and write c extensions for the performance sensitve parts. include many test cases. make public github repo w/ nice readme. reference this linked in post: <LinkedIn URL>
Assistant: noted that the first Tailscale link required login.
User: supplied the direct LinkedIn URL.
User: don't read post, just write code and save link
Assistant: built and pushed the initial public repo.
User: add many sample problems, compare w/ open source solver for correctness and runtime perf
Assistant: added sample LPs, SciPy/HiGHS and Clarabel comparisons, benchmarks, tests, and README summary.
User: find a standardized set of difficlt LP problems, apply to all solvers and put summary in README
Assistant: added Klee-Minty standardized stress cases, applied all solvers, and documented the results.
User: add a better summary / tldr to README, Add a minimal recreation of our transcript here, along with the total time to create all this.
Assistant: added this TL;DR and provenance section.
User: make sure this is handled, commit and push Prompt AI to find an algorithm for min c^T x subject to Ax=b, Gx<=h
Assistant: added a direct canonical-form API, tests, and documentation for that LP statement.
User: include summary of runtime perf comparison -- make lots of nice looking plots and put those plots in readme
Assistant: added benchmark plot generation, committed PNG assets, and embedded the plots in the README.
User: Add a very, very large problem from an online source... update all of our benchmarks and comparisons with the results of that.
Assistant: added Netlib DFL001 from SuiteSparse, benchmarked SciPy/HiGHS and Clarabel, skipped linprogx by default for dense-tableau memory reasons, and recorded the result.
User: find a large dense problem that can be solved by linprogx too, so sparse support does not hold up the comparison.
Assistant: added a deterministic dense 160x320 benchmark that all three solvers solve and embedded the results in the README.
User: create a PR that adds sparse support, with dependency-free sparse support as the most important requirement.
Assistant: added a C CSR matrix type and native dependency-free sparse simplex path, then updated DFL001 to run the native sparse attempt and compare against external baselines.
```

Recorded creation time: 3 minutes 10 seconds from the initial solver commit (`15192d8`, 2026-05-14 16:37:05 CDT) to the standardized benchmark commit (`96a5a65`, 2026-05-14 16:40:15 CDT). That is measured from git history, so it excludes the uncommitted pre-history before the first commit.

## Development

```bash
just install
just test
just test-cov
just plots
just dense-bench
just large-bench
just fc
```

## Architecture

```text
src/linprogx/
  __init__.py      Public exports
  solver.py        Problem normalization and two-phase simplex
  sparse.py        Dependency-free sparse simplex using the C CSR matrix type
  builder.py       Small modeling interface
  cli.py           JSON command-line interface
  samples.py       Deterministic sample LP library
  compare.py       SciPy/HiGHS and Clarabel correctness/timing comparison
  _fast.py         C-extension dispatch and Python fallback
  _cfast.c         In-place pivot and dot-product C helpers
  _csparse.c       C-backed compressed sparse row matrix type
tests/
  test_solver.py            Solver, bounds, CLI, and validation coverage
  test_samples_compare.py   Sample problem checks against SciPy/HiGHS
  test_sparse.py            Sparse matrix and sparse LP coverage
benchmark_data/
  netlib_dfl001/            Large public LP benchmark data and metadata
```

## License

MIT
