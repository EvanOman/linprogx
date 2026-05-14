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
from linprogx import solve

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
```

Recorded creation time: 3 minutes 10 seconds from the initial solver commit (`15192d8`, 2026-05-14 16:37:05 CDT) to the standardized benchmark commit (`96a5a65`, 2026-05-14 16:40:15 CDT). That is measured from git history, so it excludes the uncommitted pre-history before the first commit.

## Development

```bash
just install
just test
just test-cov
just fc
```

## Architecture

```text
src/linprogx/
  __init__.py      Public exports
  solver.py        Problem normalization and two-phase simplex
  builder.py       Small modeling interface
  cli.py           JSON command-line interface
  samples.py       Deterministic sample LP library
  compare.py       SciPy/HiGHS and Clarabel correctness/timing comparison
  _fast.py         C-extension dispatch and Python fallback
  _cfast.c         In-place pivot and dot-product C helpers
tests/
  test_solver.py            Solver, bounds, CLI, and validation coverage
  test_samples_compare.py   Sample problem checks against SciPy/HiGHS
```

## License

MIT
