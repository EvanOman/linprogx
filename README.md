# linprogx

[![CI](https://github.com/EvanOman/linprogx/actions/workflows/ci.yml/badge.svg)](https://github.com/EvanOman/linprogx/actions/workflows/ci.yml)
![coverage](assets/coverage.svg)

`linprogx` is a small linear programming solver written from scratch in Python, with an optional C extension for the tableau operations that run in the tight loop.

It solves continuous linear programs with maximization or minimization objectives, `<=`, `>=`, and equality constraints, finite or infinite variable bounds, and free variables. The core algorithm is a two-phase primal simplex method: phase I builds a feasible basis with artificial variables, and phase II optimizes the requested objective.

Reference link saved with the project: https://www.linkedin.com/posts/antonvorobets_if-you-think-ai-can-write-advanced-analytics-share-7460672846042710016-z_ya?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAk_CVIBr27sXDGNG8kKqOPnWZAdJrVOA7Q

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

`tests/test_samples_compare.py` solves every sample with `linprogx`, solves the same model with SciPy/HiGHS and Clarabel, and asserts matching statuses and objective values. Clarabel is an interior-point conic solver, so its objective comparison uses a slightly wider tolerance than SciPy/HiGHS.

Run the benchmark:

```bash
uv run python bench.py
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

The benchmark prints all 16 samples; the excerpt above keeps the README short. These timings are for tiny dense models where Python call overhead dominates. SciPy/HiGHS and Clarabel are the right baselines for larger models.

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
