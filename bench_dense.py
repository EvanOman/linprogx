from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from linprogx import solve
from linprogx.compare import compare_with_clarabel, compare_with_scipy
from linprogx.types import Constraint, LPProblem

SEED = 42
VARIABLES = 160
CONSTRAINTS = 320


@dataclass(frozen=True)
class DenseBenchRow:
    solver: str
    status: str
    objective: float | None
    objective_delta: float | None
    seconds: float
    notes: str


@dataclass(frozen=True)
class DenseBenchResult:
    name: str
    variables: int
    constraints: int
    dense_coefficients: int
    expected_objective: float
    rows: list[DenseBenchRow]


def main() -> int:
    out = Path("assets/dense_160x320_results.json")
    markdown = Path("assets/dense_160x320_summary.md")
    plot = Path("assets/dense_160x320_runtime.png")
    problem, expected = make_dense_problem()
    result = run_dense_benchmark(problem, expected)
    out.write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    write_markdown(result, markdown)
    write_plot(result, plot)
    print(json.dumps(_jsonable(result), indent=2))
    return 0


def make_dense_problem() -> tuple[LPProblem, float]:
    rng = random.Random(SEED)
    c = [1.0 + rng.random() for _ in range(VARIABLES)]
    constraints = []
    for row_index in range(CONSTRAINTS):
        row = [0.1 + rng.random() for _ in range(VARIABLES)]
        constraints.append(Constraint(row, "<=", sum(row), f"dense_{row_index}"))
    bounds: list[tuple[float | None, float | None]] = [(0.0, 1.0)] * VARIABLES
    expected = sum(c)
    return LPProblem(c, constraints, "max", bounds, "dense_box_160x320"), expected


def run_dense_benchmark(problem: LPProblem, expected: float) -> DenseBenchResult:
    own_start = time.perf_counter()
    own = solve(
        problem.c,
        [constraint.coefficients for constraint in problem.constraints],
        [constraint.rhs for constraint in problem.constraints],
        objective="max",
        bounds=problem.bounds,
    )
    own_seconds = time.perf_counter() - own_start
    scipy = compare_with_scipy(problem)
    clarabel = compare_with_clarabel(problem)
    return DenseBenchResult(
        name=problem.name,
        variables=VARIABLES,
        constraints=CONSTRAINTS,
        dense_coefficients=VARIABLES * CONSTRAINTS,
        expected_objective=expected,
        rows=[
            DenseBenchRow(
                "linprogx",
                own.status.value,
                own.objective_value,
                None if own.objective_value is None else abs(own.objective_value - expected),
                own_seconds,
                f"{own.iterations} simplex iterations",
            ),
            DenseBenchRow(
                "SciPy/HiGHS",
                scipy.solver_status,
                scipy.solver_objective,
                scipy.objective_delta,
                scipy.solver_seconds,
                "Open-source sparse/dense LP baseline",
            ),
            DenseBenchRow(
                "Clarabel",
                clarabel.solver_status,
                clarabel.solver_objective,
                clarabel.objective_delta,
                clarabel.solver_seconds,
                "Open-source conic interior-point baseline",
            ),
        ],
    )


def write_markdown(result: DenseBenchResult, path: Path) -> None:
    lines = [
        "| Solver | Status | Objective | Delta vs linprogx/expected | Runtime | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in result.rows:
        objective = "n/a" if row.objective is None else f"{row.objective:.6f}"
        delta = "n/a" if row.objective_delta is None else f"{row.objective_delta:.3e}"
        lines.append(
            f"| {row.solver} | {row.status} | {objective} | {delta} | {row.seconds:.3f}s | {row.notes} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_plot(result: DenseBenchResult, path: Path) -> None:
    import importlib

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    pyplot: Any = importlib.import_module("matplotlib.pyplot")

    fig, ax = pyplot.subplots(figsize=(9, 4.8))
    colors = ["#28536b", "#c44536", "#f3a712"]
    ax.bar([row.solver for row in result.rows], [row.seconds for row in result.rows], color=colors)
    ax.set_ylabel("seconds")
    ax.set_title("Dense 160 Variable / 320 Constraint Runtime")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
    for index, row in enumerate(result.rows):
        ax.text(index, row.seconds, f"{row.seconds:.3f}s", ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    pyplot.close(fig)


def _jsonable(result: DenseBenchResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "variables": result.variables,
        "constraints": result.constraints,
        "dense_coefficients": result.dense_coefficients,
        "expected_objective": result.expected_objective,
        "rows": [asdict(row) for row in result.rows],
    }


if __name__ == "__main__":
    raise SystemExit(main())
