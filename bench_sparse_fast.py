from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from linprogx.sparse import csr_matrix


@dataclass(frozen=True)
class FastSparseCase:
    name: str
    c: list[float]
    A: Any
    b: list[float]
    bounds: list[tuple[float, float]]


@dataclass(frozen=True)
class FastSparseRow:
    case: str
    variant: str
    status: str
    objective: float
    objective_delta: float | None
    max_residual: float
    iterations: int
    seconds: float
    objective_scale: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast sparse PDHG benchmark variants.")
    parser.add_argument("--iterations", type=int, default=4_000)
    parser.add_argument("--check-interval", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = run_fast_benchmark(
        max_iterations=args.iterations,
        check_interval=args.check_interval,
        repeats=args.repeats,
    )
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))
    else:
        print_table(rows)
    return 0


def run_fast_benchmark(
    *, max_iterations: int = 4_000, check_interval: int = 250, repeats: int = 3
) -> list[FastSparseRow]:
    rows: list[FastSparseRow] = []
    for case in make_cases():
        baseline = _scipy_baseline(case)
        for name, objective_scale in _variants(case):
            rows.append(
                _run_variant(
                    case,
                    name,
                    objective_scale,
                    baseline,
                    max_iterations,
                    check_interval,
                    repeats,
                )
            )
    return rows


def print_table(rows: list[FastSparseRow]) -> None:
    print(
        f"{'case':22} {'variant':14} {'status':16} {'delta':>11} "
        f"{'residual':>11} {'iters':>7} {'ms':>9} {'scale':>10}"
    )
    print("-" * 108)
    for row in rows:
        delta = "n/a" if row.objective_delta is None else f"{row.objective_delta:.2e}"
        print(
            f"{row.case:22} {row.variant:14} {row.status:16} {delta:>11} "
            f"{row.max_residual:11.2e} {row.iterations:7d} "
            f"{row.seconds * 1000:9.2f} {row.objective_scale:10.3g}"
        )


def make_cases() -> tuple[FastSparseCase, ...]:
    return (
        _grouped_box_case(groups=8, width=6, demand=3.0),
        _chain_flow_case(nodes=18),
        _random_feasible_case(rows=20, cols=80, row_width=8, seed=7),
    )


def _grouped_box_case(*, groups: int, width: int, demand: float) -> FastSparseCase:
    c: list[float] = []
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for group in range(groups):
        start = group * width
        for offset in range(width):
            c.append(float(1 + ((group + offset * 3) % width)))
            indices.append(start + offset)
            data.append(1.0)
        indptr.append(len(indices))
    cols = groups * width
    return FastSparseCase(
        "grouped_box",
        c,
        csr_matrix(groups, cols, indptr, indices, data),
        [demand] * groups,
        [(0.0, 1.0)] * cols,
    )


def _chain_flow_case(*, nodes: int) -> FastSparseCase:
    arcs = [(node, node + 1) for node in range(nodes - 1)]
    arcs.extend((node, min(nodes - 1, node + 3)) for node in range(nodes - 3))
    c = [1.0 + (index % 7) for index in range(len(arcs))]
    rows = nodes - 1
    by_row: list[list[tuple[int, float]]] = [[] for _ in range(rows)]
    for col, (tail, head) in enumerate(arcs):
        if tail < rows:
            by_row[tail].append((col, 1.0))
        if head < rows:
            by_row[head].append((col, -1.0))
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row_entries in by_row:
        for col, value in row_entries:
            indices.append(col)
            data.append(value)
        indptr.append(len(indices))
    b = [0.0] * rows
    b[0] = 1.0
    return FastSparseCase(
        "chain_flow",
        c,
        csr_matrix(rows, len(arcs), indptr, indices, data),
        b,
        [(0.0, 1.0)] * len(arcs),
    )


def _random_feasible_case(*, rows: int, cols: int, row_width: int, seed: int) -> FastSparseCase:
    rng = random.Random(seed)
    x_seed = [0.0] * cols
    for col in range(cols):
        if col % 5 == 0:
            x_seed[col] = 0.5
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    b: list[float] = []
    for _row in range(rows):
        chosen = sorted(rng.sample(range(cols), row_width))
        total = 0.0
        for col in chosen:
            value = float(rng.choice([-2, -1, 1, 2]))
            indices.append(col)
            data.append(value)
            total += value * x_seed[col]
        indptr.append(len(indices))
        b.append(total)
    c = [float(1 + (col * 17) % 23) for col in range(cols)]
    return FastSparseCase(
        "random_feasible",
        c,
        csr_matrix(rows, cols, indptr, indices, data),
        b,
        [(0.0, 1.0)] * cols,
    )


def _variants(case: FastSparseCase) -> tuple[tuple[str, float], ...]:
    nonzero = sorted(abs(value) for value in case.c if value != 0.0)
    median = nonzero[len(nonzero) // 2] if nonzero else 1.0
    max_value = max(nonzero, default=1.0)
    return (
        ("auto", 0.0),
        ("median", median),
        ("max", max_value),
        ("unit", 1.0),
    )


def _run_variant(
    case: FastSparseCase,
    variant: str,
    objective_scale: float,
    baseline: float | None,
    max_iterations: int,
    check_interval: int,
    repeats: int,
) -> FastSparseRow:
    result: dict[str, Any] | None = None
    start = time.perf_counter()
    for _ in range(repeats):
        result = case.A.solve_eq_box_pdhg(
            case.c,
            case.b,
            [lower for lower, _ in case.bounds],
            [upper for _, upper in case.bounds],
            max_iter=max_iterations,
            tol=1e-6,
            check_interval=check_interval,
            objective_scale=objective_scale,
        )
    seconds = (time.perf_counter() - start) / repeats
    assert result is not None
    objective = float(result["objective"])
    return FastSparseRow(
        case.name,
        variant,
        str(result["status"]),
        objective,
        None if baseline is None else abs(objective - baseline),
        float(result["max_primal_residual"]),
        int(result["iterations"]),
        seconds,
        float(result["objective_scale"]),
    )


def _scipy_baseline(case: FastSparseCase) -> float | None:
    try:
        from scipy import sparse
        from scipy.optimize import linprog
    except ImportError:
        return None

    indptr, indices, data = case.A.to_components()
    matrix = sparse.csr_matrix((data, indices, indptr), shape=case.A.shape)
    result = linprog(case.c, A_eq=matrix, b_eq=case.b, bounds=case.bounds, method="highs")
    return float(result.fun) if result.success else None


if __name__ == "__main__":
    raise SystemExit(main())
