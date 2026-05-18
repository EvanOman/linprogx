"""Head-to-head Rust vs C PDHG benchmark.

Reuses the fast sparse cases from ``bench_sparse_fast`` and runs each one twice:
once against ``linprogx._csparse.CSRMatrix`` and once against
``linprogx._rsparse.CSRMatrix``. Also runs a single DFL001 probe so we can see
whether the experimental Rust backend changes large-scale throughput.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_sparse_fast import (
    _chain_flow_case,
    _grouped_box_case,
    _random_feasible_case,
    _scipy_baseline,
)
from linprogx.sparse import csr_matrix, csr_matrix_rust

DFL001_PATH = Path("benchmark_data/netlib_dfl001/lp_dfl001.mat")
CYCLE_PATH = Path("benchmark_data/netlib_cycle/lp_cycle.mat")


@dataclass(frozen=True)
class Row:
    case: str
    backend: str
    status: str
    objective: float
    objective_delta: float | None
    max_residual: float
    iterations: int
    seconds: float


def _build_rust_clone(c_case: Any) -> Any:
    indptr, indices, data = c_case.A.to_components()
    return csr_matrix_rust(
        c_case.A.shape[0],
        c_case.A.shape[1],
        [int(v) for v in indptr],
        [int(v) for v in indices],
        [float(v) for v in data],
    )


def make_cases() -> list[tuple[str, Any, Any, list[float], list[float], list[tuple[float, float]]]]:
    cases = (
        _grouped_box_case(groups=8, width=6, demand=3.0),
        _chain_flow_case(nodes=18),
        _random_feasible_case(rows=20, cols=80, row_width=8, seed=7),
    )
    out = []
    for case in cases:
        rust_matrix = _build_rust_clone(case)
        out.append((case.name, case.A, rust_matrix, case.c, case.b, case.bounds))
    return out


def run_variant(
    matrix: Any,
    c: list[float],
    b: list[float],
    bounds: list[tuple[float, float]],
    *,
    max_iter: int,
    check_interval: int,
    repeats: int,
    objective_scale: float = 0.0,
) -> tuple[dict[str, Any], float]:
    result: dict[str, Any] | None = None
    start = time.perf_counter()
    for _ in range(repeats):
        result = matrix.solve_eq_box_pdhg(
            c,
            b,
            [lo for lo, _ in bounds],
            [hi for _, hi in bounds],
            max_iter=max_iter,
            tol=1e-6,
            check_interval=check_interval,
            objective_scale=objective_scale,
        )
    seconds = (time.perf_counter() - start) / repeats
    assert result is not None
    return result, seconds


def run_small_cases(max_iter: int, check_interval: int, repeats: int) -> list[Row]:
    out: list[Row] = []
    for name, c_matrix, rust_matrix, c_vec, b_vec, bounds in make_cases():
        # Compute scipy baseline for objective delta reporting.
        from bench_sparse_fast import FastSparseCase

        case = FastSparseCase(name, c_vec, c_matrix, b_vec, bounds)
        baseline = _scipy_baseline(case)
        for backend_name, matrix in (("c", c_matrix), ("rust", rust_matrix)):
            result, seconds = run_variant(
                matrix,
                c_vec,
                b_vec,
                bounds,
                max_iter=max_iter,
                check_interval=check_interval,
                repeats=repeats,
            )
            objective = float(result["objective"])
            out.append(
                Row(
                    case=name,
                    backend=backend_name,
                    status=str(result["status"]),
                    objective=objective,
                    objective_delta=None if baseline is None else abs(objective - baseline),
                    max_residual=float(result["max_primal_residual"]),
                    iterations=int(result["iterations"]),
                    seconds=seconds,
                )
            )
    return out


def run_netlib(
    path: Path,
    name: str,
    max_iter: int,
    check_interval: int,
    eps: float,
    objective_scale: float,
) -> list[Row]:
    from scipy.io import loadmat

    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    csr = raw["A"].tocsr()
    indptr = [int(v) for v in csr.indptr.tolist()]
    indices = [int(v) for v in csr.indices.tolist()]
    data = [float(v) for v in csr.data.tolist()]
    rows, cols = int(csr.shape[0]), int(csr.shape[1])
    b_vec = [float(v) for v in raw["b"].ravel().tolist()]
    c_vec = [float(v) for v in aux["c"].ravel().tolist()]
    lo = [float(v) for v in aux["lo"].ravel().tolist()]
    hi = [float(v) for v in aux["hi"].ravel().tolist()]

    out: list[Row] = []
    for backend_name, factory in (("c", csr_matrix), ("rust", csr_matrix_rust)):
        matrix = factory(rows, cols, indptr, indices, data)
        start = time.perf_counter()
        result = matrix.solve_eq_box_pdhg(
            c_vec,
            b_vec,
            lo,
            hi,
            max_iter=max_iter,
            tol=eps,
            check_interval=check_interval,
            objective_scale=objective_scale,
        )
        seconds = time.perf_counter() - start
        out.append(
            Row(
                case=name,
                backend=backend_name,
                status=str(result["status"]),
                objective=float(result["objective"]),
                objective_delta=None,
                max_residual=float(result["max_primal_residual"]),
                iterations=int(result["iterations"]),
                seconds=seconds,
            )
        )
    return out


def print_table(rows: list[Row]) -> None:
    print(
        f"{'case':18} {'backend':8} {'status':16} {'objective':>14} "
        f"{'residual':>11} {'iters':>7} {'seconds':>10}"
    )
    print("-" * 90)
    pair: dict[str, float] = {}
    for row in rows:
        print(
            f"{row.case:18} {row.backend:8} {row.status:16} "
            f"{row.objective:14.4f} {row.max_residual:11.2e} "
            f"{row.iterations:7d} {row.seconds:10.4f}"
        )
        key = row.case + ":" + row.backend
        pair[key] = row.seconds
    print()
    print("Speedup (C / Rust):")
    cases = sorted({r.case for r in rows})
    for case in cases:
        c_time = pair.get(case + ":c")
        rust_time = pair.get(case + ":rust")
        if c_time and rust_time:
            ratio = c_time / rust_time
            print(f"  {case:18} c={c_time*1000:.2f}ms rust={rust_time*1000:.2f}ms ratio={ratio:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=4_000)
    parser.add_argument("--check-interval", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dfl001", action="store_true", help="also run a Netlib DFL001 probe")
    parser.add_argument("--dfl001-iter", type=int, default=20_000)
    parser.add_argument("--dfl001-check", type=int, default=2_500)
    parser.add_argument("--dfl001-eps", type=float, default=1e-4)
    parser.add_argument("--dfl001-scale", type=float, default=15_000.0)
    parser.add_argument("--cycle", action="store_true", help="also run a Netlib CYCLE probe")
    parser.add_argument("--cycle-iter", type=int, default=20_000)
    parser.add_argument("--cycle-check", type=int, default=5_000)
    parser.add_argument("--cycle-eps", type=float, default=1e-4)
    parser.add_argument("--cycle-scale", type=float, default=0.0)
    args = parser.parse_args()

    small = run_small_cases(args.iterations, args.check_interval, args.repeats)
    print("# Small fast cases (per-call seconds; repeats =", args.repeats, ")")
    print_table(small)
    if args.dfl001:
        if not DFL001_PATH.exists():
            print(f"\nDFL001 data missing at {DFL001_PATH}; skipping.")
        else:
            print("\n# DFL001 probe")
            rows = run_netlib(
                DFL001_PATH,
                "dfl001",
                args.dfl001_iter,
                args.dfl001_check,
                args.dfl001_eps,
                args.dfl001_scale,
            )
            print_table(rows)
    if args.cycle:
        if not CYCLE_PATH.exists():
            print(f"\nCYCLE data missing at {CYCLE_PATH}; skipping.")
        else:
            print("\n# CYCLE probe")
            rows = run_netlib(
                CYCLE_PATH,
                "cycle",
                args.cycle_iter,
                args.cycle_check,
                args.cycle_eps,
                args.cycle_scale,
            )
            print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
