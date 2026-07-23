"""Falsify degree-2 unit-series contraction on the presolved pds fixtures.

This is deliberately a throwaway experiment, not solver code.  It contracts
only mathematically exact series chains: every interior row must contain
exactly the two unit, degree-2 arc columns in the chain.  The script first
checks the signed-affine bound algebra exhaustively on synthetic chains, then
measures the unchanged PDHG path on pds_10 and pds_20.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.optimize import linprog

from linprogx.presolve import PresolveResult, postsolve_x, presolve_matrix
from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

EPS = 2e-5
FIXTURES = ("pds_10", "pds_20")
HIGHS_SHAPES = {
    "pds_10": (4_092, 32_646, 78_216),
    "pds_20": (8_984, 79_990, 188_070),
}


@dataclass(frozen=True)
class Chain:
    endpoint_rows: tuple[int, int]
    interior_rows: tuple[int, ...]
    arc_cols: tuple[int, ...]
    sigma: tuple[float, ...]
    delta: tuple[float, ...]
    lo: float
    hi: float


@dataclass
class ContractedProblem:
    A: sparse.csr_matrix
    b: np.ndarray
    c: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    objective_offset: float
    chains: list[Chain]
    kept_cols: np.ndarray
    kept_rows: np.ndarray
    original_cols: int

    def reconstruct(self, y: np.ndarray) -> np.ndarray:
        x = np.empty(self.original_cols, dtype=float)
        x[self.kept_cols] = y[: len(self.kept_cols)]
        for chain_index, chain in enumerate(self.chains):
            s = float(y[len(self.kept_cols) + chain_index])
            for col, sigma, delta in zip(chain.arc_cols, chain.sigma, chain.delta, strict=True):
                x[col] = sigma * s + delta
        return x


@dataclass(frozen=True)
class SolveFacts:
    status: str
    iterations: int
    wall_seconds: float
    reduced_objective: float
    original_objective: float
    original_equality_residual: float
    original_bound_residual: float
    oracle_objective: float
    oracle_absolute_delta: float
    oracle_relative_delta: float


def load_fixture(path: Path) -> dict[str, Any]:
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": raw["A"].tocsr().astype(np.float64),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def to_scipy(matrix: Any) -> sparse.csr_matrix:
    indptr, indices, data = matrix.to_components()
    return sparse.csr_matrix((data, indices, indptr), shape=matrix.shape)


def mapped_interval(lo: float, hi: float, sigma: float, delta: float) -> tuple[float, float]:
    """Map lo <= sigma*s + delta <= hi to an interval on s."""
    if sigma == 1.0:
        return lo - delta, hi - delta
    if sigma == -1.0:
        return delta - hi, delta - lo
    raise ValueError(f"series-chain sigma must be +/-1, got {sigma}")


def find_series_chains(
    A: sparse.csr_matrix,
    b: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> list[Chain]:
    """Find disjoint exact paths with total-degree-2 interior rows."""
    A = A.tocsr()
    C = A.tocsc()
    col_degree = np.diff(C.indptr)
    unit_degree2 = np.zeros(A.shape[1], dtype=bool)
    for j in np.flatnonzero(col_degree == 2):
        values = C.data[C.indptr[j] : C.indptr[j + 1]]
        unit_degree2[j] = bool(np.all(np.abs(values) == 1.0))

    eligible_rows: set[int] = set()
    row_arcs: dict[int, tuple[int, int]] = {}
    for i in range(A.shape[0]):
        cols = A.indices[A.indptr[i] : A.indptr[i + 1]]
        if len(cols) == 2 and bool(np.all(unit_degree2[cols])):
            eligible_rows.add(i)
            row_arcs[i] = (int(cols[0]), int(cols[1]))

    arc_rows: dict[int, tuple[int, int]] = {}
    for j in np.flatnonzero(unit_degree2):
        rows = C.indices[C.indptr[j] : C.indptr[j + 1]]
        arc_rows[int(j)] = (int(rows[0]), int(rows[1]))

    unseen = set(eligible_rows)
    chains: list[Chain] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        stack = [seed]
        while stack:
            row = stack.pop()
            for arc in row_arcs[row]:
                for neighbor in arc_rows[arc]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)

        incident_arcs = {arc for row in component for arc in row_arcs[row]}
        boundary: list[tuple[int, int]] = []
        for arc in incident_arcs:
            for row in arc_rows[arc]:
                if row not in component:
                    boundary.append((row, arc))
        if len(boundary) != 2 or len(incident_arcs) != len(component) + 1:
            # A pure cycle has no boundary and is not a one-parameter series path.
            continue

        start_row, arc = boundary[0]
        end_row = boundary[1][0]
        ordered_arcs: list[int] = []
        ordered_interior: list[int] = []
        previous_row = start_row
        while True:
            ordered_arcs.append(arc)
            rows = arc_rows[arc]
            next_row = rows[1] if rows[0] == previous_row else rows[0]
            if next_row not in component:
                if next_row != end_row or len(ordered_arcs) != len(incident_arcs):
                    raise AssertionError("series component did not trace as one path")
                break
            ordered_interior.append(next_row)
            next_arcs = row_arcs[next_row]
            next_arc = next_arcs[1] if next_arcs[0] == arc else next_arcs[0]
            previous_row, arc = next_row, next_arc

        sigma = [1.0]
        delta = [0.0]
        for position, row in enumerate(ordered_interior):
            left_arc = ordered_arcs[position]
            right_arc = ordered_arcs[position + 1]
            start, stop = A.indptr[row], A.indptr[row + 1]
            terms = dict(zip(A.indices[start:stop], A.data[start:stop], strict=True))
            a_left = float(terms[left_arc])
            a_right = float(terms[right_arc])
            sigma.append((-a_left * sigma[-1]) / a_right)
            delta.append((float(b[row]) - a_left * delta[-1]) / a_right)

        combined_lo = -math.inf
        combined_hi = math.inf
        for col, sign, shift in zip(ordered_arcs, sigma, delta, strict=True):
            mapped_lo, mapped_hi = mapped_interval(float(lo[col]), float(hi[col]), sign, shift)
            combined_lo = max(combined_lo, mapped_lo)
            combined_hi = min(combined_hi, mapped_hi)
        if combined_lo > combined_hi:
            raise ValueError("series-chain bounds prove the input problem infeasible")

        chains.append(
            Chain(
                endpoint_rows=(start_row, end_row),
                interior_rows=tuple(ordered_interior),
                arc_cols=tuple(ordered_arcs),
                sigma=tuple(sigma),
                delta=tuple(delta),
                lo=combined_lo,
                hi=combined_hi,
            )
        )
    return sorted(chains, key=lambda chain: chain.arc_cols)


def contract_series_chains(
    A: sparse.csr_matrix,
    b: np.ndarray,
    c: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> ContractedProblem:
    chains = find_series_chains(A, b, lo, hi)
    removed_cols = {col for chain in chains for col in chain.arc_cols}
    removed_rows = {row for chain in chains for row in chain.interior_rows}
    kept_cols = np.array([j for j in range(A.shape[1]) if j not in removed_cols], dtype=int)
    kept_rows = np.array([i for i in range(A.shape[0]) if i not in removed_rows], dtype=int)
    row_map = {int(old): new for new, old in enumerate(kept_rows)}

    reduced = A[kept_rows][:, kept_cols].tolil()
    new_b = b[kept_rows].copy()
    new_c = c[kept_cols].tolist()
    new_lo = lo[kept_cols].tolist()
    new_hi = hi[kept_cols].tolist()
    objective_offset = 0.0

    for chain in chains:
        column = len(new_c)
        reduced.resize((len(kept_rows), column + 1))
        first_arc, last_arc = chain.arc_cols[0], chain.arc_cols[-1]
        first_row, last_row = chain.endpoint_rows
        first_coef = float(A[first_row, first_arc])
        last_coef = float(A[last_row, last_arc])
        reduced[row_map[first_row], column] = first_coef
        reduced[row_map[last_row], column] = last_coef * chain.sigma[-1]
        new_b[row_map[last_row]] -= last_coef * chain.delta[-1]
        new_c.append(
            sum(c[col] * sign for col, sign in zip(chain.arc_cols, chain.sigma, strict=True))
        )
        objective_offset += sum(
            c[col] * shift for col, shift in zip(chain.arc_cols, chain.delta, strict=True)
        )
        new_lo.append(chain.lo)
        new_hi.append(chain.hi)

    return ContractedProblem(
        A=reduced.tocsr(),
        b=new_b,
        c=np.asarray(new_c, dtype=float),
        lo=np.asarray(new_lo, dtype=float),
        hi=np.asarray(new_hi, dtype=float),
        objective_offset=float(objective_offset),
        chains=chains,
        kept_cols=kept_cols,
        kept_rows=kept_rows,
        original_cols=A.shape[1],
    )


def scipy_bounds(lo: np.ndarray, hi: np.ndarray) -> list[tuple[float | None, float | None]]:
    return [
        (
            None if math.isinf(float(low)) and low < 0 else float(low),
            None if math.isinf(float(up)) and up > 0 else float(up),
        )
        for low, up in zip(lo, hi, strict=True)
    ]


def synthetic_checks() -> dict[str, int]:
    """Exhaust all 64 coefficient-sign patterns for a three-arc chain."""
    cases = 0
    for signs in itertools.product((-1.0, 1.0), repeat=6):
        A = sparse.lil_matrix((4, 5), dtype=float)
        # Variables: three path arcs, then one private variable at each endpoint.
        for arc in range(3):
            A[arc, arc] = signs[2 * arc]
            A[arc + 1, arc] = signs[2 * arc + 1]
        A[0, 3] = 1.0
        A[3, 4] = -1.0
        target = np.array([1.0, 2.0, 1.5])
        b = np.zeros(4)
        b[1] = A[1, 0] * target[0] + A[1, 1] * target[1]
        b[2] = A[2, 1] * target[1] + A[2, 2] * target[2]
        c = np.array([3.0, -2.0, 4.0, 0.25, -0.5])
        lo = np.array([0.0, 0.0, -math.inf, -10.0, -10.0])
        hi = np.array([math.inf, 4.0, 3.0, 10.0, 10.0])

        contracted = contract_series_chains(A.tocsr(), b, c, lo, hi)
        assert len(contracted.chains) == 1
        assert contracted.A.shape == (2, 3)
        original = linprog(c, A_eq=A.tocsr(), b_eq=b, bounds=scipy_bounds(lo, hi), method="highs")
        smaller = linprog(
            contracted.c,
            A_eq=contracted.A,
            b_eq=contracted.b,
            bounds=scipy_bounds(contracted.lo, contracted.hi),
            method="highs",
        )
        assert original.success and smaller.success
        reconstructed = contracted.reconstruct(np.asarray(smaller.x))
        assert np.max(np.abs(A @ reconstructed - b)) <= 1e-10
        reconstructed_objective = float(c @ reconstructed)
        transformed_objective = float(smaller.fun + contracted.objective_offset)
        assert abs(reconstructed_objective - transformed_objective) <= 1e-10
        assert abs(float(original.fun) - reconstructed_objective) <= 1e-9
        cases += 1

    # Counterexample to the tempting but invalid generalized merge.  Eliminating
    # x from u+x=0, v-x=0 with 0<=x<=1 requires BOTH u+v=0 and -1<=u<=0.
    # One ranged equality u+v=s, s in [-1, 1] admits (u,v)=(0,1), which has no x.
    invalid_generalized_witnesses = 1
    return {
        "signed_chain_cases": cases,
        "invalid_generalized_witnesses": invalid_generalized_witnesses,
    }


def bound_residual(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(max(0.0, np.max(lo - x), np.max(x - hi)))


def solve_pdhg(
    contracted: ContractedProblem,
    presolve: PresolveResult,
    raw: dict[str, Any],
    oracle_objective: float,
) -> SolveFacts:
    problem = SparseLPProblem(
        c=contracted.c.tolist(),
        A_eq=from_scipy_sparse(contracted.A),
        b_eq=contracted.b.tolist(),
        objective="min",
        bounds=scipy_bounds(contracted.lo, contracted.hi),
    )
    start = time.perf_counter()
    result = SparseSolver(
        algorithm="pdhg",
        max_iterations=50_000,
        eps=EPS,
        check_interval=50_000,
        presolve=False,
    ).solve(problem)
    wall = time.perf_counter() - start
    reduced_x = contracted.reconstruct(np.asarray(result.solution.x, dtype=float))
    original_x = np.asarray(postsolve_x(reduced_x.tolist(), presolve), dtype=float)
    original_objective = float(raw["c"] @ original_x)
    absolute_delta = abs(original_objective - oracle_objective)
    assert result.solution.objective_value is not None
    return SolveFacts(
        status=result.solution.status.value,
        iterations=result.solution.iterations,
        wall_seconds=wall,
        reduced_objective=float(result.solution.objective_value + contracted.objective_offset),
        original_objective=original_objective,
        original_equality_residual=float(np.max(np.abs(raw["A"] @ original_x - raw["b"]))),
        original_bound_residual=bound_residual(original_x, raw["lo"], raw["hi"]),
        oracle_objective=oracle_objective,
        oracle_absolute_delta=absolute_delta,
        oracle_relative_delta=absolute_delta / max(1.0, abs(oracle_objective)),
    )


def run_fixture(directory: Path, name: str) -> dict[str, Any]:
    raw = load_fixture(directory / f"lp_{name}.mat")
    matrix = from_scipy_sparse(raw["A"])
    presolve = presolve_matrix(
        matrix,
        raw["b"].tolist(),
        raw["c"].tolist(),
        raw["lo"].tolist(),
        raw["hi"].tolist(),
        algorithm="pdhg",
    )
    if presolve is None:
        raise AssertionError(f"{name}: expected current presolve to reduce fixture")
    reduced_A = to_scipy(presolve._matrix)
    reduced_b = np.asarray(presolve.b)
    reduced_c = np.asarray(presolve.c)
    reduced_lo = np.asarray(presolve.lo)
    reduced_hi = np.asarray(presolve.hi)
    contracted = contract_series_chains(reduced_A, reduced_b, reduced_c, reduced_lo, reduced_hi)
    identity = ContractedProblem(
        A=reduced_A,
        b=reduced_b,
        c=reduced_c,
        lo=reduced_lo,
        hi=reduced_hi,
        objective_offset=0.0,
        chains=[],
        kept_cols=np.arange(reduced_A.shape[1]),
        kept_rows=np.arange(reduced_A.shape[0]),
        original_cols=reduced_A.shape[1],
    )

    oracle_start = time.perf_counter()
    oracle = linprog(
        raw["c"],
        A_eq=raw["A"],
        b_eq=raw["b"],
        bounds=scipy_bounds(raw["lo"], raw["hi"]),
        method="highs",
    )
    oracle_wall = time.perf_counter() - oracle_start
    if not oracle.success:
        raise AssertionError(f"{name}: HiGHS oracle failed: {oracle.message}")

    baseline = solve_pdhg(identity, presolve, raw, float(oracle.fun))
    contracted_solve = solve_pdhg(contracted, presolve, raw, float(oracle.fun))
    baseline_proxy = baseline.iterations * reduced_A.nnz
    contracted_proxy = contracted_solve.iterations * contracted.A.nnz
    return {
        "name": name,
        "raw_shape": (*raw["A"].shape, raw["A"].nnz),
        "baseline_shape": (*reduced_A.shape, reduced_A.nnz),
        "contracted_shape": (*contracted.A.shape, contracted.A.nnz),
        "highs_shape": HIGHS_SHAPES[name],
        "eligible_interior_rows": sum(
            1
            for i in range(reduced_A.shape[0])
            if reduced_A.indptr[i + 1] - reduced_A.indptr[i] == 2
        ),
        "chains": len(contracted.chains),
        "contracted_arcs": sum(len(chain.arc_cols) for chain in contracted.chains),
        "oracle_wall_seconds": oracle_wall,
        "baseline": asdict(baseline),
        "contracted": asdict(contracted_solve),
        "baseline_work_proxy": baseline_proxy,
        "contracted_work_proxy": contracted_proxy,
        "projected_wall_gain": 1.0 - contracted_proxy / baseline_proxy,
        "measured_wall_gain": 1.0 - contracted_solve.wall_seconds / baseline.wall_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("/tmp/lpsuite"))
    args = parser.parse_args()
    synthetic = synthetic_checks()
    results = [run_fixture(args.directory, name) for name in FIXTURES]
    pds10, pds20 = results
    gates = {
        "pds_10_proxy_gain_at_least_15pct": pds10["projected_wall_gain"] >= 0.15,
        "pds_10_objective_and_residual": (
            pds10["contracted"]["status"] == "optimal"
            and pds10["contracted"]["oracle_relative_delta"] <= EPS
            and pds10["contracted"]["original_equality_residual"] <= EPS
            and pds10["contracted"]["original_bound_residual"] <= EPS
        ),
        "pds_20_flat_or_better": pds20["contracted_work_proxy"] <= pds20["baseline_work_proxy"],
    }
    output = {
        "synthetic": synthetic,
        "fixtures": results,
        "gates": gates,
        "verdict": "LIVE" if all(gates.values()) else "KILLED",
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
