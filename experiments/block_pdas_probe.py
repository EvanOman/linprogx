"""Standalone full-KKT block principal-pivot characterization.

This diagnostic deliberately does not call the production simplex trajectory.
It reconstructs the native triangular crash basis, assigns every nonbasic
structural variable to a true legal bound (or zero when it is genuinely free),
and repeatedly proposes rank-safe block basis exchanges.  Every proposal is
judged only after a fresh sparse LU and a complete recomputation of ``x_B``,
``y``, and reduced costs.

The policy is global and deterministic.  It contains no fixture-name, shape,
checkpoint, or greenbea-derived branch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import SuperLU, splu

from experiments.greenbea_pivot_gap_probe import prepare
from linprogx.presolve import postsolve_x

OUT_DIR = Path("/tmp/block-pdas-falsifier")
RESULTS = OUT_DIR / "results.json"

EPS = 2e-5
KKT_TOL = 1e-8
PIVOT_TOL = 1e-10
STEP_FP_MULTIPLIER = 64.0
BATCH_LADDER = (64, 32, 16, 8, 4, 2, 1)
MAX_ACCEPTED = 256
MAX_ATTEMPTED = 384
MEDIAN_WIDTH_GATE = 18.0
CONTROL_SECONDS = 0.560439628
BOARD_TARGET_SECONDS = 0.461234528
PROBE_TARGET_SECONDS = 0.448351702
REFAC_REFERENCE_SECONDS = 0.00103394
SENTINELS = ("woodw", "stocfor3", "80bau3b")

AT_LO = 0
AT_HI = 1
FREE = 2
FIXED = 3
BASIC = 4


def ruiz_scaled(a: sparse.csc_matrix) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Reproduce the native ten-pass inf-norm plus one L2 Ruiz scaling."""
    m, n = a.shape
    row_scale = np.ones(m)
    col_scale = np.ones(n)
    row_norm = np.asarray(abs(a).max(axis=1).toarray()).ravel()
    nonzero = row_norm[row_norm > 0.0]
    active = bool(nonzero.size and nonzero.max() / nonzero.min() >= 100.0)

    for _ in range(10 if active else 0):
        scaled = sparse.diags(row_scale) @ a @ sparse.diags(col_scale)
        row_norm = np.asarray(abs(scaled).max(axis=1).toarray()).ravel()
        col_norm = np.asarray(abs(scaled).max(axis=0).toarray()).ravel()
        row_scale[row_norm > 0.0] /= np.sqrt(row_norm[row_norm > 0.0])
        col_scale[col_norm > 0.0] /= np.sqrt(col_norm[col_norm > 0.0])

    if active:
        scaled = sparse.diags(row_scale) @ a @ sparse.diags(col_scale)
        row_l2_sq = np.asarray(scaled.multiply(scaled).sum(axis=1)).ravel()
        col_l2_sq = np.asarray(scaled.multiply(scaled).sum(axis=0)).ravel()
        row_scale[row_l2_sq > 0.0] /= np.sqrt(np.sqrt(row_l2_sq[row_l2_sq > 0.0]))
        col_scale[col_l2_sq > 0.0] /= np.sqrt(np.sqrt(col_l2_sq[col_l2_sq > 0.0]))
        row_scale = np.clip(row_scale, 1e-8, 1e8)
        col_scale = np.clip(col_scale, 1e-8, 1e8)

    return (
        (sparse.diags(row_scale) @ a @ sparse.diags(col_scale)).tocsc(),
        row_scale,
        col_scale,
    )


def basis_column(a: sparse.csc_matrix, column: int) -> sparse.csc_matrix:
    """Return one structural or identity-artificial column."""
    m, n = a.shape
    if column < n:
        return a[:, column]
    return sparse.csc_matrix(([1.0], ([column - n], [0])), shape=(m, 1))


def basis_matrix(a: sparse.csc_matrix, basis: np.ndarray) -> sparse.csc_matrix:
    return sparse.hstack([basis_column(a, int(column)) for column in basis], format="csc")


def crash_basis(
    a: sparse.csc_matrix, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce the production singleton-cascade triangular crash."""
    m, n = a.shape
    csr = a.tocsr()
    uncovered_count = np.diff(a.indptr).astype(np.int64)
    covered = np.zeros(m, dtype=bool)
    done = np.zeros(n, dtype=bool)

    penalties = np.empty(n, dtype=np.int8)
    for j in range(n):
        lo_finite = np.isfinite(lo[j])
        hi_finite = np.isfinite(hi[j])
        if not lo_finite and not hi_finite:
            penalties[j] = 0
        elif lo_finite and hi_finite:
            penalties[j] = 3 if hi[j] - lo[j] <= 1e-30 else 2
        else:
            penalties[j] = 1

    order = sorted(range(n), key=lambda j: (int(penalties[j]), int(uncovered_count[j]), j))
    queue = [j for j in order if uncovered_count[j] == 1]
    head = 0
    chosen: list[int] = []
    while head < len(queue) and len(chosen) < m:
        j = queue[head]
        head += 1
        if done[j] or uncovered_count[j] != 1:
            continue
        done[j] = True
        start, stop = a.indptr[j], a.indptr[j + 1]
        rows = a.indices[start:stop]
        values = np.abs(a.data[start:stop])
        live = ~covered[rows]
        if not np.any(live):
            continue
        pivot_row = int(rows[np.flatnonzero(live)[-1]])
        pivot_value = float(values[np.flatnonzero(live)[-1]])
        column_max = float(values.max(initial=0.0))
        if column_max <= 1e-12 or pivot_value < 0.5 * column_max:
            continue
        chosen.append(j)
        covered[pivot_row] = True
        for p in range(csr.indptr[pivot_row], csr.indptr[pivot_row + 1]):
            other = int(csr.indices[p])
            if not done[other] and uncovered_count[other] > 0:
                uncovered_count[other] -= 1
                if uncovered_count[other] == 1 and len(queue) < n:
                    queue.append(other)

    uncovered_rows = np.flatnonzero(~covered)
    basis = np.asarray(chosen + [n + int(row) for row in uncovered_rows], dtype=np.int64)
    if basis.size != m:
        raise RuntimeError(f"crash produced {basis.size} columns for {m} rows")

    matrix = basis_matrix(a, basis)
    fallback = False
    try:
        factor = splu(matrix)
        diagonal = np.abs(factor.U.diagonal())
        positive = diagonal[diagonal > 0.0]
        growth = float(diagonal.max() / positive.min()) if positive.size else math.inf
        if not np.isfinite(growth) or growth > 1e10:
            fallback = True
    except RuntimeError:
        growth = math.inf
        fallback = True
    if fallback:
        basis = np.arange(n, n + m, dtype=np.int64)
    return basis, {
        "structural_columns": int(np.sum(basis < n)),
        "artificial_columns": int(np.sum(basis >= n)),
        "growth_proxy": growth,
        "identity_fallback": fallback,
    }


@dataclass
class Model:
    original: dict[str, Any]
    prepared: dict[str, Any]
    a: sparse.csc_matrix
    a_aug: sparse.csc_matrix
    b: np.ndarray
    c: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    row_scale: np.ndarray
    col_scale: np.ndarray
    n: int
    m: int


@dataclass
class State:
    basis: np.ndarray
    status: np.ndarray
    factor: SuperLU
    x_basis: np.ndarray
    x_full: np.ndarray
    y: np.ndarray
    reduced_cost: np.ndarray
    primal_violation: np.ndarray
    dual_violation: np.ndarray
    merit: tuple[float, float, float, float]
    factor_seconds: float
    factor_nnz: int
    growth_proxy: float


@dataclass(frozen=True)
class Edge:
    row: int
    entering: int
    leaving_status: int
    source_kind: Literal["primal", "dual"]
    source_index: int
    source_rank: int
    severity: float
    ratio: float
    pivot_abs: float


def make_model() -> tuple[Model, dict[str, Any]]:
    original, prepared = prepare()
    raw = prepared["A_scipy"].tocsc().astype(np.float64)
    a, row_scale, col_scale = ruiz_scaled(raw)
    m, n = a.shape
    model = Model(
        original=original,
        prepared=prepared,
        a=a,
        a_aug=sparse.hstack([a, sparse.identity(m, format="csc")], format="csc"),
        b=prepared["b"] * row_scale,
        c=np.r_[prepared["c"] * col_scale, np.zeros(m)],
        lo=np.r_[prepared["lo"] / col_scale, np.zeros(m)],
        hi=np.r_[prepared["hi"] / col_scale, np.zeros(m)],
        row_scale=row_scale,
        col_scale=col_scale,
        n=n,
        m=m,
    )
    return model, prepared


def initial_status(model: Model, basis: np.ndarray) -> np.ndarray:
    """Place nonbasics only at true legal endpoints, never Big-M bounds."""
    matrix = model.a_aug[:, basis].tocsc()
    factor = splu(matrix)
    dual = factor.solve(model.c[basis], trans="T")
    reduced = model.c - model.a_aug.T @ dual
    basis_set = set(int(column) for column in basis)
    status = np.empty(model.n + model.m, dtype=np.int8)
    for j in range(model.n + model.m):
        if j in basis_set:
            status[j] = BASIC
            continue
        lo_finite = np.isfinite(model.lo[j])
        hi_finite = np.isfinite(model.hi[j])
        if lo_finite and hi_finite and abs(model.hi[j] - model.lo[j]) < 1e-14:
            status[j] = FIXED
        elif not lo_finite and not hi_finite:
            status[j] = FREE
        elif reduced[j] >= 0.0 and lo_finite:
            status[j] = AT_LO
        elif reduced[j] < 0.0 and hi_finite:
            status[j] = AT_HI
        elif lo_finite:
            status[j] = AT_LO
        else:
            status[j] = AT_HI
    return status


def endpoint_values(model: Model, status: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.zeros(model.n + model.m)
    basis_mask = np.zeros(model.n + model.m, dtype=bool)
    basis_mask[basis] = True
    for j in np.flatnonzero(~basis_mask):
        code = int(status[j])
        if code == AT_LO:
            if not np.isfinite(model.lo[j]):
                raise ValueError(f"column {j} has no legal lower endpoint")
            values[j] = model.lo[j]
        elif code == AT_HI:
            if not np.isfinite(model.hi[j]):
                raise ValueError(f"column {j} has no legal upper endpoint")
            values[j] = model.hi[j]
        elif code == FREE:
            if np.isfinite(model.lo[j]) or np.isfinite(model.hi[j]):
                raise ValueError(f"column {j} marked free despite finite endpoint")
            values[j] = 0.0
        elif code == FIXED:
            if not (np.isfinite(model.lo[j]) and np.isfinite(model.hi[j])):
                raise ValueError(f"column {j} marked fixed without finite bounds")
            values[j] = model.lo[j]
        else:
            raise ValueError(f"nonbasic column {j} has status {code}")
    return values


def make_state(model: Model, basis: np.ndarray, status: np.ndarray) -> State:
    if len(set(map(int, basis))) != model.m:
        raise ValueError("basis columns are not unique")
    status = status.copy()
    status[basis] = BASIC
    nonbasic_values = endpoint_values(model, status, basis)
    matrix = model.a_aug[:, basis].tocsc()
    started = time.perf_counter()
    factor = splu(matrix)
    factor_seconds = time.perf_counter() - started
    diagonal = np.abs(factor.U.diagonal())
    positive = diagonal[diagonal > 0.0]
    growth = float(diagonal.max() / positive.min()) if positive.size else math.inf
    if not np.isfinite(growth) or growth > 1e12:
        raise RuntimeError(f"factor growth proxy {growth:.3e}")

    rhs = model.b - model.a_aug @ nonbasic_values
    x_basis = factor.solve(rhs)
    y = factor.solve(model.c[basis], trans="T")
    reduced = np.asarray(model.c - model.a_aug.T @ y).ravel()
    reduced[basis] = 0.0
    x_full = nonbasic_values
    x_full[basis] = x_basis
    if not (
        np.all(np.isfinite(x_basis)) and np.all(np.isfinite(y)) and np.all(np.isfinite(reduced))
    ):
        raise RuntimeError("nonfinite reconstructed KKT state")

    lo_violation = np.maximum(model.lo[basis] - x_basis, 0.0)
    hi_violation = np.maximum(x_basis - model.hi[basis], 0.0)
    bound_scale = 1.0 + np.maximum(
        np.where(np.isfinite(model.lo[basis]), np.abs(model.lo[basis]), 0.0),
        np.where(np.isfinite(model.hi[basis]), np.abs(model.hi[basis]), 0.0),
    )
    primal = np.maximum(lo_violation, hi_violation) / bound_scale

    dual = np.zeros(model.n + model.m)
    c_scale = 1.0 + np.abs(model.c)
    for j in range(model.n + model.m):
        if status[j] == AT_LO:
            dual[j] = max(0.0, -reduced[j]) / c_scale[j]
        elif status[j] == AT_HI:
            dual[j] = max(0.0, reduced[j]) / c_scale[j]
        elif status[j] == FREE:
            dual[j] = abs(reduced[j]) / c_scale[j]

    artificial_positions = np.flatnonzero(basis >= model.n)
    artificial_mass = float(np.sum(np.abs(x_basis[artificial_positions])))
    merit = (
        artificial_mass,
        float(primal.max(initial=0.0)),
        float(dual.max(initial=0.0)),
        float(primal.sum() + dual.sum()),
    )
    return State(
        basis=basis.copy(),
        status=status,
        factor=factor,
        x_basis=x_basis,
        x_full=x_full,
        y=y,
        reduced_cost=reduced,
        primal_violation=primal,
        dual_violation=dual,
        merit=merit,
        factor_seconds=factor_seconds,
        factor_nnz=int(factor.L.nnz + factor.U.nnz),
        growth_proxy=growth,
    )


def violation_order(state: State) -> list[tuple[str, int, float]]:
    records = [
        ("primal", int(row), float(state.primal_violation[row]))
        for row in np.flatnonzero(state.primal_violation > KKT_TOL)
    ]
    records.extend(
        ("dual", int(column), float(state.dual_violation[column]))
        for column in np.flatnonzero(state.dual_violation > KKT_TOL)
    )
    return sorted(records, key=lambda item: (-item[2], item[0] != "primal", item[1]))


def leaving_endpoint(model: Model, column: int, movement: float) -> int | None:
    if movement < 0.0 and np.isfinite(model.lo[column]):
        return FIXED if abs(model.hi[column] - model.lo[column]) < 1e-14 else AT_LO
    if movement > 0.0 and np.isfinite(model.hi[column]):
        return FIXED if abs(model.hi[column] - model.lo[column]) < 1e-14 else AT_HI
    if not np.isfinite(model.lo[column]) and not np.isfinite(model.hi[column]):
        return FREE
    return None


def forward_step(current: float, endpoint: float, movement: float) -> float | None:
    """Return the nonnegative endpoint step, tolerating only FP-scale reversal."""
    if movement == 0.0:
        return None
    step = (endpoint - current) / movement
    tolerance = STEP_FP_MULTIPLIER * np.finfo(np.float64).eps * (1.0 + abs(step))
    if not np.isfinite(step) or step < -tolerance:
        return None
    return max(0.0, float(step))


def candidate_edges(model: Model, state: State) -> tuple[list[Edge], float]:
    """Build deterministic weighted edges from the top 64 KKT violations."""
    started = time.perf_counter()
    violations = violation_order(state)[: BATCH_LADDER[0]]
    basis_set = set(map(int, state.basis))
    rows = [index for kind, index, _ in violations if kind == "primal"]
    primal_alpha: dict[int, np.ndarray] = {}
    if rows:
        rhs = np.zeros((model.m, len(rows)))
        rhs[rows, np.arange(len(rows))] = 1.0
        rho = state.factor.solve(rhs, trans="T")
        alpha = np.asarray(model.a_aug.T @ rho)
        primal_alpha = {row: alpha[:, slot] for slot, row in enumerate(rows)}

    dual_columns = [index for kind, index, _ in violations if kind == "dual"]
    dual_q: dict[int, np.ndarray] = {}
    if dual_columns:
        block = model.a_aug[:, dual_columns].toarray()
        solved = state.factor.solve(block)
        dual_q = {column: solved[:, slot] for slot, column in enumerate(dual_columns)}

    edges: list[Edge] = []
    for rank, (kind, index, severity) in enumerate(violations):
        if kind == "primal":
            row = index
            basic_column = int(state.basis[row])
            below = state.x_basis[row] < model.lo[basic_column]
            desired = 1.0 if below else -1.0
            alpha = primal_alpha[row]
            local: list[Edge] = []
            for entering in range(model.n + model.m):
                if entering in basis_set or state.status[entering] in (FIXED, BASIC):
                    continue
                coefficient = float(alpha[entering])
                if abs(coefficient) <= PIVOT_TOL:
                    continue
                status = int(state.status[entering])
                if status == AT_LO:
                    direction = 1.0
                elif status == AT_HI:
                    direction = -1.0
                else:
                    direction = -desired * math.copysign(1.0, coefficient)
                if -coefficient * direction * desired <= PIVOT_TOL:
                    continue
                leave_status = leaving_endpoint(model, basic_column, -desired)
                if leave_status is None:
                    continue
                ratio = abs(float(state.reduced_cost[entering])) / abs(coefficient)
                local.append(
                    Edge(
                        row,
                        entering,
                        leave_status,
                        "primal",
                        row,
                        rank,
                        severity,
                        ratio,
                        abs(coefficient),
                    )
                )
            local.sort(key=lambda edge: (edge.ratio, -edge.pivot_abs, edge.entering))
            edges.extend(local[:8])
        else:
            entering = index
            status = int(state.status[entering])
            if status == AT_LO:
                direction = 1.0
            elif status == AT_HI:
                direction = -1.0
            elif status == FREE:
                direction = -math.copysign(1.0, state.reduced_cost[entering])
            else:
                continue
            local = []
            q = dual_q[entering]
            for row, coefficient in enumerate(q):
                if abs(coefficient) <= PIVOT_TOL:
                    continue
                basic_column = int(state.basis[row])
                movement = -float(coefficient) * direction
                leave_status = leaving_endpoint(model, basic_column, movement)
                if leave_status is None:
                    continue
                endpoint = (
                    model.lo[basic_column]
                    if leave_status in (AT_LO, FIXED)
                    else model.hi[basic_column]
                    if leave_status == AT_HI
                    else 0.0
                )
                ratio = forward_step(float(state.x_basis[row]), float(endpoint), movement)
                if ratio is None:
                    continue
                local.append(
                    Edge(
                        row,
                        entering,
                        leave_status,
                        "dual",
                        entering,
                        rank,
                        severity,
                        ratio,
                        abs(float(coefficient)),
                    )
                )
            local.sort(key=lambda edge: (edge.ratio, -edge.pivot_abs, edge.row))
            edges.extend(local[:8])

    edges.sort(
        key=lambda edge: (
            edge.source_rank,
            edge.ratio,
            -edge.pivot_abs,
            edge.row,
            edge.entering,
        )
    )
    return edges, time.perf_counter() - started


def rank_safe_matching(
    model: Model, state: State, edges: list[Edge], limit: int = 64
) -> tuple[list[Edge], float]:
    """Greedily form a matching whose exact basis-exchange minor stays full rank."""
    started = time.perf_counter()
    selected: list[Edge] = []
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    q_cache: dict[int, np.ndarray] = {}
    for edge in edges:
        if edge.row in used_rows or edge.entering in used_columns:
            continue
        if edge.entering not in q_cache:
            q_cache[edge.entering] = state.factor.solve(
                model.a_aug[:, edge.entering].toarray().ravel()
            )
        trial = [*selected, edge]
        rows = [candidate.row for candidate in trial]
        exchange = np.column_stack([q_cache[candidate.entering][rows] for candidate in trial])
        singular_values = np.linalg.svd(exchange, compute_uv=False)
        if singular_values[-1] <= PIVOT_TOL * max(1.0, singular_values[0]):
            continue
        selected.append(edge)
        used_rows.add(edge.row)
        used_columns.add(edge.entering)
        if len(selected) == limit:
            break
    return selected, time.perf_counter() - started


def criss_cross_edge(model: Model, state: State) -> Edge | None:
    """Least-index Bland-style criss-cross edge for the scalar fallback."""
    violations = violation_order(state)
    if not violations:
        return None
    # The criss-cross fallback is variable-index lexicographic, not magnitude
    # priced.  A basic violation is keyed by its global column index rather
    # than by the implementation's basis-row position.
    kind, index, severity = min(
        violations,
        key=lambda item: (
            int(state.basis[item[1]]) if item[0] == "primal" else item[1],
            item[0] != "primal",
        ),
    )
    basis_set = set(map(int, state.basis))
    if kind == "primal":
        row = index
        e = np.zeros(model.m)
        e[row] = 1.0
        alpha = np.asarray(model.a_aug.T @ state.factor.solve(e, trans="T")).ravel()
        basic_column = int(state.basis[row])
        desired = 1.0 if state.x_basis[row] < model.lo[basic_column] else -1.0
        leave_status = leaving_endpoint(model, basic_column, -desired)
        if leave_status is None:
            return None
        for entering in range(model.n + model.m):
            if entering in basis_set or state.status[entering] in (FIXED, BASIC):
                continue
            coefficient = float(alpha[entering])
            if abs(coefficient) <= PIVOT_TOL:
                continue
            status = int(state.status[entering])
            direction = (
                1.0
                if status == AT_LO
                else -1.0
                if status == AT_HI
                else -desired * math.copysign(1.0, coefficient)
            )
            if -coefficient * direction * desired > PIVOT_TOL:
                return Edge(
                    row,
                    entering,
                    leave_status,
                    "primal",
                    row,
                    0,
                    severity,
                    abs(float(state.reduced_cost[entering])) / abs(coefficient),
                    abs(coefficient),
                )
        return None

    entering = index
    q = state.factor.solve(model.a_aug[:, entering].toarray().ravel())
    status = int(state.status[entering])
    direction = (
        1.0
        if status == AT_LO
        else -1.0
        if status == AT_HI
        else -math.copysign(1.0, state.reduced_cost[entering])
    )
    for row in sorted(range(model.m), key=lambda position: (int(state.basis[position]), position)):
        coefficient = float(q[row])
        if abs(coefficient) <= PIVOT_TOL:
            continue
        leave_status = leaving_endpoint(model, int(state.basis[row]), -coefficient * direction)
        if leave_status is not None:
            basic_column = int(state.basis[row])
            endpoint = (
                model.lo[basic_column]
                if leave_status in (AT_LO, FIXED)
                else model.hi[basic_column]
                if leave_status == AT_HI
                else 0.0
            )
            step = forward_step(
                float(state.x_basis[row]),
                float(endpoint),
                -coefficient * direction,
            )
            if step is None:
                continue
            return Edge(
                row,
                entering,
                leave_status,
                "dual",
                entering,
                0,
                severity,
                step,
                abs(coefficient),
            )
    return None


def state_key(state: State) -> str:
    digest = hashlib.sha256()
    digest.update(state.basis.astype("<i8", copy=False).tobytes())
    digest.update(state.status.astype("i1", copy=False).tobytes())
    return digest.hexdigest()


def merit_json(merit: tuple[float, float, float, float]) -> list[float]:
    return list(map(float, merit))


def proposed_state(model: Model, state: State, edges: list[Edge]) -> State:
    basis = state.basis.copy()
    status = state.status.copy()
    for edge in edges:
        leaving = int(basis[edge.row])
        status[leaving] = edge.leaving_status
        status[edge.entering] = BASIC
        basis[edge.row] = edge.entering
    return make_state(model, basis, status)


def certificate(model: Model, state: State) -> dict[str, Any]:
    reduced_x = state.x_full[: model.n] * model.col_scale
    original_x = np.asarray(postsolve_x(reduced_x.tolist(), model.prepared["reduction"]))
    original = model.original
    objective = float(original["c"] @ original_x)
    equality = float(np.max(np.abs(original["A_scipy"] @ original_x - original["b"])))
    lower = float(np.max(np.maximum(original["lo"] - original_x, 0.0)))
    upper = float(np.max(np.maximum(original_x - original["hi"], 0.0)))
    kkt = state.merit[0] <= KKT_TOL and state.merit[1] <= KKT_TOL and state.merit[2] <= KKT_TOL
    passed = bool(kkt and equality <= EPS and max(lower, upper) <= EPS)
    return {
        "kkt_optimal": kkt,
        "objective_original": objective,
        "max_equality_residual_original": equality,
        "max_bound_violation_original": max(lower, upper),
        "eps": EPS,
        "passed": passed,
    }


def run_greenbea() -> dict[str, Any]:
    model, _ = make_model()
    basis, crash = crash_basis(model.a, model.lo[: model.n], model.hi[: model.n])

    # Basis-only one-pivot warm replay proves that the reconstructed crash is
    # the native iteration-zero basis without importing our endpoint policy.
    cold_one = model.prepared["matrix"].solve_eq_box_dual_simplex(
        model.prepared["c"].tolist(),
        model.prepared["b"].tolist(),
        model.prepared["lo"].tolist(),
        model.prepared["hi"].tolist(),
        max_iter=1,
        tol=1e-8,
        expand=1,
        leaving_rule=1,
        bfrt=0,
    )
    warm_one = model.prepared["matrix"].solve_eq_box_dual_simplex(
        model.prepared["c"].tolist(),
        model.prepared["b"].tolist(),
        model.prepared["lo"].tolist(),
        model.prepared["hi"].tolist(),
        max_iter=1,
        tol=1e-8,
        expand=1,
        leaving_rule=1,
        bfrt=0,
        initial_basis=basis.tolist(),
    )
    crash["native_plus_one_basis_match"] = bool(cold_one["basis"] == warm_one["basis"])
    crash["native_plus_one_status_match"] = bool(
        cold_one["bound_status"] == warm_one["bound_status"]
    )
    if not (crash["native_plus_one_basis_match"] and crash["native_plus_one_status_match"]):
        raise RuntimeError("Python crash does not reproduce native one-pivot basis and status")

    status = initial_status(model, basis)
    algorithm_started = time.perf_counter()
    state = make_state(model, basis, status)
    initial_merit = state.merit
    seen = {state_key(state)}
    proposals: list[dict[str, Any]] = []
    accepted_widths: list[int] = []
    attempted = 0
    accepted = 0
    cycle = False
    stop_reason = "unknown"
    policy_seconds = 0.0

    while accepted < MAX_ACCEPTED and attempted < MAX_ATTEMPTED:
        if state.merit[0] <= KKT_TOL and state.merit[1] <= KKT_TOL and state.merit[2] <= KKT_TOL:
            stop_reason = "kkt_optimal"
            break
        edges, build_seconds = candidate_edges(model, state)
        matching, matching_seconds = rank_safe_matching(model, state, edges)
        policy_seconds += build_seconds + matching_seconds
        accepted_this_cycle = False

        for requested in BATCH_LADDER:
            if attempted >= MAX_ATTEMPTED:
                break
            if requested == 1:
                scalar = criss_cross_edge(model, state)
                trial_edges = [] if scalar is None else [scalar]
            else:
                trial_edges = matching[: min(requested, len(matching))]
                if len(trial_edges) < 2:
                    continue
            if not trial_edges:
                continue
            width = len(trial_edges)
            attempted += 1
            before = state.merit
            record: dict[str, Any] = {
                "attempt": attempted,
                "requested_width": requested,
                "width": width,
                "merit_before": merit_json(before),
                "candidate_build_seconds": build_seconds,
                "rank_matching_seconds": matching_seconds,
                "edges": [
                    {
                        "row": edge.row,
                        "entering": edge.entering,
                        "leaving": int(state.basis[edge.row]),
                        "leaving_status": edge.leaving_status,
                        "source_kind": edge.source_kind,
                        "source_index": edge.source_index,
                        "severity": edge.severity,
                        "ratio": edge.ratio,
                        "pivot_abs": edge.pivot_abs,
                    }
                    for edge in trial_edges
                ],
            }
            try:
                trial = proposed_state(model, state, trial_edges)
            except (RuntimeError, ValueError) as error:
                record.update(
                    {
                        "accepted": False,
                        "reason": f"fail_closed:{type(error).__name__}:{error}",
                    }
                )
                proposals.append(record)
                continue
            key = state_key(trial)
            repeated = key in seen
            improves = trial.merit < before
            record.update(
                {
                    "factor_seconds": trial.factor_seconds,
                    "factor_nnz": trial.factor_nnz,
                    "factor_growth_proxy": trial.growth_proxy,
                    "merit_after": merit_json(trial.merit),
                    "repeated_state": repeated,
                    "accepted": bool(improves and not repeated),
                    "reason": (
                        "strict_lexicographic_decrease"
                        if improves and not repeated
                        else "repeated_state"
                        if repeated
                        else "no_strict_lexicographic_decrease"
                    ),
                }
            )
            proposals.append(record)
            if repeated:
                cycle = True
            if improves and not repeated:
                state = trial
                seen.add(key)
                accepted += 1
                accepted_widths.append(width)
                accepted_this_cycle = True
                break

        if not accepted_this_cycle:
            stop_reason = "no_selected_block_or_criss_cross_merit_decrease"
            break
    else:
        stop_reason = "accepted_cap" if accepted >= MAX_ACCEPTED else "attempt_cap"

    algorithm_seconds = time.perf_counter() - algorithm_started
    cert = certificate(model, state)
    median_width = statistics.median(accepted_widths) if accepted_widths else 0.0
    factor_count = 1 + sum("factor_seconds" in proposal for proposal in proposals)
    refactor_floor = factor_count * REFAC_REFERENCE_SECONDS
    projected_complete = algorithm_seconds if cert["passed"] else math.inf
    gates = {
        "accepted_count_le_256": accepted <= MAX_ACCEPTED,
        "attempted_count_le_384": attempted <= MAX_ATTEMPTED,
        "median_accepted_width_ge_18": median_width >= MEDIAN_WIDTH_GATE,
        "no_repeated_state": not cycle,
        "original_space_certificate": cert["passed"],
        "complete_projected_cost_le_gate": projected_complete <= PROBE_TARGET_SECONDS,
    }
    passed = all(gates.values())
    return {
        "fixture": "/tmp/lpsuite/lp_greenbea.mat",
        "shape": [model.m, model.n, int(model.a.nnz)],
        "verdict": "PASS_BLOCK_PDAS_CHARACTERIZATION"
        if passed
        else "KILL_BLOCK_PDAS_CHARACTERIZATION",
        "fixed_policy": {
            "batch_ladder": list(BATCH_LADDER),
            "max_accepted": MAX_ACCEPTED,
            "max_attempted": MAX_ATTEMPTED,
            "median_width_gate": MEDIAN_WIDTH_GATE,
            "kkt_tol": KKT_TOL,
            "eps": EPS,
            "merit": [
                "artificial_mass",
                "max_scaled_primal_violation",
                "max_scaled_dual_violation",
                "l1_scaled_KKT_violation",
            ],
            "scalar_fallback": "least-index Bland-style criss-cross",
            "dual_edge_step_rule": "(endpoint-x_B)/movement >= 0 within 64*FP-epsilon",
            "step_fp_multiplier": STEP_FP_MULTIPLIER,
        },
        "timing_contract": {
            "control_seconds": CONTROL_SECONDS,
            "board_target_seconds": BOARD_TARGET_SECONDS,
            "probe_target_seconds": PROBE_TARGET_SECONDS,
            "refactor_reference_seconds": REFAC_REFERENCE_SECONDS,
        },
        "crash": crash,
        "initial_merit": merit_json(initial_merit),
        "final_merit": merit_json(state.merit),
        "accepted": accepted,
        "attempted": attempted,
        "accepted_widths": accepted_widths,
        "median_accepted_width": median_width,
        "unique_states": len(seen),
        "cycle_observed": cycle,
        "stop_reason": stop_reason,
        "policy_seconds": policy_seconds,
        "algorithm_seconds": algorithm_seconds,
        "fresh_factor_count": factor_count,
        "refactor_only_projection_floor_seconds": refactor_floor,
        "projected_complete_cost_seconds": projected_complete,
        "certificate": cert,
        "gates": gates,
        "passed": passed,
        "proposals": proposals,
    }


def run() -> dict[str, Any]:
    greenbea = run_greenbea()
    # Sentinels are intentionally conditional: a failed primary gate cannot
    # earn broader fixture spend or retroactively rescue the mechanism.
    sentinels: list[dict[str, Any]] = []
    if greenbea["passed"]:
        raise NotImplementedError(
            "greenbea unexpectedly passed; implement the predeclared sentinel loader before ship review"
        )
    return {
        "verdict": greenbea["verdict"],
        "greenbea": greenbea,
        "sentinel_policy": {
            "fixtures": list(SENTINELS),
            "run_only_after_greenbea_pass": True,
            "executed": False,
        },
        "sentinels": sentinels,
    }


def main() -> None:
    os.environ.setdefault("LINPROGX_DS_EXPORT_BASIS", "1")
    os.environ.setdefault("LINPROGX_DS_WARM_START", "1")
    OUT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(OUT_DIR, 0o700)
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    RESULTS.write_text(payload)
    os.chmod(RESULTS, 0o600)
    print(RESULTS)
    print(hashlib.sha256(payload.encode()).hexdigest())
    summary = result["greenbea"]
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "verdict",
                    "accepted",
                    "attempted",
                    "median_accepted_width",
                    "stop_reason",
                    "algorithm_seconds",
                    "final_merit",
                    "gates",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
