"""Exact compatible and endpoint-dominated parallel-column presolve."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import linprogx.presolve as P
from linprogx.presolve import (
    PresolveResult,
    _DominatedColumn,
    _pack_dbls,
    _ParallelColumn,
    _result_from_c,
    postsolve_x,
    presolve_matrix,
)
from linprogx.sparse import SparseLPProblem, SparseSolver, csr_matrix, from_scipy_sparse
from linprogx.types import Status


def _native_parallel(
    rows: int,
    cols: int,
    indptr: list[int],
    indices: list[int],
    data: list[float],
    b: list[float],
    c: list[float],
    lo: list[float],
    hi: list[float],
):
    assert P._c_presolve_parallel_cols is not None
    raw = P._c_presolve_parallel_cols(
        csr_matrix(rows, cols, indptr, indices, data),
        _pack_dbls(b),
        _pack_dbls(c),
        _pack_dbls(lo),
        _pack_dbls(hi),
    )
    assert raw is not None
    return _result_from_c(raw)


@pytest.mark.parametrize("signs", itertools.product((-1.0, 1.0), repeat=6))
def test_parallel_merge_split_64_sign_cost_patterns(signs: tuple[float, ...]) -> None:
    row0, row1, q0, q1, gamma_sign, bound_family = signs
    base = (row0, 2.0 * row1)
    t = q1 / q0
    gamma = 3.0 * gamma_sign
    if bound_family > 0.0:
        lo = [-2.0, -1.0, -5.0]
        hi = [3.0, 4.0, 5.0]
    else:
        lo = [0.0, 0.0, -5.0]
        hi = [math.inf, math.inf, 5.0]
    target = [1.0, 2.0, 1.0]
    data = [q0 * base[0], q1 * base[0], 1.0, q0 * base[1], q1 * base[1], -1.0]
    indices = [0, 1, 2, 0, 1, 2]
    indptr = [0, 3, 6]
    b = [
        sum(data[p] * target[indices[p]] for p in range(indptr[i], indptr[i + 1])) for i in range(2)
    ]
    c = [q0 * gamma, q1 * gamma, 0.5]

    reduction = _native_parallel(2, 3, indptr, indices, data, b, c, lo, hi)
    assert (reduction.rows, reduction.cols, reduction._matrix.nnz) == (2, 2, 4)
    records = list(reduction._records)
    assert len(records) == 1
    assert isinstance(records[0], _ParallelColumn)
    assert records[0].scale == t
    assert reduction._reduction_counts["parallel_columns"] == 1

    merged_value = target[0] + t * target[1]
    reconstructed = postsolve_x([merged_value, target[2]], reduction)
    for row in range(2):
        lhs = sum(data[p] * reconstructed[indices[p]] for p in range(indptr[row], indptr[row + 1]))
        assert lhs == pytest.approx(b[row], abs=1e-12)
    assert all(
        lower <= value <= upper for value, lower, upper in zip(reconstructed, lo, hi, strict=True)
    )
    reduced_objective = sum(
        coef * value for coef, value in zip(reduction.c, [merged_value, target[2]], strict=True)
    )
    original_objective = sum(coef * value for coef, value in zip(c, reconstructed, strict=True))
    assert reduced_objective + reduction.objective_offset == pytest.approx(
        original_objective, abs=1e-12
    )


@pytest.mark.parametrize("mode", ("lower", "upper"))
@pytest.mark.parametrize("orient", (-1.0, 1.0))
def test_endpoint_dominance_preserves_an_optimal_split(mode: str, orient: float) -> None:
    # Two equal-cost retained members exercise the signed merge record; two
    # strictly worse groups exercise endpoint-fix records in the same class.
    if mode == "lower":
        gamma = [1.0, 1.0, 2.0, 3.0]
        intervals = [(0.0, math.inf), (1.0, 3.0), (2.0, 5.0), (4.0, 7.0)]
        aggregate = 15.0
        fixed_z = [2.0, 4.0]
    else:
        gamma = [1.0, 2.0, 3.0, 3.0]
        intervals = [(-7.0, -4.0), (-5.0, -2.0), (-3.0, -1.0), (-math.inf, 0.0)]
        aggregate = -15.0
        fixed_z = [-4.0, -2.0]

    scales = [orient, orient, orient, orient]
    lo = [L if orient > 0.0 else -U for L, U in intervals]
    hi = [U if orient > 0.0 else -L for L, U in intervals]
    c = [orient * value for value in gamma]
    data = scales.copy()
    reduction = _native_parallel(1, 4, [0, 4], [0, 1, 2, 3], data, [aggregate], c, lo, hi)

    assert (reduction.rows, reduction.cols, reduction._matrix.nnz) == (1, 1, 1)
    assert reduction._reduction_counts["parallel_columns"] == 1
    assert reduction._reduction_counts["dominated_columns"] == 2
    assert sum(isinstance(record, _ParallelColumn) for record in reduction._records) == 1
    assert sum(isinstance(record, _DominatedColumn) for record in reduction._records) == 2

    retained_aggregate = aggregate - sum(fixed_z)
    reconstructed = postsolve_x([orient * retained_aggregate], reduction)
    z = [orient * value for value in reconstructed]
    assert sum(z) == pytest.approx(aggregate, abs=1e-12)
    if mode == "lower":
        assert z[2:] == pytest.approx(fixed_z)
    else:
        assert z[:2] == pytest.approx(fixed_z)
    assert all(L <= value <= U for value, (L, U) in zip(z, intervals, strict=True))
    reduced_objective = reduction.c[0] * orient * retained_aggregate
    original_objective = sum(cost * value for cost, value in zip(c, reconstructed, strict=True))
    assert reduced_objective + reduction.objective_offset == pytest.approx(
        original_objective, abs=1e-12
    )


def test_cost_improving_recession_is_not_reduced() -> None:
    assert P._c_presolve_parallel_cols is not None
    raw = P._c_presolve_parallel_cols(
        csr_matrix(1, 2, [0, 2], [0, 1], [1.0, 1.0]),
        _pack_dbls([0.0]),
        _pack_dbls([1.0, 2.0]),
        _pack_dbls([0.0, -math.inf]),
        _pack_dbls([math.inf, 0.0]),
    )
    assert raw is None


def test_parallel_stage_default_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINPROGX_PRESOLVE_PARALLEL_COLS", raising=False)
    assert P._parallel_cols_enabled() is True  # shipped default ON post-certification
    monkeypatch.setenv("LINPROGX_PRESOLVE_PARALLEL_COLS", "1")
    assert P._parallel_cols_enabled() is True
    monkeypatch.setenv("LINPROGX_PRESOLVE_PARALLEL_COLS", "0")
    assert P._parallel_cols_enabled() is False


def test_parallel_stage_large_reduced_network_is_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = PresolveResult(
        rows=10_001,
        cols=1,
        indptr=[],
        indices=[],
        data=[],
        b=[0.0] * 10_001,
        c=[0.0],
        lo=[0.0],
        hi=[math.inf],
        objective_offset=0.0,
        removed_rows=0,
        removed_cols=0,
        _records=[],
        _active_cols=[0],
        _original_cols=1,
        _reduction_counts={**P._empty_reduction_counts(), "net_aggregations": 1},
        _matrix=csr_matrix(10_001, 1, [0] * 10_002, [], []),
    )

    def unexpected_call(*_args):
        raise AssertionError("large reduced network reached the native scan")

    monkeypatch.setattr(P, "_c_presolve_parallel_cols", unexpected_call)
    assert P._maybe_parallel_columns(result) is result


@pytest.mark.skipif(
    not Path("/tmp/lpsuite/lp_pds_10.mat").exists(), reason="pds_10 fixture unavailable"
)
def test_pds10_combined_shape_and_original_space_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = loadmat("/tmp/lpsuite/lp_pds_10.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    matrix = raw["A"].tocsr().astype(float)
    b = raw["b"].ravel().astype(float)
    c = aux["c"].ravel().astype(float)
    lo = aux["lo"].ravel().astype(float)
    hi = aux["hi"].ravel().astype(float)
    sparse_matrix = from_scipy_sparse(matrix)

    monkeypatch.setenv("LINPROGX_PRESOLVE_PARALLEL_COLS", "0")
    baseline = presolve_matrix(
        sparse_matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist(), algorithm="pdhg"
    )
    assert baseline is not None
    baseline_components = baseline._matrix.to_components()
    assert (baseline.rows, baseline.cols, baseline._matrix.nnz) == (4_955, 38_329, 84_923)

    monkeypatch.setenv("LINPROGX_PRESOLVE_PARALLEL_COLS", "1")
    reduced = presolve_matrix(
        sparse_matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist(), algorithm="pdhg"
    )
    assert reduced is not None
    assert (reduced.rows, reduced.cols, reduced._matrix.nnz) == (4_955, 34_454, 77_162)
    assert reduced._reduction_counts["parallel_columns"] == 176
    assert reduced._reduction_counts["dominated_columns"] == 3_699

    monkeypatch.setenv("LINPROGX_PRESOLVE_PARALLEL_COLS", "0")
    restored = presolve_matrix(
        sparse_matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist(), algorithm="pdhg"
    )
    assert restored is not None
    assert restored._matrix.to_components() == baseline_components
    assert restored.b == baseline.b
    assert restored.c == baseline.c
    assert restored.lo == baseline.lo
    assert restored.hi == baseline.hi

    monkeypatch.setenv("LINPROGX_PRESOLVE_PARALLEL_COLS", "1")
    bounds = [
        (
            None if not math.isfinite(lower) else float(lower),
            None if not math.isfinite(upper) else float(upper),
        )
        for lower, upper in zip(lo, hi, strict=True)
    ]
    result = SparseSolver(
        algorithm="auto", eps=2e-5, max_iterations=50_000, check_interval=50_000
    ).solve(
        SparseLPProblem(
            c.tolist(), A_eq=sparse_matrix, b_eq=b.tolist(), objective="min", bounds=bounds
        )
    )
    assert result.backend == "native-c-sparse-pdhg"
    assert result.solution.status == Status.OPTIMAL
    x = np.asarray(result.solution.x)
    assert float(np.max(np.abs(matrix @ x - b))) <= 2e-5
    assert float(max(0.0, np.max(lo - x), np.max(x - hi))) <= 2e-5
    assert result.solution.objective_value is not None
    assert abs(result.solution.objective_value - 26_727_094_976.0) / 26_727_094_976.0 <= 2e-5
