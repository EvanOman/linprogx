"""Clean-room tests for multi-row implied-bound network aggregation."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
from scipy.io import loadmat

import linprogx.presolve as P
from linprogx.presolve import (
    _compose_reductions,
    _empty_reduction_counts,
    _NetAggregation,
    _pack_dbls,
    _remap_record,
    _result_from_c,
    postsolve_x,
    presolve_matrix,
)
from linprogx.sparse import SparseLPProblem, SparseSolver, csr_matrix, from_scipy_sparse
from linprogx.types import Status


def _signed_interval_row(
    a: float,
    d: float,
    target_x: float,
    target_y: float,
    implied_lo: float,
    implied_hi: float,
) -> tuple[float, float, float]:
    """Return rhs and a y-box that maps to the requested x interval."""
    rhs = a * target_x + d * target_y
    alpha = -d / a
    beta = rhs / a
    if alpha > 0.0:
        y_lo = -math.inf if not math.isfinite(implied_lo) else (implied_lo - beta) / alpha
        y_hi = math.inf if not math.isfinite(implied_hi) else (implied_hi - beta) / alpha
    else:
        y_lo = -math.inf if not math.isfinite(implied_hi) else (implied_hi - beta) / alpha
        y_hi = math.inf if not math.isfinite(implied_lo) else (implied_lo - beta) / alpha
    return rhs, y_lo, y_hi


@pytest.mark.parametrize("signs", itertools.product((-1.0, 1.0), repeat=6))
def test_netagg_exact_postsolve_on_64_sign_patterns(signs: tuple[float, ...]) -> None:
    """Different incident rows prove the lower and upper bound on x0."""
    if P._c_presolve_netagg is None:
        pytest.skip("native net aggregation extension unavailable")

    target = [1.0, 2.0, 3.0, -1.0]
    intervals = [(0.0, math.inf), (-math.inf, 4.0), (-10.0, 10.0)]
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    b: list[float] = []
    lo = [0.0, 0.0, 0.0, 0.0]
    hi = [4.0, 0.0, 0.0, 0.0]
    for row, (implied_lo, implied_hi) in enumerate(intervals):
        a, d = signs[2 * row : 2 * row + 2]
        rhs, private_lo, private_hi = _signed_interval_row(
            a, d, target[0], target[row + 1], implied_lo, implied_hi
        )
        indices.extend((0, row + 1))
        data.extend((a, d))
        indptr.append(len(indices))
        b.append(rhs)
        lo[row + 1] = private_lo
        hi[row + 1] = private_hi

    c = [2.0, -3.0, 5.0, 7.0]
    matrix = csr_matrix(3, 4, indptr, indices, data)
    raw = P._c_presolve_netagg(
        matrix,
        _pack_dbls(b),
        _pack_dbls(c),
        _pack_dbls(lo),
        _pack_dbls(hi),
        5,
        -1,
        matrix.nnz,
    )
    assert raw is not None
    reduction = _result_from_c(raw)
    assert (reduction.rows, reduction.cols, reduction._matrix.nnz) == (2, 3, 4)
    records = list(reduction._records)
    assert len(records) == 1
    assert isinstance(records[0], _NetAggregation)

    reduced_target = [target[j] for j in reduction._active_cols]
    reconstructed = postsolve_x(reduced_target, reduction)
    assert reconstructed == pytest.approx(target, abs=1e-12)
    for row in range(3):
        lhs = sum(
            data[offset] * reconstructed[indices[offset]]
            for offset in range(indptr[row], indptr[row + 1])
        )
        assert lhs == pytest.approx(b[row], abs=1e-12)
    assert (
        max(
            max(lower - x, 0.0, x - upper)
            for x, lower, upper in zip(reconstructed, lo, hi, strict=True)
        )
        <= 1e-12
    )
    reduced_objective = sum(
        coef * value for coef, value in zip(reduction.c, reduced_target, strict=True)
    )
    original_objective = sum(coef * value for coef, value in zip(c, reconstructed, strict=True))
    assert reduced_objective + reduction.objective_offset == pytest.approx(
        original_objective, abs=1e-12
    )


def test_netagg_record_remaps_and_composes() -> None:
    record = _NetAggregation(1, -2.0, 7.0, ((0, 3.0), (2, -4.0)))
    assert _remap_record(record, [10, 20, 30]) == _NetAggregation(
        20, -2.0, 7.0, ((10, 3.0), (30, -4.0))
    )

    first = P.PresolveResult(
        rows=1,
        cols=3,
        indptr=[],
        indices=[],
        data=[],
        b=[0.0],
        c=[0.0, 0.0, 0.0],
        lo=[-math.inf] * 3,
        hi=[math.inf] * 3,
        objective_offset=0.0,
        removed_rows=0,
        removed_cols=1,
        _records=[],
        _active_cols=[0, 2, 3],
        _original_cols=4,
        _reduction_counts=_empty_reduction_counts(),
        _matrix=csr_matrix(1, 3, [0, 0], [], []),
    )
    second = P.PresolveResult(
        rows=0,
        cols=2,
        indptr=[],
        indices=[],
        data=[],
        b=[],
        c=[0.0, 0.0],
        lo=[-math.inf] * 2,
        hi=[math.inf] * 2,
        objective_offset=0.0,
        removed_rows=1,
        removed_cols=1,
        _records=[_NetAggregation(1, 1.0, 5.0, ((0, -1.0),))],
        _active_cols=[0, 2],
        _original_cols=3,
        _reduction_counts={**_empty_reduction_counts(), "net_aggregations": 1},
        _matrix=csr_matrix(0, 2, [0], [], []),
    )
    composed = _compose_reductions(first, second)
    assert composed._active_cols == [0, 3]
    assert composed._reduction_counts["net_aggregations"] == 1
    assert postsolve_x([2.0, 9.0], composed) == pytest.approx([2.0, 0.0, 7.0, 9.0])


def test_netagg_default_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Shipped default ON after certification gates passed; the structural
    # size gates keep the pass inert off pds-scale PDHG problems.
    monkeypatch.delenv("LINPROGX_PRESOLVE_NETAGG", raising=False)
    assert P._netagg_enabled() is True
    monkeypatch.setenv("LINPROGX_PRESOLVE_NETAGG", "1")
    assert P._netagg_enabled() is True
    monkeypatch.setenv("LINPROGX_PRESOLVE_NETAGG", "0")
    assert P._netagg_enabled() is False


@pytest.mark.skipif(
    not Path("/tmp/lpsuite/lp_pds_10.mat").exists(), reason="pds_10 fixture unavailable"
)
def test_pds10_netagg_public_route_and_original_space_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    raw = loadmat("/tmp/lpsuite/lp_pds_10.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    matrix = raw["A"].tocsr().astype(float)
    b = raw["b"].ravel().astype(float)
    c = aux["c"].ravel().astype(float)
    lo = aux["lo"].ravel().astype(float)
    hi = aux["hi"].ravel().astype(float)
    sparse_matrix = from_scipy_sparse(matrix)

    monkeypatch.setenv("LINPROGX_PRESOLVE_NETAGG", "0")
    baseline = presolve_matrix(
        sparse_matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist(), algorithm="pdhg"
    )
    assert baseline is not None
    assert (baseline.rows, baseline.cols, baseline._matrix.nnz) == (14_438, 47_812, 103_230)

    monkeypatch.setenv("LINPROGX_PRESOLVE_NETAGG", "1")
    reduced = presolve_matrix(
        sparse_matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist(), algorithm="pdhg"
    )
    assert reduced is not None
    assert (reduced.rows, reduced.cols, reduced._matrix.nnz) == (4_955, 38_329, 84_923)
    assert reduced._reduction_counts["net_aggregations"] == 9_483

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
    assert result.solution.iterations == 7_104
    x = np.asarray(result.solution.x)
    assert float(np.max(np.abs(matrix @ x - b))) <= 2e-5
    assert float(max(0.0, np.max(lo - x), np.max(x - hi))) <= 2e-5
    assert result.solution.objective_value is not None
    assert abs(result.solution.objective_value - 26_727_094_976.0) / 26_727_094_976.0 <= 2e-5
