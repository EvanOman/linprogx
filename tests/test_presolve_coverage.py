"""Focused coverage for presolve policy gates and defensive helpers."""

from __future__ import annotations

import array
import math

import pytest

import linprogx.presolve as P
from linprogx.sparse import csr_matrix


class _MatrixStub:
    def __init__(self, nnz: int, shape: tuple[int, int] = (1, 1)) -> None:
        self.nnz = nnz
        self.shape = shape


def _result(
    *,
    rows: int = 10,
    cols: int = 1,
    nnz: int = 10,
    net_aggregations: int = 0,
) -> P.PresolveResult:
    counts = P._empty_reduction_counts()
    counts["net_aggregations"] = net_aggregations
    return P.PresolveResult(
        rows=rows,
        cols=cols,
        indptr=[],
        indices=[],
        data=[],
        b=[0.0] * rows,
        c=[0.0] * cols,
        lo=[-math.inf] * cols,
        hi=[math.inf] * cols,
        objective_offset=0.0,
        removed_rows=0,
        removed_cols=0,
        _records=[],
        _active_cols=list(range(cols)),
        _original_cols=cols,
        _reduction_counts=counts,
        _matrix=_MatrixStub(nnz, (rows, cols)),
    )


def test_lazy_record_list_rejects_unknown_tags_in_both_directions() -> None:
    records = P._LazyRecordList([(99,)])

    with pytest.raises(ValueError, match="unknown presolve record tag 99"):
        list(records)
    with pytest.raises(ValueError, match="unknown presolve record tag 99"):
        list(reversed(records))


def test_remap_record_rejects_unknown_record_type() -> None:
    with pytest.raises(ValueError, match="unknown presolve record type"):
        P._remap_record(object(), [0])


def test_aggressive_aggregation_applies_exchange_rate_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _result(rows=10, nnz=100)

    monkeypatch.setattr(P, "_maybe_aggregate", lambda *_args, **_kwargs: base)
    assert P.aggressive_aggregate_for_ds2(base) is None

    insufficient_row_gain = _result(rows=9, nnz=100)
    monkeypatch.setattr(P, "_maybe_aggregate", lambda *_args, **_kwargs: insufficient_row_gain)
    assert P.aggressive_aggregate_for_ds2(base) is None

    excessive_fill = _result(rows=8, nnz=106)
    monkeypatch.setattr(P, "_maybe_aggregate", lambda *_args, **_kwargs: excessive_fill)
    assert P.aggressive_aggregate_for_ds2(base) is None

    accepted = _result(rows=8, nnz=105)
    monkeypatch.setattr(P, "_maybe_aggregate", lambda *_args, **_kwargs: accepted)
    assert P.aggressive_aggregate_for_ds2(base) is accepted


def test_netaggregation_stage_rejects_unavailable_empty_and_too_small_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _result(rows=10_000, nnz=100_000)

    monkeypatch.setattr(P, "_c_presolve_netagg", None)
    assert P._maybe_netaggregate(base, 5) is base

    monkeypatch.setattr(P, "_c_presolve_netagg", lambda *_args: None)
    assert P._maybe_netaggregate(base, 5) is base

    too_small_a_gain = _result(rows=9_000, nnz=90_001)
    monkeypatch.setattr(P, "_c_presolve_netagg", lambda *_args: object())
    monkeypatch.setattr(P, "_result_from_c", lambda _raw: too_small_a_gain)
    assert P._maybe_netaggregate(base, 5) is base


def test_parallel_column_stage_rejects_unavailable_empty_and_too_small_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _result(rows=100, nnz=100, net_aggregations=1)

    monkeypatch.setattr(P, "_c_presolve_parallel_cols", None)
    assert P._maybe_parallel_columns(base) is base

    monkeypatch.setattr(P, "_c_presolve_parallel_cols", lambda *_args: None)
    assert P._maybe_parallel_columns(base) is base

    too_small_a_gain = _result(rows=100, nnz=93)
    monkeypatch.setattr(P, "_c_presolve_parallel_cols", lambda *_args: object())
    monkeypatch.setattr(P, "_result_from_c", lambda _raw: too_small_a_gain)
    assert P._maybe_parallel_columns(base) is base


def test_legacy_native_result_infers_record_counts() -> None:
    matrix = csr_matrix(1, 1, [0, 0], [], [])
    records = [
        (0, 1, 2.0),
        (1, 2, 3, 1.0, -1.0, 4.0),
        (7, 4, 5, 2.0, 0.0, 1.0, -1.0, 3.0),
        (8, 6, 0.0),
    ]
    active_cols = array.array(P._INT_TC, [0]).tobytes()
    raw = (
        matrix,
        P._pack_dbls([0.0]),
        P._pack_dbls([1.0]),
        P._pack_dbls([-math.inf]),
        P._pack_dbls([math.inf]),
        3.0,
        2,
        6,
        records,
        active_cols,
        7,
    )

    result = P._result_from_c(raw)

    assert result._reduction_counts["singleton_rows"] == 1
    assert result._reduction_counts["doubletons"] == 1
    assert result._reduction_counts["parallel_columns"] == 1
    assert result._reduction_counts["dominated_columns"] == 1
    assert result.objective_offset == 3.0
    assert result._active_cols == [0]


def test_bound_helpers_cover_free_and_one_sided_columns() -> None:
    assert P._column_bounds_are_redundant(
        0,
        1.0,
        0.0,
        ((1, 1.0),),
        [-math.inf, 0.0],
        [math.inf, 1.0],
    )
    assert P._row_activity({0: -2.0}, [-math.inf], [math.inf]) == (
        0.0,
        0.0,
        1,
        1,
        0,
        2.0,
    )
    assert P._choose_empty_column_value(0, [0.0], [2.0], [math.inf]) == 2.0
    assert P._choose_empty_column_value(0, [0.0], [-math.inf], [-2.0]) == -2.0


def test_aggregation_only_keeps_unbounded_improving_empty_column() -> None:
    result = P._presolve_eq_box_python(
        0,
        1,
        [0],
        [],
        [],
        [],
        [1.0],
        [-math.inf],
        [math.inf],
        agg=True,
        agg_only=True,
    )

    assert result is None


@pytest.mark.parametrize(
    ("cols", "indices", "data"),
    [
        (1, [0], [1e-13]),
        (2, [0, 1], [1e-13, 2e-13]),
        (2, [0, 1], [1.0, 1e-13]),
    ],
)
def test_tiny_pivots_are_not_eliminated(
    monkeypatch: pytest.MonkeyPatch,
    cols: int,
    indices: list[int],
    data: list[float],
) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_V2", "0")

    result = P._presolve_eq_box_python(
        1,
        cols,
        [0, len(data)],
        indices,
        data,
        [0.0],
        [0.0] * cols,
        [-math.inf] * cols,
        [math.inf] * cols,
    )

    assert result is None


def test_tiny_column_singleton_pivot_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_V2", "1")

    result = P._presolve_eq_box_python(
        1,
        3,
        [0, 3],
        [0, 1, 2],
        [1e-13, 1.0, 1.0],
        [0.0],
        [0.0, 0.0, 0.0],
        [-math.inf, -math.inf, -math.inf],
        [math.inf, math.inf, math.inf],
    )

    assert result is not None
    assert result._reduction_counts["column_singletons"] == 1


def test_matrix_api_falls_back_to_python_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ComponentMatrix:
        shape = (1, 1)

        def to_components(self) -> tuple[list[int], list[int], list[float]]:
            return [0, 1], [0], [1.0]

    monkeypatch.setattr(P, "_c_presolve", None)

    result = P.presolve_matrix(
        ComponentMatrix(),
        [2.0],
        [3.0],
        [0.0],
        [5.0],
    )

    assert result is not None
    assert result.objective_offset == 6.0
    assert P.postsolve_x([], result) == [2.0]


def test_duplicate_column_postsolve_clamps_roundoff_below_removed_bound() -> None:
    reduction = P.PresolveResult(
        rows=0,
        cols=1,
        indptr=[0],
        indices=[],
        data=[],
        b=[],
        c=[0.0],
        lo=[0.0],
        hi=[20.0],
        objective_offset=0.0,
        removed_rows=0,
        removed_cols=1,
        _records=[P._DuplicateColumn(1, 0, 1.0, 10.0, 5.0, 10.0)],
        _active_cols=[0],
        _original_cols=2,
    )

    assert P.postsolve_x([5.0], reduction) == [4.0, 1.0]


def test_w2b_force_agg_hook_is_off_by_default_and_bypasses_the_exchange_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measurement hook: LINPROGX_W2B_FORCE_AGG admits a gate-declined candidate."""
    base = _result(rows=10, nnz=100)
    excessive_fill = _result(rows=8, nnz=106)
    monkeypatch.setattr(P, "_maybe_aggregate", lambda *_args, **_kwargs: excessive_fill)

    monkeypatch.delenv("LINPROGX_W2B_FORCE_AGG", raising=False)
    assert P.aggressive_aggregate_for_ds2(base) is None

    monkeypatch.setenv("LINPROGX_W2B_FORCE_AGG", "1")
    assert P.aggressive_aggregate_for_ds2(base) is excessive_fill
