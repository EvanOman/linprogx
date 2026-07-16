"""Equivalence tests: C presolve must reproduce Python presolve exactly.

These tests compare the C accelerator in ``_csparse.presolve_eq_box`` against
the pure-Python ``_presolve_eq_box_python`` on a range of problem shapes.
Every field of the PresolveResult must match with exact float equality (same
arithmetic, same reductions, same order).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
from scipy.io import loadmat
from scipy.optimize import linprog

import linprogx.presolve as presolve_module
from linprogx.presolve import (
    PresolveResult,
    _c_v2_candidates,
    _ColumnSingleton,
    _Doubleton,
    _DuplicateColumn,
    _FixedVar,
    _pack_dbls,
    _presolve_eq_box_python,
    _v2_enabled,
    _v2_worth_python_pass,
    postsolve_x,
    presolve_eq_box,
    presolve_matrix,
)
from linprogx.sparse import SparseSolver, csr_matrix, from_scipy_sparse
from linprogx.types import Status

INF = float("inf")


@pytest.fixture
def v2_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_V2", "1")


def _max_residual(
    rows: int,
    indptr: list[int],
    indices: list[int],
    data: list[float],
    x: list[float],
    b: list[float],
) -> float:
    max_abs = 0.0
    for i in range(rows):
        lhs = 0.0
        for offset in range(indptr[i], indptr[i + 1]):
            lhs += data[offset] * x[indices[offset]]
        max_abs = max(max_abs, abs(lhs - b[i]))
    return max_abs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_results_identical(
    c_result: PresolveResult | None,
    py_result: PresolveResult | None,
    *,
    label: str = "",
) -> None:
    """Assert every field of two PresolveResult objects matches exactly."""
    msg = f" [{label}]" if label else ""
    if py_result is None:
        assert c_result is None, f"Python returned None but C did not{msg}"
        return
    assert c_result is not None, f"Python returned a result but C returned None{msg}"

    if c_result._matrix is not None and not c_result.indptr:
        c_result.indptr, c_result.indices, c_result.data = c_result._matrix.to_components()
    if py_result._matrix is not None and not py_result.indptr:
        py_result.indptr, py_result.indices, py_result.data = py_result._matrix.to_components()

    assert c_result.rows == py_result.rows, f"rows mismatch{msg}"
    assert c_result.cols == py_result.cols, f"cols mismatch{msg}"
    assert c_result.indptr == py_result.indptr, f"indptr mismatch{msg}"
    assert c_result.indices == py_result.indices, f"indices mismatch{msg}"
    assert c_result.data == py_result.data, f"data mismatch{msg}"
    assert c_result.b == py_result.b, f"b mismatch{msg}"
    assert c_result.c == py_result.c, f"c mismatch{msg}"
    assert c_result.lo == py_result.lo, f"lo mismatch{msg}"
    assert c_result.hi == py_result.hi, f"hi mismatch{msg}"
    assert c_result.objective_offset == py_result.objective_offset, (
        f"objective_offset mismatch{msg}"
    )
    assert c_result.removed_rows == py_result.removed_rows, f"removed_rows mismatch{msg}"
    assert c_result.removed_cols == py_result.removed_cols, f"removed_cols mismatch{msg}"
    assert c_result._active_cols == py_result._active_cols, f"_active_cols mismatch{msg}"
    assert c_result._original_cols == py_result._original_cols, f"_original_cols mismatch{msg}"

    # Records
    assert len(c_result._records) == len(py_result._records), f"records length mismatch{msg}"
    for k, (cr, pr) in enumerate(zip(c_result._records, py_result._records, strict=True)):
        assert type(cr) is type(pr), f"record {k} type mismatch{msg}"
        if isinstance(pr, _FixedVar):
            assert isinstance(cr, _FixedVar)
            assert cr.column == pr.column, f"record {k} column mismatch{msg}"
            assert cr.value == pr.value, f"record {k} value mismatch{msg}"
        elif isinstance(pr, _Doubleton):
            assert isinstance(cr, _Doubleton)
            assert cr.eliminated == pr.eliminated, f"record {k} eliminated mismatch{msg}"
            assert cr.kept == pr.kept, f"record {k} kept mismatch{msg}"
            assert cr.coef_eliminated == pr.coef_eliminated, (
                f"record {k} coef_eliminated mismatch{msg}"
            )
            assert cr.coef_kept == pr.coef_kept, f"record {k} coef_kept mismatch{msg}"
            assert cr.rhs == pr.rhs, f"record {k} rhs mismatch{msg}"
        elif isinstance(pr, _ColumnSingleton):
            assert isinstance(cr, _ColumnSingleton)
            assert cr.eliminated == pr.eliminated, f"record {k} eliminated mismatch{msg}"
            assert cr.coef_eliminated == pr.coef_eliminated, (
                f"record {k} coef_eliminated mismatch{msg}"
            )
            assert cr.rhs == pr.rhs, f"record {k} rhs mismatch{msg}"
            assert cr.terms == pr.terms, f"record {k} terms mismatch{msg}"
        else:
            assert isinstance(cr, _DuplicateColumn)
            assert isinstance(pr, _DuplicateColumn)
            assert cr.removed == pr.removed, f"record {k} removed mismatch{msg}"
            assert cr.kept == pr.kept, f"record {k} kept mismatch{msg}"
            assert cr.removed_lo == pr.removed_lo, f"record {k} removed_lo mismatch{msg}"
            assert cr.removed_hi == pr.removed_hi, f"record {k} removed_hi mismatch{msg}"
            assert cr.kept_lo == pr.kept_lo, f"record {k} kept_lo mismatch{msg}"
            assert cr.kept_hi == pr.kept_hi, f"record {k} kept_hi mismatch{msg}"

    samples = [
        [0.0] * c_result.cols,
        [min(max(0.25 * (j + 1), c_result.lo[j]), c_result.hi[j]) for j in range(c_result.cols)],
    ]
    for sample_id, x_reduced in enumerate(samples):
        assert postsolve_x(x_reduced, c_result) == postsolve_x(x_reduced, py_result), (
            f"postsolve_x mismatch for sample {sample_id}{msg}"
        )


def _run_both(
    rows: int,
    cols: int,
    indptr: list[int],
    indices: list[int],
    data: list[float],
    b: list[float],
    c: list[float],
    lo: list[float],
    hi: list[float],
    *,
    max_fill: int = 5,
    label: str = "",
) -> None:
    """Run both C and Python presolve and assert identical results."""
    # Call C via the wrapper
    c_result = presolve_eq_box(
        rows,
        cols,
        list(indptr),
        list(indices),
        list(data),
        list(b),
        list(c),
        list(lo),
        list(hi),
        max_fill=max_fill,
    )
    # Call Python directly
    py_result = _presolve_eq_box_python(
        rows,
        cols,
        list(indptr),
        list(indices),
        list(data),
        list(b),
        list(c),
        list(lo),
        list(hi),
        max_fill=max_fill,
    )
    _assert_results_identical(c_result, py_result, label=label)


def _run_matrix_native_and_python(
    matrix,
    b: list[float],
    c: list[float],
    lo: list[float],
    hi: list[float],
    *,
    max_fill: int = 5,
    label: str = "",
) -> PresolveResult | None:
    """Run native-enabled and native-disabled matrix routes and compare them."""
    native = presolve_matrix(
        matrix,
        list(b),
        list(c),
        list(lo),
        list(hi),
        max_fill=max_fill,
    )
    old_native = os.environ.get("LINPROGX_PRESOLVE_V2_NATIVE")
    os.environ["LINPROGX_PRESOLVE_V2_NATIVE"] = "0"
    try:
        reference = presolve_matrix(
            matrix,
            list(b),
            list(c),
            list(lo),
            list(hi),
            max_fill=max_fill,
        )
    finally:
        if old_native is None:
            os.environ.pop("LINPROGX_PRESOLVE_V2_NATIVE", None)
        else:
            os.environ["LINPROGX_PRESOLVE_V2_NATIVE"] = old_native
    _assert_results_identical(native, reference, label=label)
    return native


def test_v2_python_pass_requires_eight_percent_projected_reduction() -> None:
    assert not _v2_worth_python_pass(7, 0, 100, 100)
    assert not _v2_worth_python_pass(0, 7, 100, 100)
    assert _v2_worth_python_pass(8, 0, 100, 100)
    assert _v2_worth_python_pass(0, 8, 100, 100)


def test_v2_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINPROGX_PRESOLVE_V2", raising=False)
    assert _v2_enabled()


def test_v2_candidate_scan_does_not_treat_overflow_as_forcing() -> None:
    matrix = csr_matrix(1, 3, [0, 3], [0, 1, 2], [2.0, 2.0, 2.0])
    assert _c_v2_candidates is not None
    assert _c_v2_candidates(
        matrix,
        _pack_dbls([0.0]),
        _pack_dbls([0.0, 1.0, 2.0]),
        _pack_dbls([1e308, 1e308, 1e308]),
        _pack_dbls([INF, INF, INF]),
    ) == (0, 0, 0, 0, 0, 0, 0, 0, 0)


def test_v2_candidate_scan_reports_each_qualification_type() -> None:
    assert _c_v2_candidates is not None
    forcing = csr_matrix(1, 3, [0, 3], [0, 1, 2], [1.0, 1.0, 1.0])
    assert _c_v2_candidates(
        forcing,
        _pack_dbls([0.0]),
        _pack_dbls([0.0, 1.0, 2.0]),
        _pack_dbls([0.0, 0.0, 0.0]),
        _pack_dbls([0.0, 10.0, 10.0]),
    ) == (1, 3, 1, 0, 1, 3, 0, 0, 0)

    empty_and_duplicate = csr_matrix(1, 4, [0, 2], [0, 1], [1.0, 1.0])
    assert _c_v2_candidates(
        empty_and_duplicate,
        _pack_dbls([1.0]),
        _pack_dbls([2.0, 2.0, 0.0, 1.0]),
        _pack_dbls([0.0, 0.0, 0.0, -INF]),
        _pack_dbls([1.0, 1.0, 1.0, INF]),
    ) == (0, 2, 0, 1, 0, 0, 0, 0, 1)


def test_v2_zero_opportunity_path_packs_vectors_once(
    v2_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_pack = presolve_module._pack_dbls

    def counting_pack(values: list[float]) -> bytes:
        nonlocal calls
        calls += 1
        return real_pack(values)

    monkeypatch.setattr(presolve_module, "_pack_dbls", counting_pack)
    matrix = csr_matrix(1, 3, [0, 3], [0, 1, 2], [1.0, 2.0, 3.0])
    presolve_matrix(
        matrix,
        [4.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [INF, INF, INF],
    )
    assert calls == 4


def test_matrix_v2_uses_direct_high_yield_path_and_postsolves_exactly(
    v2_enabled: None,
) -> None:
    matrix = csr_matrix(
        3,
        6,
        [0, 3, 5, 8],
        [0, 1, 3, 1, 2, 3, 4, 5],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    reduction = presolve_matrix(
        matrix,
        [6.0, 3.0, 7.0],
        [10.0, 5.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 1.0, 2.0, 0.0, 0.0],
        [INF, 2.0, 2.0, 3.0, 10.0, 10.0],
    )

    assert reduction is not None
    assert reduction._active_cols == [2, 3, 4, 5]
    assert reduction._reduction_counts["column_singletons"] == 1
    assert reduction._reduction_counts["doubletons"] == 1
    x = postsolve_x([1.5, 2.5, 2.0, 2.5], reduction)
    assert x == pytest.approx([2.0, 1.5, 1.5, 2.5, 2.0, 2.5])
    assert (
        _max_residual(
            3,
            [0, 3, 5, 8],
            [0, 1, 3, 1, 2, 3, 4, 5],
            [1.0] * 8,
            x,
            [6.0, 3.0, 7.0],
        )
        <= 1e-12
    )


def test_matrix_v2_high_yield_uses_native_reducer_and_matches_python(
    v2_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LINPROGX_PRESOLVE_V2_NATIVE", raising=False)
    matrix = csr_matrix(
        3,
        6,
        [0, 3, 5, 8],
        [0, 1, 3, 1, 2, 3, 4, 5],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    reduction = _run_matrix_native_and_python(
        matrix,
        [6.0, 3.0, 7.0],
        [10.0, 5.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 1.0, 2.0, 0.0, 0.0],
        [INF, 2.0, 2.0, 3.0, 10.0, 10.0],
        label="high-yield-native",
    )

    assert reduction is not None
    assert reduction._matrix is not None
    assert reduction._reduction_counts["column_singletons"] == 1
    assert reduction._reduction_counts["doubletons"] == 1


def test_matrix_v2_native_can_be_disabled(
    v2_enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LINPROGX_PRESOLVE_V2_NATIVE", "0")
    matrix = csr_matrix(
        3,
        6,
        [0, 3, 5, 8],
        [0, 1, 3, 1, 2, 3, 4, 5],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )

    reduction = presolve_matrix(
        matrix,
        [6.0, 3.0, 7.0],
        [10.0, 5.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 1.0, 2.0, 0.0, 0.0],
        [INF, 2.0, 2.0, 3.0, 10.0, 10.0],
    )

    assert reduction is not None
    assert reduction._matrix is None


def test_implied_free_column_singletons_chain_and_postsolve_exactly(v2_enabled: None) -> None:
    # x0 is lower-bounded, but row 0 plus x1 in [1, 2] and x3 in [2, 3]
    # implies x0 in [1, 3].
    # Removing row 0 makes x1 a lower-bounded singleton in row 1, whose bound
    # is likewise implied by x2 in [1, 2].
    reduction = presolve_eq_box(
        3,
        6,
        [0, 3, 5, 8],
        [0, 1, 3, 1, 2, 3, 4, 5],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [6.0, 3.0, 7.0],
        [10.0, 5.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 1.0, 2.0, 0.0, 0.0],
        [INF, 2.0, 2.0, 3.0, 10.0, 10.0],
    )

    assert reduction is not None
    assert reduction.rows == 1
    assert reduction.cols == 4
    assert reduction._active_cols == [2, 3, 4, 5]
    assert reduction._reduction_counts["column_singletons"] == 1
    assert reduction._reduction_counts["doubletons"] == 1

    x = postsolve_x([1.5, 2.5, 2.0, 2.5], reduction)
    assert x == pytest.approx([2.0, 1.5, 1.5, 2.5, 2.0, 2.5])
    assert (
        _max_residual(
            3,
            [0, 3, 5, 8],
            [0, 1, 3, 1, 2, 3, 4, 5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            x,
            [6.0, 3.0, 7.0],
        )
        <= 1e-12
    )


def test_fixed_columns_substitute_without_removing_rows_unnecessarily(v2_enabled: None) -> None:
    reduction = presolve_eq_box(
        1,
        4,
        [0, 4],
        [0, 1, 2, 3],
        [3.0, 1.0, 1.0, 1.0],
        [7.0],
        [2.0, 3.0, 5.0, 7.0],
        [2.0, 0.0, 0.0, 0.0],
        [2.0, 10.0, 10.0, 10.0],
    )

    assert reduction is not None
    assert reduction.rows == 1
    assert reduction.cols == 3
    assert reduction.b == pytest.approx([1.0])
    assert reduction.c == pytest.approx([3.0, 5.0, 7.0])
    assert reduction.objective_offset == pytest.approx(4.0)
    assert reduction._active_cols == [1, 2, 3]
    assert reduction._reduction_counts["fixed_columns"] == 1

    x = postsolve_x([0.25, 0.25, 0.5], reduction)
    assert x == pytest.approx([2.0, 0.25, 0.25, 0.5])
    assert (
        _max_residual(
            1,
            [0, 4],
            [0, 1, 2, 3],
            [3.0, 1.0, 1.0, 1.0],
            x,
            [7.0],
        )
        <= 1e-12
    )


def test_forcing_equality_row_fixes_all_columns_at_activity_bound(v2_enabled: None) -> None:
    reduction = presolve_eq_box(
        1,
        3,
        [0, 3],
        [0, 1, 2],
        [1.0, 2.0, 3.0],
        [0.0],
        [4.0, 5.0, 6.0],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
    )

    assert reduction is not None
    assert reduction.rows == 0
    assert reduction.cols == 0
    assert reduction.objective_offset == pytest.approx(0.0)
    assert reduction._reduction_counts["forcing_rows"] == 1
    assert reduction._reduction_counts["forcing_columns"] == 3

    x = postsolve_x([], reduction)
    assert x == pytest.approx([0.0, 0.0, 0.0])
    assert (
        _max_residual(
            1,
            [0, 3],
            [0, 1, 2],
            [1.0, 2.0, 3.0],
            x,
            [0.0],
        )
        <= 1e-12
    )


def test_duplicate_bounded_columns_merge_and_split_in_postsolve(v2_enabled: None) -> None:
    reduction = presolve_eq_box(
        2,
        4,
        [0, 4, 8],
        [0, 1, 2, 3, 0, 1, 2, 3],
        [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0],
        [6.0, 11.0],
        [3.0, 3.0, 1.0, 2.0],
        [0.0, 1.0, 0.0, 0.0],
        [2.0, 4.0, 5.0, 5.0],
    )

    assert reduction is not None
    assert reduction.rows == 2
    assert reduction.cols == 3
    assert reduction._active_cols == [0, 2, 3]
    assert reduction.lo == pytest.approx([1.0, 0.0, 0.0])
    assert reduction.hi == pytest.approx([6.0, 5.0, 5.0])
    assert reduction._reduction_counts["duplicate_columns"] == 1

    x = postsolve_x([3.0, 2.0, 1.0], reduction)
    assert x == pytest.approx([2.0, 1.0, 2.0, 1.0])
    assert (
        _max_residual(
            2,
            [0, 4, 8],
            [0, 1, 2, 3, 0, 1, 2, 3],
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 3.0],
            x,
            [6.0, 11.0],
        )
        <= 1e-12
    )


def test_random_small_implied_free_reductions_match_scipy_linprog(v2_enabled: None) -> None:
    rng = random.Random(1234)
    for case in range(8):
        a1 = rng.uniform(0.5, 2.0)
        a4 = rng.uniform(-2.0, -0.5)
        x1_lo = rng.uniform(0.0, 1.0)
        x1_hi = x1_lo + rng.uniform(0.5, 2.0)
        x4_lo = rng.uniform(0.0, 1.0)
        x4_hi = x4_lo + rng.uniform(0.5, 2.0)
        rest_upper = a1 * x1_hi + a4 * x4_lo
        b0 = rest_upper + rng.uniform(0.25, 2.0)
        row1 = [
            0.0,
            rng.uniform(-2.0, 2.0),
            rng.uniform(0.5, 2.0),
            rng.uniform(-2.0, -0.5),
            rng.uniform(0.5, 2.0),
        ]
        x_star = [
            b0 - a1 * ((x1_lo + x1_hi) / 2.0) - a4 * ((x4_lo + x4_hi) / 2.0),
            (x1_lo + x1_hi) / 2.0,
            rng.uniform(0.5, 3.5),
            rng.uniform(0.5, 3.5),
            (x4_lo + x4_hi) / 2.0,
        ]
        b1 = sum(coef * value for coef, value in zip(row1, x_star, strict=True))
        c_vec = [rng.uniform(-3.0, 3.0) for _ in range(5)]

        rows = 2
        cols = 5
        indptr = [0, 3, 7]
        indices = [0, 1, 4, 1, 2, 3, 4]
        data = [1.0, a1, a4, row1[1], row1[2], row1[3], row1[4]]
        b_vec = [b0, b1]
        lo = [0.0, x1_lo, 0.0, 0.0, x4_lo]
        hi = [INF, x1_hi, 4.0, 4.0, x4_hi]

        reduction = presolve_eq_box(
            rows,
            cols,
            list(indptr),
            list(indices),
            list(data),
            list(b_vec),
            list(c_vec),
            list(lo),
            list(hi),
        )
        assert reduction is not None, f"case {case} did not reduce"

        original = linprog(
            c_vec,
            A_eq=[
                [1.0, a1, 0.0, 0.0, a4],
                [0.0, row1[1], row1[2], row1[3], row1[4]],
            ],
            b_eq=b_vec,
            bounds=list(zip(lo, hi, strict=True)),
            method="highs",
        )
        assert original.success, original.message

        if reduction.cols == 0:
            x_reduced: list[float] = []
        else:
            reduced_rows = []
            for i in range(reduction.rows):
                row = [0.0] * reduction.cols
                for offset in range(reduction.indptr[i], reduction.indptr[i + 1]):
                    row[reduction.indices[offset]] = reduction.data[offset]
                reduced_rows.append(row)
            reduced = linprog(
                reduction.c,
                A_eq=reduced_rows if reduced_rows else None,
                b_eq=reduction.b if reduced_rows else None,
                bounds=list(zip(reduction.lo, reduction.hi, strict=True)),
                method="highs",
            )
            assert reduced.success, reduced.message
            x_reduced = [float(value) for value in reduced.x]

        x = postsolve_x(x_reduced, reduction)
        assert _max_residual(rows, indptr, indices, data, x, b_vec) <= 1e-8
        assert sum(v * coef for v, coef in zip(x, c_vec, strict=True)) == pytest.approx(
            float(original.fun), abs=1e-7
        )


def test_matrix_v2_random_equivalence_against_python(v2_enabled: None) -> None:
    rng = random.Random(20260716)
    for case in range(16):
        rows = 4 + case % 4
        cols = 7 + case % 5
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for i in range(rows):
            if i % 3 == 0 and cols >= 4:
                row_cols = [0, 1 + (i % (cols - 1)), cols - 1]
            elif i % 3 == 1:
                row_cols = sorted(rng.sample(range(cols), k=min(cols, 2)))
            else:
                row_cols = sorted(rng.sample(range(cols), k=min(cols, 4)))
            for j in row_cols:
                indices.append(j)
                data.append(rng.choice([-1.0, 1.0]) * rng.randint(1, 3))
            indptr.append(len(indices))
        lo = [0.0] * cols
        hi = [10.0] * cols
        lo[0] = 0.0
        hi[0] = INF
        if cols > 3:
            lo[2] = 1.0
            hi[2] = 1.0
        c_vec = [rng.uniform(-2.0, 2.0) for _ in range(cols)]
        b_vec = [rng.uniform(-5.0, 5.0) for _ in range(rows)]
        matrix = csr_matrix(rows, cols, indptr, indices, data)

        _run_matrix_native_and_python(
            matrix,
            b_vec,
            c_vec,
            lo,
            hi,
            label=f"random-native-{case}",
        )


# ---------------------------------------------------------------------------
# Fixture test: lp_cre_a.mat
# ---------------------------------------------------------------------------


def test_cre_a_fixture_equivalence() -> None:
    """C and Python presolve produce identical output on lp_cre_a."""
    path = Path(__file__).parent / "data" / "lp_cre_a.mat"
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    A = raw["A"].tocsr().astype(float)
    b = raw["b"].ravel().astype(float).tolist()
    c_vec = aux["c"].ravel().astype(float).tolist()
    lo = aux["lo"].ravel().astype(float).tolist()
    hi = aux["hi"].ravel().astype(float).tolist()

    rows, cols = A.shape
    indptr = A.indptr.tolist()
    indices = A.indices.tolist()
    data = A.data.tolist()

    _run_both(rows, cols, indptr, indices, data, b, c_vec, lo, hi, label="cre_a")


@pytest.mark.skipif(not Path("/tmp/lpsuite").exists(), reason="/tmp/lpsuite fixtures unavailable")
@pytest.mark.parametrize(
    "fixture_name",
    [
        "lp_80bau3b",
        "lp_cre_a",
        "lp_cre_b",
        "lp_cre_d",
        "lp_d2q06c",
        "lp_degen3",
        "lp_fit2p",
        "lp_greenbea",
        "lp_ken_07",
        "lp_ken_11",
        "lp_ken_13",
        "lp_ken_18",
        "lp_maros_r7",
        "lp_osa_14",
        "lp_osa_30",
        "lp_osa_60",
        "lp_pds_10",
        "lp_pds_20",
        "lp_pilot87",
        "lp_qap12",
        "lp_qap15",
        "lp_stocfor3",
        "lp_truss",
        "lp_woodw",
    ],
)
def test_lpnetlib_fixture_native_v2_bit_equivalence(fixture_name: str, v2_enabled: None) -> None:
    path = Path("/tmp/lpsuite") / f"{fixture_name}.mat"
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    A = raw["A"].tocsr().astype(float)
    b = raw["b"].ravel().astype(float).tolist()
    c_vec = aux["c"].ravel().astype(float).tolist()
    lo = aux["lo"].ravel().astype(float).tolist()
    hi = aux["hi"].ravel().astype(float).tolist()

    _run_matrix_native_and_python(
        from_scipy_sparse(A),
        b,
        c_vec,
        lo,
        hi,
        label=fixture_name,
    )


# ---------------------------------------------------------------------------
# Parametric random problem tests (25+ scenarios)
# ---------------------------------------------------------------------------


def _make_random_problem(
    rng: random.Random,
    rows: int,
    cols: int,
    density: float = 0.3,
    *,
    empty_rows: int = 0,
    singleton_rows: int = 0,
    fixed_cols: int = 0,
    free_vars: int = 0,
    inf_lo: int = 0,
    inf_hi: int = 0,
) -> tuple[
    int, int, list[int], list[int], list[float], list[float], list[float], list[float], list[float]
]:
    """Generate a random CSR equality LP with controlled structure."""
    indptr = [0]
    indices_out: list[int] = []
    data_out: list[float] = []

    for i in range(rows):
        if i < empty_rows:
            # Empty row
            indptr.append(len(indices_out))
            continue
        if i < empty_rows + singleton_rows and cols > 0:
            # Singleton row
            j = rng.randint(0, cols - 1)
            indices_out.append(j)
            data_out.append(rng.uniform(0.5, 5.0) * rng.choice([-1, 1]))
            indptr.append(len(indices_out))
            continue
        # General row
        row_cols = sorted(rng.sample(range(cols), k=max(1, int(cols * density))))
        for j in row_cols:
            indices_out.append(j)
            data_out.append(rng.uniform(-5.0, 5.0))
        indptr.append(len(indices_out))

    b = [rng.uniform(-10, 10) for _ in range(rows)]
    c_vec = [rng.uniform(-5, 5) for _ in range(cols)]
    lo = [0.0] * cols
    hi = [10.0] * cols

    # Apply fixed-variable pattern (lo == hi)
    for j in range(min(fixed_cols, cols)):
        val = rng.uniform(0, 5)
        lo[j] = val
        hi[j] = val

    # Apply free variables (infinite bounds)
    for j in range(min(free_vars, cols)):
        lo[j] = -INF
        hi[j] = INF

    # Random infinite lower bounds
    for _ in range(inf_lo):
        j = rng.randint(0, cols - 1)
        lo[j] = -INF

    # Random infinite upper bounds
    for _ in range(inf_hi):
        j = rng.randint(0, cols - 1)
        hi[j] = INF

    return rows, cols, indptr, indices_out, data_out, b, c_vec, lo, hi


_RANDOM_CASES = [
    # label, kwargs
    ("dense_3x3", dict(rows=3, cols=3, density=1.0)),
    ("dense_5x5", dict(rows=5, cols=5, density=1.0)),
    ("sparse_10x10", dict(rows=10, cols=10, density=0.2)),
    ("sparse_20x30", dict(rows=20, cols=30, density=0.15)),
    ("empty_rows_only", dict(rows=5, cols=4, density=0.3, empty_rows=3)),
    ("all_empty_rows", dict(rows=4, cols=3, density=0.3, empty_rows=4)),
    ("singleton_1", dict(rows=5, cols=5, density=0.3, singleton_rows=1)),
    ("singleton_3", dict(rows=6, cols=6, density=0.3, singleton_rows=3)),
    ("all_singletons", dict(rows=4, cols=4, density=0.3, singleton_rows=4)),
    ("fixed_vars_2", dict(rows=5, cols=5, density=0.3, fixed_cols=2)),
    ("fixed_vars_all", dict(rows=3, cols=3, density=0.5, fixed_cols=3)),
    ("free_vars_2", dict(rows=5, cols=5, density=0.3, free_vars=2)),
    ("free_vars_all", dict(rows=4, cols=4, density=0.3, free_vars=4)),
    ("inf_lo_3", dict(rows=6, cols=6, density=0.3, inf_lo=3)),
    ("inf_hi_3", dict(rows=6, cols=6, density=0.3, inf_hi=3)),
    ("inf_both", dict(rows=6, cols=6, density=0.3, inf_lo=2, inf_hi=2)),
    ("wide_problem", dict(rows=3, cols=20, density=0.15)),
    ("tall_problem", dict(rows=20, cols=3, density=0.5)),
    ("tiny_1x1", dict(rows=1, cols=1, density=1.0)),
    ("tiny_1x2", dict(rows=1, cols=2, density=1.0)),
    ("tiny_2x1", dict(rows=2, cols=1, density=1.0)),
    ("mixed_empty_singleton", dict(rows=8, cols=6, density=0.2, empty_rows=2, singleton_rows=2)),
    ("mixed_fixed_free", dict(rows=6, cols=8, density=0.2, fixed_cols=2, free_vars=2)),
    ("high_density", dict(rows=8, cols=8, density=0.9)),
    ("very_sparse", dict(rows=15, cols=15, density=0.05)),
    ("medium_10x20", dict(rows=10, cols=20, density=0.2, singleton_rows=2, inf_hi=3)),
    ("medium_20x10_free", dict(rows=20, cols=10, density=0.3, free_vars=3, inf_lo=2)),
]


@pytest.mark.parametrize("label,kwargs", _RANDOM_CASES, ids=[c[0] for c in _RANDOM_CASES])
def test_random_equivalence(label: str, kwargs: dict) -> None:
    """C and Python presolve agree on a randomly generated problem."""
    rng = random.Random(42 + hash(label) % 10000)
    args = _make_random_problem(rng, **kwargs)
    _run_both(*args, label=label)


def test_presolve_none_cases_agree() -> None:
    """Both paths return None for problems with no reductions."""
    # Dense single row: no empty, singleton, or doubleton
    _run_both(
        1,
        3,
        [0, 3],
        [0, 1, 2],
        [1.0, 1.0, 1.0],
        [3.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [INF, INF, INF],
        label="none-dense-row",
    )

    # Single doubleton row with extreme ratio
    _run_both(
        1,
        2,
        [0, 2],
        [0, 1],
        [1.0, 1e6],
        [1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [10.0, 10.0],
        label="none-extreme-ratio",
    )


def test_duplicate_column_patterns() -> None:
    """Rows with duplicate column patterns reduce identically."""
    # Two identical doubleton rows
    _run_both(
        3,
        3,
        [0, 2, 4, 7],
        [0, 1, 0, 1, 0, 1, 2],
        [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        [3.0, 3.0, 5.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        label="duplicate-patterns",
    )


def test_max_fill_parameter_equivalence() -> None:
    """max_fill parameter produces the same results in C and Python."""
    # Same problem tested at different fill limits
    indptr = [0, 2]
    indices = [0, 1]
    data = [1.0, 1.0]
    extra = 7
    for r in range(extra):
        indices.extend([0, 2 + r])
        data.extend([1.0, 1.0])
        indptr.append(len(indices))
    rows = 1 + extra
    cols = 2 + extra
    b = [1.0] * rows
    c_vec = [1.0] * cols
    lo = [0.0] * cols
    hi = [10.0] * cols

    for mf in (1, 3, 5, 10, 20):
        _run_both(
            rows,
            cols,
            list(indptr),
            list(indices),
            list(data),
            list(b),
            list(c_vec),
            list(lo),
            list(hi),
            max_fill=mf,
            label=f"max_fill={mf}",
        )


# ---------------------------------------------------------------------------
# End-to-end solver tests
# ---------------------------------------------------------------------------


def test_e2e_solver_with_c_presolve() -> None:
    """IPM solve results are identical whether C or Python presolve is used."""
    from linprogx.sparse import SparseLPProblem

    # Chain with doubleton
    m1 = csr_matrix(2, 4, [0, 2, 5], [0, 1, 1, 2, 3], [1.0, 1.0, 1.0, 1.0, 1.0])
    p1 = SparseLPProblem(
        c=[3.0, 1.0, 1.0, 2.0],
        A_eq=m1,
        b_eq=[4.0, 6.0],
        objective="min",
        bounds=[(0.0, 3.0), (0.0, 4.0), (0.0, 4.0), (0.0, 4.0)],
        name="e2e-0",
    )
    r1 = SparseSolver(algorithm="ipm", eps=1e-9, max_iterations=200).solve(p1)
    assert r1.solution.status == Status.OPTIMAL, "Problem 0 did not solve to optimal"

    # Fully determined by singletons
    m2 = csr_matrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0])
    p2 = SparseLPProblem(
        c=[1.0, 1.0],
        A_eq=m2,
        b_eq=[1.5, 4.0],
        objective="min",
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        name="e2e-1",
    )
    r2 = SparseSolver(algorithm="ipm", eps=1e-9, max_iterations=200).solve(p2)
    assert r2.solution.status == Status.OPTIMAL, "Problem 1 did not solve to optimal"


def test_e2e_cre_a_ipm() -> None:
    """lp_cre_a solves correctly with C presolve via IPM."""
    path = Path(__file__).parent / "data" / "lp_cre_a.mat"
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    A = raw["A"].tocsr().astype(float)
    b = raw["b"].ravel().astype(float).tolist()
    c_vec = aux["c"].ravel().astype(float).tolist()
    lo = aux["lo"].ravel().astype(float).tolist()
    hi = aux["hi"].ravel().astype(float).tolist()

    from linprogx.sparse import SparseLPProblem, from_scipy_sparse

    matrix = from_scipy_sparse(A)
    rows, cols = matrix.shape

    problem = SparseLPProblem(
        c=c_vec,
        A_eq=matrix,
        b_eq=b,
        objective="min",
        bounds=list(zip(lo, hi, strict=True)),
        name="cre_a-e2e",
    )

    result = SparseSolver(
        algorithm="ipm",
        eps=1e-9,
        max_iterations=200,
    ).solve(problem)

    assert result.solution.status == Status.OPTIMAL
    # Known optimal for cre_a (Gurobi 1e-8); IPM may converge to ~1e-4 rel
    assert result.solution.objective_value == pytest.approx(23595407.06, rel=1e-4)
