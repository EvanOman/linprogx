"""Equivalence tests: C presolve must reproduce Python presolve exactly.

These tests compare the C accelerator in ``_csparse.presolve_eq_box`` against
the pure-Python ``_presolve_eq_box_python`` on a range of problem shapes.
Every field of the PresolveResult must match with exact float equality (same
arithmetic, same reductions, same order).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from scipy.io import loadmat

from linprogx.presolve import (
    PresolveResult,
    _Doubleton,
    _FixedVar,
    _presolve_eq_box_python,
    presolve_eq_box,
)
from linprogx.sparse import SparseSolver, csr_matrix
from linprogx.types import Status

INF = float("inf")


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
        else:
            assert isinstance(cr, _Doubleton)
            assert isinstance(pr, _Doubleton)
            assert cr.eliminated == pr.eliminated, f"record {k} eliminated mismatch{msg}"
            assert cr.kept == pr.kept, f"record {k} kept mismatch{msg}"
            assert cr.coef_eliminated == pr.coef_eliminated, (
                f"record {k} coef_eliminated mismatch{msg}"
            )
            assert cr.coef_kept == pr.coef_kept, f"record {k} coef_kept mismatch{msg}"
            assert cr.rhs == pr.rhs, f"record {k} rhs mismatch{msg}"


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
