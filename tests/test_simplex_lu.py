"""Characterization tests for the sparse LU factorization (Markowitz pivot selection).

These tests exercise the C LU factorization via the ``_csparse.lu_solve_test``
and ``_csparse.lu_stats_test`` test hooks, comparing against scipy.sparse.linalg.splu
as the correctness oracle.

Dense RHS for milestone 1; sparse-RHS Gilbert-Peierls comes later.
"""

from __future__ import annotations

import importlib

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

_csparse = importlib.import_module("linprogx._csparse")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _csc_args(A: sp.csc_matrix) -> tuple[list[int], list[int], list[float], int]:
    """Return (indptr, indices, data, m) for the C test hooks."""
    A = A.tocsc()
    return (
        A.indptr.tolist(),
        A.indices.tolist(),
        A.data.tolist(),
        A.shape[0],
    )


def _max_residual(A: sp.spmatrix, x: np.ndarray, b: np.ndarray) -> float:
    """Return max |A x - b|."""
    return float(np.max(np.abs(A @ x - b)))


def _relative_tol(b: np.ndarray, base: float = 1e-10) -> float:
    """Return base * (1 + |b|_inf)."""
    return base * (1.0 + float(np.max(np.abs(b))))


# ---------------------------------------------------------------------------
# FTRAN tests: solve B x = b
# ---------------------------------------------------------------------------


class TestFTRAN:
    """Tests for lu_ftran (forward transformation, Bx = b)."""

    def test_random_50(self) -> None:
        np.random.seed(101)
        m = 50
        A = sp.random(m, m, density=0.05, format="csc", random_state=101) + sp.eye(m)
        for seed in [0, 1, 2]:
            rng = np.random.RandomState(seed)
            b = rng.randn(m)
            x_scipy = sla.splu(A.tocsc()).solve(b)
            result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 0)
            assert result is not None, "should not be singular"
            x_ours = np.array(result[0])
            res = _max_residual(A, x_ours, b)
            assert res <= _relative_tol(b), f"residual {res} too large"
            assert np.allclose(x_ours, x_scipy, atol=1e-10)

    def test_random_200(self) -> None:
        np.random.seed(202)
        m = 200
        A = sp.random(m, m, density=0.02, format="csc", random_state=202) + sp.eye(m)
        b = np.random.randn(m)
        x_scipy = sla.splu(A.tocsc()).solve(b)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 0)
        assert result is not None
        x_ours = np.array(result[0])
        res = _max_residual(A, x_ours, b)
        assert res <= _relative_tol(b), f"residual {res} too large"
        assert np.allclose(x_ours, x_scipy, atol=1e-10)

    def test_random_1000(self) -> None:
        np.random.seed(303)
        m = 1000
        A = sp.random(m, m, density=0.01, format="csc", random_state=303) + sp.eye(m)
        b = np.random.randn(m)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 0)
        assert result is not None
        x_ours = np.array(result[0])
        res = _max_residual(A, x_ours, b)
        assert res <= _relative_tol(b), f"residual {res} too large for m=1000"


# ---------------------------------------------------------------------------
# BTRAN tests: solve B^T x = b
# ---------------------------------------------------------------------------


class TestBTRAN:
    """Tests for lu_btran (backward transformation, B^T x = b)."""

    def test_random_50(self) -> None:
        np.random.seed(111)
        m = 50
        A = sp.random(m, m, density=0.05, format="csc", random_state=111) + sp.eye(m)
        for seed in [10, 11, 12]:
            rng = np.random.RandomState(seed)
            b = rng.randn(m)
            x_scipy = sla.splu(A.tocsc()).solve(b, trans="T")
            result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 1)
            assert result is not None
            x_ours = np.array(result[0])
            res = _max_residual(A.T, x_ours, b)
            assert res <= _relative_tol(b), f"btran residual {res} too large"
            assert np.allclose(x_ours, x_scipy, atol=1e-10)

    def test_random_200(self) -> None:
        np.random.seed(222)
        m = 200
        A = sp.random(m, m, density=0.02, format="csc", random_state=222) + sp.eye(m)
        b = np.random.randn(m)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 1)
        assert result is not None
        x_ours = np.array(result[0])
        res = _max_residual(A.T, x_ours, b)
        assert res <= _relative_tol(b), f"btran residual {res} too large"

    def test_random_1000(self) -> None:
        np.random.seed(333)
        m = 1000
        A = sp.random(m, m, density=0.01, format="csc", random_state=333) + sp.eye(m)
        b = np.random.randn(m)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 1)
        assert result is not None
        x_ours = np.array(result[0])
        res = _max_residual(A.T, x_ours, b)
        assert res <= _relative_tol(b), f"btran residual {res} too large for m=1000"


# ---------------------------------------------------------------------------
# Singularity tests
# ---------------------------------------------------------------------------


class TestSingular:
    """Verify singularity detection without crashes."""

    def test_structurally_singular_zero_column(self) -> None:
        """A matrix with a zero column is structurally singular."""
        m = 5
        # Build a matrix where column 2 is all zeros
        A = sp.eye(m, format="csc").tolil()
        A[:, 2] = 0.0
        A = A.tocsc()
        A.eliminate_zeros()
        result = _csparse.lu_solve_test(*_csc_args(A), [[1.0] * m], 0)
        assert result is None, "zero column should be detected as singular"
        stats = _csparse.lu_stats_test(*_csc_args(A))
        assert stats[2] >= 0, "singular_step should be non-negative"

    def test_numerically_singular_duplicate_columns(self) -> None:
        """A matrix with two identical columns is rank-deficient."""
        m = 5
        A = sp.eye(m, format="csc").tolil()
        A[:, 3] = A[:, 1].toarray()  # duplicate column 1 into column 3
        A = A.tocsc()
        result = _csparse.lu_solve_test(*_csc_args(A), [[1.0] * m], 0)
        assert result is None, "duplicate columns should be detected as singular"
        stats = _csparse.lu_stats_test(*_csc_args(A))
        assert stats[2] >= 0

    def test_zero_matrix(self) -> None:
        """All-zero matrix."""
        m = 4
        A = sp.csc_matrix((m, m))
        result = _csparse.lu_solve_test(*_csc_args(A), [[1.0] * m], 0)
        assert result is None

    def test_singular_stats_no_crash(self) -> None:
        """lu_stats_test should not crash on singular input."""
        m = 3
        A = sp.csc_matrix((m, m))
        stats = _csparse.lu_stats_test(*_csc_args(A))
        assert stats[2] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Diagonal matrices, permutation matrices, and 1x1."""

    def test_diagonal(self) -> None:
        m = 10
        diag_vals = np.array([2.0, 0.5, 3.0, 1.0, 4.0, 0.25, 7.0, 0.1, 5.0, 6.0])
        A = sp.diags(diag_vals, format="csc")
        b = np.ones(m)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 0)
        assert result is not None
        x = np.array(result[0])
        expected = 1.0 / diag_vals
        assert np.allclose(x, expected, atol=1e-14)

    def test_permutation_matrix(self) -> None:
        m = 6
        perm = [3, 0, 5, 2, 4, 1]
        rows = list(range(m))
        A = sp.csc_matrix((np.ones(m), (rows, perm)), shape=(m, m))
        b = np.arange(1.0, m + 1)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 0)
        assert result is not None
        x = np.array(result[0])
        res = _max_residual(A, x, b)
        assert res < 1e-14

    def test_1x1(self) -> None:
        A = sp.csc_matrix(np.array([[3.0]]))
        result = _csparse.lu_solve_test(*_csc_args(A), [[6.0]], 0)
        assert result is not None
        assert abs(result[0][0] - 2.0) < 1e-14

    def test_identity(self) -> None:
        m = 20
        A = sp.eye(m, format="csc")
        b = np.random.RandomState(42).randn(m)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 0)
        assert result is not None
        assert np.allclose(result[0], b.tolist(), atol=1e-15)


# ---------------------------------------------------------------------------
# Fill-in stress: arrow matrix
# ---------------------------------------------------------------------------


class TestFillIn:
    """Arrow matrix: dense last row + column causes fill-in."""

    def test_arrow_correctness_and_fill(self) -> None:
        m = 100
        # Arrow matrix: diagonal + dense last row + dense last column
        A = sp.eye(m, format="lil") * 2.0
        rng = np.random.RandomState(777)
        for i in range(m - 1):
            A[i, m - 1] = rng.randn()
            A[m - 1, i] = rng.randn()
        A[m - 1, m - 1] = 10.0  # strong diagonal for stability
        A = A.tocsc()

        b = rng.randn(m)
        result = _csparse.lu_solve_test(*_csc_args(A), [b.tolist()], 0)
        assert result is not None
        x_ours = np.array(result[0])
        res = _max_residual(A, x_ours, b)
        assert res <= _relative_tol(b), f"arrow residual {res} too large"

        # Check fill-in is sane: nnz(L) + nnz(U) should be much less than m*m
        stats = _csparse.lu_stats_test(*_csc_args(A))
        nnz_l, nnz_u, singular_step = stats
        assert singular_step == -1
        total_fill = nnz_l + nnz_u
        dense_entries = m * m
        # Arrow matrix should have moderate fill; certainly < m*m/2
        assert total_fill < dense_entries // 2, (
            f"fill-in too large: {total_fill} vs dense {dense_entries}"
        )
        # The input has ~3m-2 nonzeros; fill shouldn't exceed ~10x that
        input_nnz = A.nnz
        assert total_fill < input_nnz * 10, f"excessive fill-in: {total_fill} vs input {input_nnz}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Factorizing the same matrix twice must give identical results."""

    def test_deterministic_stats_and_solve(self) -> None:
        np.random.seed(999)
        m = 100
        A = sp.random(m, m, density=0.03, format="csc", random_state=999) + sp.eye(m)
        b = np.random.randn(m)

        args = _csc_args(A)
        stats1 = _csparse.lu_stats_test(*args)
        stats2 = _csparse.lu_stats_test(*args)
        assert stats1 == stats2, "LU stats should be deterministic"

        result1 = _csparse.lu_solve_test(*args, [b.tolist()], 0)
        result2 = _csparse.lu_solve_test(*args, [b.tolist()], 0)
        assert result1 is not None and result2 is not None
        x1 = np.array(result1[0])
        x2 = np.array(result2[0])
        assert np.array_equal(x1, x2), "LU solve should be deterministic (bit-identical)"

    def test_deterministic_btran(self) -> None:
        np.random.seed(888)
        m = 80
        A = sp.random(m, m, density=0.04, format="csc", random_state=888) + sp.eye(m)
        b = np.random.randn(m)

        args = _csc_args(A)
        result1 = _csparse.lu_solve_test(*args, [b.tolist()], 1)
        result2 = _csparse.lu_solve_test(*args, [b.tolist()], 1)
        assert result1 is not None and result2 is not None
        assert np.array_equal(result1[0], result2[0])


# ---------------------------------------------------------------------------
# Multiple RHS
# ---------------------------------------------------------------------------


class TestMultipleRHS:
    """Verify solving with multiple right-hand sides in one call."""

    def test_multi_rhs_ftran(self) -> None:
        np.random.seed(555)
        m = 50
        A = sp.random(m, m, density=0.05, format="csc", random_state=555) + sp.eye(m)
        rhs_list = [np.random.randn(m) for _ in range(5)]
        rhs_py = [b.tolist() for b in rhs_list]

        result = _csparse.lu_solve_test(*_csc_args(A), rhs_py, 0)
        assert result is not None
        assert len(result) == 5

        for i, b in enumerate(rhs_list):
            x_ours = np.array(result[i])
            res = _max_residual(A, x_ours, b)
            assert res <= _relative_tol(b), f"rhs {i}: residual {res} too large"


# ---------------------------------------------------------------------------
# Basis-change update tests (Milestone 2)
# ---------------------------------------------------------------------------


def _sparse_col(col: np.ndarray) -> tuple[list[int], list[float]]:
    """Return (indices, values) for a dense column vector's nonzero entries."""
    nz = np.nonzero(col)[0]
    return nz.tolist(), col[nz].tolist()


class TestUpdate:
    """Tests for lu_update (Forrest-Tomlin / product-form basis updates)."""

    def test_sequential_ftran_50_updates(self) -> None:
        """Apply 50 column replacements and verify FTRAN against scipy."""
        np.random.seed(4001)
        m = 60
        B = sp.random(m, m, density=0.05, format="csc", random_state=4001) + 2.0 * sp.eye(m)
        B_dense = B.toarray()

        rng = np.random.RandomState(4002)
        updates = []
        for _ in range(50):
            # Pick a random column to replace with a random column
            pos = rng.randint(0, m)
            new_col = rng.randn(m) * 0.5
            # Strengthen diagonal to keep nonsingular
            new_col[pos] += 2.0
            idx, vals = _sparse_col(new_col)
            updates.append((pos, idx, vals))
            B_dense[:, pos] = new_col

        b = rng.randn(m)
        result = _csparse.lu_update_test(*_csc_args(B), updates, [b.tolist()], 0)
        assert result is not None, "initial factorization should not be singular"
        solutions, should_refac, n_singular = result
        x_ours = np.array(solutions[0])

        # Verify against scipy (solve B_final x = b)
        B_final = sp.csc_matrix(B_dense)
        x_scipy = sla.spsolve(B_final, b)

        # Check residual: B_final * x_ours = b
        res = float(np.max(np.abs(B_final @ x_ours - b)))
        tol = _relative_tol(b, base=1e-8)
        assert res <= tol, f"FTRAN residual after 50 updates: {res} > {tol}"
        assert np.allclose(x_ours, x_scipy, atol=1e-6), (
            f"max diff from scipy: {np.max(np.abs(x_ours - x_scipy))}"
        )

    def test_sequential_btran_verification(self) -> None:
        """Apply updates and verify BTRAN (B^T x = b) against scipy."""
        np.random.seed(4010)
        m = 40
        B = sp.random(m, m, density=0.06, format="csc", random_state=4010) + 3.0 * sp.eye(m)
        B_dense = B.toarray()

        rng = np.random.RandomState(4011)
        updates = []
        for _ in range(20):
            pos = rng.randint(0, m)
            new_col = rng.randn(m) * 0.3
            new_col[pos] += 3.0
            idx, vals = _sparse_col(new_col)
            updates.append((pos, idx, vals))
            B_dense[:, pos] = new_col

        b = rng.randn(m)
        result = _csparse.lu_update_test(*_csc_args(B), updates, [b.tolist()], 1)
        assert result is not None
        solutions, _, _ = result
        x_ours = np.array(solutions[0])

        # Verify: B_final^T * x_ours = b
        B_final = sp.csc_matrix(B_dense)
        x_scipy = sla.spsolve(B_final.T.tocsc(), b)

        res = float(np.max(np.abs(B_final.T @ x_ours - b)))
        tol = _relative_tol(b, base=1e-8)
        assert res <= tol, f"BTRAN residual after 20 updates: {res} > {tol}"
        assert np.allclose(x_ours, x_scipy, atol=1e-6), (
            f"BTRAN max diff from scipy: {np.max(np.abs(x_ours - x_scipy))}"
        )

    def test_should_refactor_fires_by_100(self) -> None:
        """lu_should_refactor must fire by 100 updates."""
        np.random.seed(4020)
        m = 30
        B = sp.random(m, m, density=0.08, format="csc", random_state=4020) + 5.0 * sp.eye(m)

        rng = np.random.RandomState(4021)
        updates = []
        for _ in range(100):
            pos = rng.randint(0, m)
            new_col = rng.randn(m) * 0.2
            new_col[pos] += 5.0
            idx, vals = _sparse_col(new_col)
            updates.append((pos, idx, vals))

        b = rng.randn(m)
        result = _csparse.lu_update_test(*_csc_args(B), updates, [b.tolist()], 0)
        assert result is not None
        _, should_refac, _ = result
        assert should_refac == 1, "should_refactor must fire by 100 updates"

    def test_singular_update_detection(self) -> None:
        """Replacing a column with a duplicate of another column is singular."""
        np.random.seed(4030)
        m = 10
        B = sp.eye(m, format="csc") * 2.0

        # Replace column 3 with a copy of column 5 (makes B singular)
        col5 = np.zeros(m)
        col5[5] = 2.0  # column 5 of 2*I
        idx, vals = _sparse_col(col5)
        updates = [(3, idx, vals)]

        b = np.ones(m)
        result = _csparse.lu_update_test(*_csc_args(B), updates, [b.tolist()], 0)
        assert result is not None
        _, _, n_singular = result
        assert n_singular >= 1, "singular update should be detected"

    def test_update_determinism(self) -> None:
        """Applying the same updates twice must give bit-identical results."""
        np.random.seed(4040)
        m = 50
        B = sp.random(m, m, density=0.05, format="csc", random_state=4040) + 2.0 * sp.eye(m)

        rng = np.random.RandomState(4041)
        updates = []
        for _ in range(25):
            pos = rng.randint(0, m)
            new_col = rng.randn(m) * 0.4
            new_col[pos] += 2.0
            idx, vals = _sparse_col(new_col)
            updates.append((pos, idx, vals))

        b = np.random.RandomState(4042).randn(m)
        args = _csc_args(B)

        result1 = _csparse.lu_update_test(*args, updates, [b.tolist()], 0)
        result2 = _csparse.lu_update_test(*args, updates, [b.tolist()], 0)
        assert result1 is not None and result2 is not None

        x1 = np.array(result1[0][0])
        x2 = np.array(result2[0][0])
        assert np.array_equal(x1, x2), "LU update solve should be deterministic"
        assert result1[1] == result2[1], "should_refactor should be deterministic"
        assert result1[2] == result2[2], "n_singular should be deterministic"
