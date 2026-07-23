"""Falsifier for certificate-corrected iterative greenbea basis solves.

The probe is diagnostic-only.  It exports native linprogx bases at fixed
iteration limits, reconstructs the solver's Ruiz-scaled basis matrices, and
uses the single changed basis column at k+1 to identify the actual next
FTRAN and BTRAN right-hand sides at checkpoint k.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.linalg import splu

from experiments.greenbea_pivot_gap_probe import prepare

CHECKPOINTS = (512, 1536, 3072, 4096)
TAUS = np.linspace(0.0, 1e-8, 101)
RECORDED_DEGREES = (1, 2, 3, 4, 8, 16, 32)
MAX_DEGREE = max(RECORDED_DEGREES)
REPS = 5000
BATCHES = 11
CURRENT_PAIR_US = 47.425
CURRENT_SOLVE_SHARE = 0.3462
PROBE_WALL_GATE = 0.20
BOARD_GAP = 0.177013
OUT_DIR = Path("/tmp/krylov-basis-solve-falsifier")
RESULTS = OUT_DIR / "results.json"


def ruiz_scaled(a: sparse.csc_matrix) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Reproduce the native DS 10-pass inf-norm + one l2 Ruiz scaling."""
    m, n = a.shape
    row_scale = np.ones(m)
    col_scale = np.ones(n)
    row_norm = np.asarray(abs(a).max(axis=1).toarray()).ravel()
    nz_row_norm = row_norm[row_norm > 0.0]
    active = bool(nz_row_norm.size and nz_row_norm.max() / nz_row_norm.min() >= 100.0)

    for _ in range(10 if active else 0):
        scaled = sparse.diags(row_scale) @ a @ sparse.diags(col_scale)
        row_norm = np.asarray(abs(scaled).max(axis=1).toarray()).ravel()
        col_norm = np.asarray(abs(scaled).max(axis=0).toarray()).ravel()
        row_scale[row_norm > 0.0] /= np.sqrt(row_norm[row_norm > 0.0])
        col_scale[col_norm > 0.0] /= np.sqrt(col_norm[col_norm > 0.0])

    if active:
        scaled = sparse.diags(row_scale) @ a @ sparse.diags(col_scale)
        row_norm = np.asarray(scaled.multiply(scaled).sum(axis=1)).ravel()
        col_norm = np.asarray(scaled.multiply(scaled).sum(axis=0)).ravel()
        row_scale[row_norm > 0.0] /= np.sqrt(np.sqrt(row_norm[row_norm > 0.0]))
        col_scale[col_norm > 0.0] /= np.sqrt(np.sqrt(col_norm[col_norm > 0.0]))
        row_scale = np.clip(row_scale, 1e-8, 1e8)
        col_scale = np.clip(col_scale, 1e-8, 1e8)

    scaled = (sparse.diags(row_scale) @ a @ sparse.diags(col_scale)).tocsc()
    return scaled, row_scale, col_scale


def basis_column(a: sparse.csc_matrix, j: int) -> sparse.csc_matrix:
    m, n = a.shape
    if j < n:
        return a[:, j]
    return sparse.csc_matrix(([1.0], ([j - n], [0])), shape=(m, 1))


def native_prefix(prepared: dict[str, Any], cap: int) -> dict[str, Any]:
    return prepared["matrix"].solve_eq_box_dual_simplex(
        prepared["c"].tolist(),
        prepared["b"].tolist(),
        prepared["lo"].tolist(),
        prepared["hi"].tolist(),
        max_iter=cap,
        tol=1e-8,
        expand=1,
        leaving_rule=1,
        bfrt=0,
    )


def timed_pair(
    basis: sparse.csr_matrix,
    basis_t: sparse.csr_matrix,
    x: np.ndarray,
    y: np.ndarray,
) -> list[float]:
    for _ in range(100):
        basis.dot(x)
        basis_t.dot(y)
    batches = []
    for _ in range(BATCHES):
        started = time.perf_counter_ns()
        for _ in range(REPS):
            basis.dot(x)
            basis_t.dot(y)
        batches.append((time.perf_counter_ns() - started) / REPS / 1000.0)
    return batches


def matched_jacobi_minres_series(
    basis: sparse.csc_matrix,
    matching: np.ndarray,
    diagonal: np.ndarray,
    rhs: np.ndarray,
    *,
    transpose: bool,
) -> dict[int, tuple[float, np.ndarray]]:
    """Repeated matched-Jacobi corrections with exact residual line search."""
    estimate = np.zeros(basis.shape[0])
    residual = rhs.copy()
    rhs_norm = max(float(np.linalg.norm(rhs)), 1e-300)
    recorded: dict[int, tuple[float, np.ndarray]] = {}
    for degree in range(1, MAX_DEGREE + 1):
        if transpose:
            direction = residual[matching] / diagonal
            image = basis.T @ direction
        else:
            direction = np.zeros(basis.shape[0])
            direction[matching] = residual / diagonal
            image = basis @ direction
        denominator = float(image @ image)
        step = float(image @ residual) / denominator if denominator else 0.0
        estimate += step * direction
        residual -= step * image
        if degree in RECORDED_DEGREES:
            recorded[degree] = (
                float(np.linalg.norm(residual) / rhs_norm),
                estimate.copy(),
            )
    return recorded


def harris_choice(
    alpha: np.ndarray,
    reduced_cost: np.ndarray,
    status: np.ndarray,
    basis_set: set[int],
    sigma: int,
    tau: float,
) -> int:
    candidates: list[tuple[int, float, float, float]] = []
    for j, alpha_j in enumerate(alpha):
        if j in basis_set or status[j] == 3 or abs(alpha_j) < 1e-9:
            continue
        admissible = (
            (status[j] == 0 and sigma * alpha_j < 0.0)
            or (status[j] == 1 and sigma * alpha_j > 0.0)
            or status[j] == 2
        )
        if not admissible:
            continue
        candidates.append(
            (
                j,
                float(alpha_j),
                (abs(reduced_cost[j]) + tau) / abs(alpha_j),
                abs(reduced_cost[j]) / abs(alpha_j),
            )
        )
    if not candidates:
        return -1
    theta_max = min(candidate[2] for candidate in candidates) + 1e-7
    entering = -1
    best_alpha = 0.0
    for j, alpha_j, _, plain_ratio in candidates:
        if plain_ratio <= theta_max and abs(alpha_j) > best_alpha:
            entering = j
            best_alpha = abs(alpha_j)
    return entering


def run() -> dict[str, Any]:
    _, prepared = prepare()
    raw_a = prepared["A_scipy"].tocsc().astype(np.float64)
    scaled_a, row_scale, col_scale = ruiz_scaled(raw_a)
    m, n = raw_a.shape
    scaled_c = prepared["c"] * col_scale
    scaled_lo = prepared["lo"] / col_scale
    scaled_hi = prepared["hi"] / col_scale
    scaled_b = prepared["b"] * row_scale
    finite_bounds = np.r_[
        np.abs(scaled_lo[np.isfinite(scaled_lo)]),
        np.abs(scaled_hi[np.isfinite(scaled_hi)]),
    ]
    big_m = 1e5 * max(1.0, float(np.max(np.abs(scaled_b))), float(np.max(finite_bounds)))

    rows = []
    for checkpoint in CHECKPOINTS:
        current = native_prefix(prepared, checkpoint)
        following = native_prefix(prepared, checkpoint + 1)
        basis_index = np.asarray(current["basis"], dtype=np.int64)
        next_basis = np.asarray(following["basis"], dtype=np.int64)
        changed = np.flatnonzero(basis_index != next_basis)
        if changed.size != 1:
            raise RuntimeError(f"checkpoint {checkpoint}: expected one changed basis position")
        leaving = int(changed[0])
        entering = int(next_basis[leaving])
        columns = [basis_column(scaled_a, int(j)) for j in basis_index]
        basis = sparse.hstack(columns, format="csc")
        basis_csr = basis.tocsr()
        basis_t_csr = basis.T.tocsr()
        factor = splu(basis)

        ftran_rhs = basis_column(scaled_a, entering).toarray().ravel()
        btran_rhs = np.zeros(m)
        btran_rhs[leaving] = 1.0
        exact_ftran = factor.solve(ftran_rhs)
        exact_btran = factor.solve(btran_rhs, trans="T")

        # Matched-diagonal stationary corrections. Each degree applies D^-1
        # to the current true residual and receives an exact residual-minimizing
        # scalar for free.
        matching = maximum_bipartite_matching(basis, perm_type="column")
        if np.any(matching < 0):
            raise RuntimeError(f"checkpoint {checkpoint}: incomplete structural matching")
        diagonal = np.asarray(basis[np.arange(m), matching]).ravel()
        ftran_series = matched_jacobi_minres_series(
            basis, matching, diagonal, ftran_rhs, transpose=False
        )
        btran_series = matched_jacobi_minres_series(
            basis, matching, diagonal, btran_rhs, transpose=True
        )

        basis_set = set(int(value) for value in basis_index)
        status = np.asarray(current["bound_status"], dtype=np.int8)
        c_basis = np.asarray([scaled_c[j] if j < n else 0.0 for j in basis_index])
        dual = factor.solve(c_basis, trans="T")
        reduced_cost = np.r_[scaled_c - scaled_a.T @ dual, -dual]
        reduced_cost[basis_index] = 0.0

        nonbasic_x = np.zeros(n + m)
        for j in range(n):
            if j in basis_set:
                continue
            if status[j] == 0:
                nonbasic_x[j] = scaled_lo[j] if np.isfinite(scaled_lo[j]) else scaled_hi[j] - big_m
            elif status[j] == 1:
                nonbasic_x[j] = scaled_hi[j] if np.isfinite(scaled_hi[j]) else scaled_lo[j] + big_m
            elif status[j] == 2:
                nonbasic_x[j] = 0.0
            elif status[j] == 3:
                nonbasic_x[j] = scaled_lo[j]
        primal_rhs = scaled_b - scaled_a @ nonbasic_x[:n]
        x_basis = factor.solve(primal_rhs)
        leaving_column = int(basis_index[leaving])
        leaving_lo = scaled_lo[leaving_column] if leaving_column < n else 0.0
        leaving_hi = scaled_hi[leaving_column] if leaving_column < n else 0.0
        if np.isfinite(leaving_lo) and x_basis[leaving] < leaving_lo - 1e-8:
            sigma = 1
        elif np.isfinite(leaving_hi) and x_basis[leaving] > leaving_hi + 1e-8:
            sigma = -1
        else:
            raise RuntimeError(f"checkpoint {checkpoint}: changed row is not primal-infeasible")

        exact_alpha = np.r_[scaled_a.T @ exact_btran, exact_btran]
        exact_choice_matches = sum(
            harris_choice(exact_alpha, reduced_cost, status, basis_set, sigma, float(tau))
            == entering
            for tau in TAUS
        )
        degree_rows = {}
        for degree in RECORDED_DEGREES:
            ftran_residual, approximate_ftran = ftran_series[degree]
            btran_residual, approximate_btran = btran_series[degree]
            approximate_alpha = np.r_[scaled_a.T @ approximate_btran, approximate_btran]
            approximate_choice_matches = sum(
                harris_choice(
                    approximate_alpha,
                    reduced_cost,
                    status,
                    basis_set,
                    sigma,
                    float(tau),
                )
                == entering
                for tau in TAUS
            )
            degree_rows[str(degree)] = {
                "ftran_relative_residual": ftran_residual,
                "btran_relative_residual": btran_residual,
                "ftran_relative_solution_error": float(
                    np.linalg.norm(approximate_ftran - exact_ftran) / np.linalg.norm(exact_ftran)
                ),
                "btran_relative_solution_error": float(
                    np.linalg.norm(approximate_btran - exact_btran) / np.linalg.norm(exact_btran)
                ),
                "approximate_entering_matches_across_tau_grid": approximate_choice_matches,
                "tau_grid_size": len(TAUS),
            }

        probe_x = np.linspace(-1.0, 1.0, m)
        probe_y = np.linspace(1.0, -1.0, m)
        full_batches = timed_pair(basis_csr, basis_t_csr, probe_x, probe_y)
        empty = sparse.csr_matrix((m, m))
        empty_batches = timed_pair(empty, empty.T.tocsr(), probe_x, probe_y)

        rows.append(
            {
                "checkpoint": checkpoint,
                "native_iterations": [int(current["iterations"]), int(following["iterations"])],
                "leaving_basis_position": leaving,
                "entering_column": entering,
                "basis_nnz": int(basis.nnz),
                "ftran_rhs_nnz": int(np.count_nonzero(ftran_rhs)),
                "btran_rhs_nnz": 1,
                "full_residual_pair_us_batches": full_batches,
                "empty_pair_us_batches": empty_batches,
                "full_residual_pair_us_median": statistics.median(full_batches),
                "observed_empty_subtracted_pair_us": statistics.median(full_batches)
                - statistics.median(empty_batches),
                "exact_entering_matches_across_tau_grid": exact_choice_matches,
                "matched_jacobi_minres_by_degree": degree_rows,
            }
        )

    pair_gate = CURRENT_PAIR_US * (1.0 - PROBE_WALL_GATE / CURRENT_SOLVE_SHARE)
    board_pair_gate = CURRENT_PAIR_US * (1.0 - BOARD_GAP / CURRENT_SOLVE_SHARE)
    full_pair_medians = [row["full_residual_pair_us_median"] for row in rows]
    observed_net_pairs = [row["observed_empty_subtracted_pair_us"] for row in rows]
    result = {
        "verdict": "KILL_MATCHED_JACOBI_AND_STALE_LU_KRYLOV",
        "fixture": "/tmp/lpsuite/lp_greenbea.mat",
        "shape": [m, n, int(raw_a.nnz)],
        "fixed_policy": {
            "eps": 2e-5,
            "checkpoints": list(CHECKPOINTS),
            "tol": 1e-8,
            "leaving_rule": 1,
            "expand": 1,
            "bfrt": 0,
        },
        "funding": {
            "current_comparable_pair_us": CURRENT_PAIR_US,
            "current_solve_share": CURRENT_SOLVE_SHARE,
            "probe_wall_gate": PROBE_WALL_GATE,
            "probe_pair_ceiling_us": pair_gate,
            "board_gap": BOARD_GAP,
            "board_pair_ceiling_us": board_pair_gate,
            "full_residual_pair_us_median_range": [
                min(full_pair_medians),
                max(full_pair_medians),
            ],
            "observed_empty_subtracted_pair_us_range": [
                min(observed_net_pairs),
                max(observed_net_pairs),
            ],
            "timing_interpretation": (
                "Full-call timings are implementation observations. Empty-call subtraction "
                "is unstable narrative context, not a lower bound or funding theorem."
            ),
        },
        "rows": rows,
        "summary": {
            "all_native_prefixes_exact_length": all(
                row["native_iterations"] == [row["checkpoint"], row["checkpoint"] + 1]
                for row in rows
            ),
            "exact_choice_matches": sum(
                row["exact_entering_matches_across_tau_grid"] for row in rows
            ),
            "approximate_choice_matches_by_degree": {
                str(degree): sum(
                    row["matched_jacobi_minres_by_degree"][str(degree)][
                        "approximate_entering_matches_across_tau_grid"
                    ]
                    for row in rows
                )
                for degree in RECORDED_DEGREES
            },
            "tau_choice_trials": len(rows) * len(TAUS),
            "recorded_degrees": list(RECORDED_DEGREES),
            "any_recorded_degree_authoritative": False,
        },
    }
    return result


def main() -> None:
    if os.environ.get("LINPROGX_DS_EXPORT_BASIS") is None:
        raise RuntimeError("set LINPROGX_DS_EXPORT_BASIS=1")
    OUT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(OUT_DIR, 0o700)
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    RESULTS.write_text(payload)
    os.chmod(RESULTS, 0o600)
    print(RESULTS)
    print(hashlib.sha256(payload.encode()).hexdigest())
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
