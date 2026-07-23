"""Oracle-pattern sparsified-whole-inverse falsifier for greenbea.

This diagnostic reconstructs four actual native simplex bases and their next
pivots.  At each basis it forms the exact dense inverse, retains the union of
the K largest-magnitude coefficients in every row and every column, and tests
that oracle sparsification against the actual next FTRAN/BTRAN right-hand
sides and Harris entering decision.

The oracle is deliberately more favorable than an implementable projected
rank-one update: it receives the best pattern and exact retained coefficients
at every sampled basis for free.  Maintenance is evaluated separately with a
host-calibrated traffic model; it is not a timing lower-bound theorem.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from experiments.greenbea_pivot_gap_probe import prepare

CHECKPOINTS = (512, 1536, 3072, 4096)
KEEP_COUNTS = (16, 32, 64, 128)
TAUS = np.linspace(0.0, 1e-8, 101)

EPS = 2e-5
CURRENT_PAIR_US = 47.425
CURRENT_SOLVE_SHARE = 0.3462
BOARD_GAP = 0.177013
PROBE_WALL_GATE = 0.20
FAVORABLE_BANDWIDTH_GBPS = 52.7
FAVORABLE_BYTES_PER_RETAINED_COEFFICIENT = 16
TRAJECTORY_PIVOTS = 4399
COMPARABLE_EQUIVALENT_PAIRS = 4873

OUT_DIR = Path("/tmp/rich-inverse-falsifier")
RESULTS = OUT_DIR / "results.json"


def ruiz_scaled(a: sparse.csc_matrix) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
    """Reproduce native DS 10-pass inf-norm plus one l2 Ruiz scaling."""
    m, n = a.shape
    row_scale = np.ones(m)
    col_scale = np.ones(n)
    row_norm = np.asarray(abs(a).max(axis=1).toarray()).ravel()
    nonzero_row_norm = row_norm[row_norm > 0.0]
    active = bool(
        nonzero_row_norm.size and nonzero_row_norm.max() / nonzero_row_norm.min() >= 100.0
    )

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


def basis_column(a: sparse.csc_matrix, column: int) -> sparse.csc_matrix:
    """Return a structural or logical basis column."""
    m, n = a.shape
    if column < n:
        return a[:, column]
    return sparse.csc_matrix(([1.0], ([column - n], [0])), shape=(m, 1))


def native_prefix(prepared: dict[str, Any], cap: int) -> dict[str, Any]:
    """Run the fixed native dual-simplex policy to an exact iteration cap."""
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


def harris_choice(
    alpha: np.ndarray,
    reduced_cost: np.ndarray,
    status: np.ndarray,
    basis_set: set[int],
    sigma: int,
    tau: float,
) -> int:
    """Replay the native admissibility and Harris two-pass entering choice."""
    candidates: list[tuple[int, float, float, float]] = []
    for column, alpha_j in enumerate(alpha):
        if column in basis_set or status[column] == 3 or abs(alpha_j) < 1e-9:
            continue
        admissible = (
            (status[column] == 0 and sigma * alpha_j < 0.0)
            or (status[column] == 1 and sigma * alpha_j > 0.0)
            or status[column] == 2
        )
        if not admissible:
            continue
        candidates.append(
            (
                column,
                float(alpha_j),
                (abs(reduced_cost[column]) + tau) / abs(alpha_j),
                abs(reduced_cost[column]) / abs(alpha_j),
            )
        )
    if not candidates:
        return -1

    theta_max = min(candidate[2] for candidate in candidates) + 1e-7
    entering = -1
    best_alpha = 0.0
    for column, alpha_j, _, plain_ratio in candidates:
        if plain_ratio <= theta_max and abs(alpha_j) > best_alpha:
            entering = column
            best_alpha = abs(alpha_j)
    return entering


def top_union_mask(inverse: np.ndarray, keep: int) -> np.ndarray:
    """Union NumPy-deterministic top-|.| entries in every row and column.

    ``argpartition`` is intentionally used because this reproduces the proposed
    policy's choice among tied exact-zero coefficients.  Those zero-valued
    locations still count as maintained pattern slots: a later projected
    rank-one update can make them nonzero.
    """
    dimension = inverse.shape[0]
    if not 0 < keep <= dimension:
        raise ValueError(f"invalid keep count {keep} for dimension {dimension}")

    absolute = np.abs(inverse)
    mask = np.zeros(inverse.shape, dtype=bool)
    row_order = np.argpartition(absolute, -keep, axis=1)[:, -keep:]
    row_indices = np.arange(dimension)[:, None]
    mask[row_indices, row_order] = True

    column_order = np.argpartition(absolute, -keep, axis=0)[-keep:, :]
    column_indices = np.arange(dimension)[None, :]
    mask[column_order, column_indices] = True
    return mask


def relative_residual(
    matrix: sparse.csc_matrix,
    estimate: np.ndarray,
    rhs: np.ndarray,
    *,
    transpose: bool,
) -> float:
    image = matrix.T @ estimate if transpose else matrix @ estimate
    return float(np.linalg.norm(rhs - image) / max(np.linalg.norm(rhs), 1e-300))


def favorable_update_model(nnz: float) -> dict[str, float]:
    """Convert projected-update coefficient traffic into pair-equivalent cost."""
    bytes_moved = FAVORABLE_BYTES_PER_RETAINED_COEFFICIENT * nnz
    update_us = bytes_moved / (FAVORABLE_BANDWIDTH_GBPS * 1e9) * 1e6
    updates_per_equivalent_pair = TRAJECTORY_PIVOTS / COMPARABLE_EQUIVALENT_PAIRS
    pair_charge_us = update_us * updates_per_equivalent_pair
    pair_fraction = pair_charge_us / CURRENT_PAIR_US
    pool_reduction = 1.0 - pair_fraction
    return {
        "retained_coefficients": nnz,
        "modeled_bytes_per_pivot": bytes_moved,
        "modeled_update_us_per_pivot": update_us,
        "updates_per_equivalent_pair": updates_per_equivalent_pair,
        "modeled_equivalent_pair_charge_us": pair_charge_us,
        "modeled_fraction_of_current_pair": pair_fraction,
        "modeled_solve_pool_reduction": pool_reduction,
        "modeled_whole_wall_reduction": CURRENT_SOLVE_SHARE * pool_reduction,
    }


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
    big_m = 1e5 * max(
        1.0,
        float(np.max(np.abs(scaled_b))),
        float(np.max(finite_bounds)),
    )

    rows: list[dict[str, Any]] = []
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

        basis = sparse.hstack(
            [basis_column(scaled_a, int(column)) for column in basis_index],
            format="csc",
        )
        factor = splu(basis)
        exact_inverse = factor.solve(np.eye(m))

        ftran_rhs = basis_column(scaled_a, entering).toarray().ravel()
        btran_rhs = np.zeros(m)
        btran_rhs[leaving] = 1.0
        exact_ftran = exact_inverse @ ftran_rhs
        exact_btran = exact_inverse.T @ btran_rhs

        basis_set = set(int(value) for value in basis_index)
        status = np.asarray(current["bound_status"], dtype=np.int8)
        c_basis = np.asarray([scaled_c[j] if j < n else 0.0 for j in basis_index])
        dual = factor.solve(c_basis, trans="T")
        reduced_cost = np.r_[scaled_c - scaled_a.T @ dual, -dual]
        reduced_cost[basis_index] = 0.0

        nonbasic_x = np.zeros(n + m)
        for column in range(n):
            if column in basis_set:
                continue
            if status[column] == 0:
                nonbasic_x[column] = (
                    scaled_lo[column]
                    if np.isfinite(scaled_lo[column])
                    else scaled_hi[column] - big_m
                )
            elif status[column] == 1:
                nonbasic_x[column] = (
                    scaled_hi[column]
                    if np.isfinite(scaled_hi[column])
                    else scaled_lo[column] + big_m
                )
            elif status[column] == 2:
                nonbasic_x[column] = 0.0
            elif status[column] == 3:
                nonbasic_x[column] = scaled_lo[column]
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
            raise RuntimeError(f"checkpoint {checkpoint}: changed row is not infeasible")

        exact_alpha = np.r_[scaled_a.T @ exact_btran, exact_btran]
        exact_matches = sum(
            harris_choice(exact_alpha, reduced_cost, status, basis_set, sigma, float(tau))
            == entering
            for tau in TAUS
        )

        keep_rows: dict[str, Any] = {}
        for keep in KEEP_COUNTS:
            mask = top_union_mask(exact_inverse, keep)
            sparse_inverse = sparse.csr_matrix(
                (
                    exact_inverse[mask],
                    np.nonzero(mask),
                ),
                shape=exact_inverse.shape,
            )
            approximate_ftran = np.asarray(sparse_inverse @ ftran_rhs).ravel()
            approximate_btran = np.asarray(sparse_inverse.T @ btran_rhs).ravel()
            approximate_alpha = np.r_[scaled_a.T @ approximate_btran, approximate_btran]
            approximate_matches = sum(
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
            keep_rows[str(keep)] = {
                "retained_nnz": int(sparse_inverse.nnz),
                "numerically_nonzero_retained_coefficients_at_capture": int(
                    np.count_nonzero(exact_inverse[mask])
                ),
                "retained_density": float(sparse_inverse.nnz / (m * m)),
                "ftran_relative_residual": relative_residual(
                    basis,
                    approximate_ftran,
                    ftran_rhs,
                    transpose=False,
                ),
                "btran_relative_residual": relative_residual(
                    basis,
                    approximate_btran,
                    btran_rhs,
                    transpose=True,
                ),
                "ftran_relative_solution_error": float(
                    np.linalg.norm(approximate_ftran - exact_ftran)
                    / max(np.linalg.norm(exact_ftran), 1e-300)
                ),
                "btran_relative_solution_error": float(
                    np.linalg.norm(approximate_btran - exact_btran)
                    / max(np.linalg.norm(exact_btran), 1e-300)
                ),
                "approximate_entering_matches_across_tau_grid": approximate_matches,
                "tau_grid_size": len(TAUS),
            }

        rows.append(
            {
                "checkpoint": checkpoint,
                "native_iterations": [int(current["iterations"]), int(following["iterations"])],
                "changed_basis_positions": int(changed.size),
                "leaving_basis_position": leaving,
                "entering_column": entering,
                "basis_nnz": int(basis.nnz),
                "ftran_rhs_nnz": int(np.count_nonzero(ftran_rhs)),
                "btran_rhs_nnz": 1,
                "exact_inverse_inf_residual": float(
                    np.linalg.norm(basis @ exact_inverse - np.eye(m), ord=np.inf)
                ),
                "exact_ftran_relative_residual": relative_residual(
                    basis,
                    exact_ftran,
                    ftran_rhs,
                    transpose=False,
                ),
                "exact_btran_relative_residual": relative_residual(
                    basis,
                    exact_btran,
                    btran_rhs,
                    transpose=True,
                ),
                "exact_entering_matches_across_tau_grid": exact_matches,
                "oracle_sparsification_by_keep_count": keep_rows,
            }
        )

    board_pair_ceiling = CURRENT_PAIR_US * (1.0 - BOARD_GAP / CURRENT_SOLVE_SHARE)
    probe_pair_ceiling = CURRENT_PAIR_US * (1.0 - PROBE_WALL_GATE / CURRENT_SOLVE_SHARE)
    trials = len(CHECKPOINTS) * len(TAUS)
    matches_by_keep = {
        str(keep): sum(
            row["oracle_sparsification_by_keep_count"][str(keep)][
                "approximate_entering_matches_across_tau_grid"
            ]
            for row in rows
        )
        for keep in KEEP_COUNTS
    }
    funding_by_keep: dict[str, Any] = {}
    for keep in KEEP_COUNTS:
        retained = [
            row["oracle_sparsification_by_keep_count"][str(keep)]["retained_nnz"] for row in rows
        ]
        average_model = favorable_update_model(float(np.mean(retained)))
        maximum_model = favorable_update_model(float(max(retained)))
        decision_failure_fraction = 1.0 - matches_by_keep[str(keep)] / trials
        average_with_decision_fallback_us = (
            average_model["modeled_equivalent_pair_charge_us"]
            + decision_failure_fraction * CURRENT_PAIR_US
        )
        maximum_with_decision_fallback_us = (
            maximum_model["modeled_equivalent_pair_charge_us"]
            + decision_failure_fraction * CURRENT_PAIR_US
        )
        funding_by_keep[str(keep)] = {
            "retained_nnz_by_checkpoint": retained,
            "retained_nnz_mean": float(np.mean(retained)),
            "retained_nnz_max": max(retained),
            "favorable_average_update_model": average_model,
            "favorable_maximum_update_model": maximum_model,
            "observed_decision_failure_fraction": decision_failure_fraction,
            "favorable_average_charge_with_observed_decision_fallback_us": (
                average_with_decision_fallback_us
            ),
            "favorable_maximum_charge_with_observed_decision_fallback_us": (
                maximum_with_decision_fallback_us
            ),
            "favorable_average_with_fallback_modeled_whole_wall_reduction": (
                CURRENT_SOLVE_SHARE * (1.0 - average_with_decision_fallback_us / CURRENT_PAIR_US)
            ),
            "average_model_funds_board_gap": (
                average_model["modeled_equivalent_pair_charge_us"] <= board_pair_ceiling
            ),
            "average_model_funds_20_percent_probe_gate": (
                average_model["modeled_equivalent_pair_charge_us"] <= probe_pair_ceiling
            ),
            "maximum_model_funds_board_gap": (
                maximum_model["modeled_equivalent_pair_charge_us"] <= board_pair_ceiling
            ),
            "maximum_model_funds_20_percent_probe_gate": (
                maximum_model["modeled_equivalent_pair_charge_us"] <= probe_pair_ceiling
            ),
            "average_model_with_decision_fallback_funds_board_gap": (
                average_with_decision_fallback_us <= board_pair_ceiling
            ),
            "average_model_with_decision_fallback_funds_20_percent_probe_gate": (
                average_with_decision_fallback_us <= probe_pair_ceiling
            ),
            "maximum_model_with_decision_fallback_funds_board_gap": (
                maximum_with_decision_fallback_us <= board_pair_ceiling
            ),
            "maximum_model_with_decision_fallback_funds_20_percent_probe_gate": (
                maximum_with_decision_fallback_us <= probe_pair_ceiling
            ),
        }
    result = {
        "verdict": "KILL_FIXED_PATTERN_SPARSIFIED_WHOLE_INVERSE_IN_TESTED_SCOPE",
        "fixture": "/tmp/lpsuite/lp_greenbea.mat",
        "shape": [m, n, int(raw_a.nnz)],
        "numpy_version": np.__version__,
        "fixed_policy": {
            "eps": EPS,
            "checkpoints": list(CHECKPOINTS),
            "keep_counts": list(KEEP_COUNTS),
            "tol": 1e-8,
            "leaving_rule": 1,
            "expand": 1,
            "bfrt": 0,
            "tau_grid": [float(TAUS[0]), float(TAUS[-1]), len(TAUS)],
            "pattern_rule": (
                "At each captured exact inverse, union NumPy argpartition's top-K "
                "absolute entries in every row and every column; retain exact "
                "coefficients. Exact-zero selected locations remain charged pattern "
                "slots because projected rank-one updates can make them nonzero."
            ),
        },
        "funding": {
            "current_comparable_pair_us": CURRENT_PAIR_US,
            "current_solve_share": CURRENT_SOLVE_SHARE,
            "board_gap": BOARD_GAP,
            "board_required_solve_pool_reduction": BOARD_GAP / CURRENT_SOLVE_SHARE,
            "board_pair_ceiling_us": board_pair_ceiling,
            "probe_wall_gate": PROBE_WALL_GATE,
            "probe_required_solve_pool_reduction": PROBE_WALL_GATE / CURRENT_SOLVE_SHARE,
            "probe_pair_ceiling_us": probe_pair_ceiling,
            "favorable_bandwidth_gbps": FAVORABLE_BANDWIDTH_GBPS,
            "favorable_bytes_per_retained_coefficient": (FAVORABLE_BYTES_PER_RETAINED_COEFFICIENT),
            "trajectory_pivots": TRAJECTORY_PIVOTS,
            "comparable_equivalent_pairs": COMPARABLE_EQUIVALENT_PAIRS,
            "updates_per_equivalent_pair": TRAJECTORY_PIVOTS / COMPARABLE_EQUIVALENT_PAIRS,
            "interpretation": (
                "Host-calibrated favorable model only: every pivot receives one "
                "projected rank-one inverse update, charged only one 8-byte read and "
                "one 8-byte write per retained coefficient. Pattern refresh, rank-one "
                "factor reads, arithmetic, sparse indices, applications, residuals, "
                "certificates are free. Per-pivot update cost is multiplied by "
                "4399/4873 to charge it across the audited comparable solve-pair "
                "population. The separate decision-fallback model charges an exact "
                "pair only at the observed Harris failure fraction and still ignores "
                "residual-triggered fallback. These four samples are a favorable "
                "proxy, not a full-trace average or a lower-bound theorem."
            ),
            "by_keep_count": funding_by_keep,
        },
        "rows": rows,
        "summary": {
            "all_native_prefixes_exact_length": all(
                row["native_iterations"] == [row["checkpoint"], row["checkpoint"] + 1]
                for row in rows
            ),
            "all_adjacent_bases_have_one_changed_position": all(
                row["changed_basis_positions"] == 1 for row in rows
            ),
            "exact_choice_matches": sum(
                row["exact_entering_matches_across_tau_grid"] for row in rows
            ),
            "approximate_choice_matches_by_keep_count": matches_by_keep,
            "tau_choice_trials": trials,
            "fully_authoritative_keep_counts": [
                keep for keep in KEEP_COUNTS if matches_by_keep[str(keep)] == trials
            ],
            "residual_qualified_checkpoints_by_keep_count": {
                str(keep): sum(
                    row["oracle_sparsification_by_keep_count"][str(keep)]["ftran_relative_residual"]
                    <= EPS
                    and row["oracle_sparsification_by_keep_count"][str(keep)][
                        "btran_relative_residual"
                    ]
                    <= EPS
                    for row in rows
                )
                for keep in KEEP_COUNTS
            },
            "keep_counts_funding_board_under_favorable_average_model": [
                keep
                for keep in KEEP_COUNTS
                if funding_by_keep[str(keep)]["average_model_funds_board_gap"]
            ],
            "keep_counts_funding_probe_gate_under_favorable_average_model": [
                keep
                for keep in KEEP_COUNTS
                if funding_by_keep[str(keep)]["average_model_funds_20_percent_probe_gate"]
            ],
            "keep_counts_funding_board_with_observed_decision_fallback": [
                keep
                for keep in KEEP_COUNTS
                if funding_by_keep[str(keep)][
                    "average_model_with_decision_fallback_funds_board_gap"
                ]
            ],
            "keep_counts_funding_probe_gate_with_observed_decision_fallback": [
                keep
                for keep in KEEP_COUNTS
                if funding_by_keep[str(keep)][
                    "average_model_with_decision_fallback_funds_20_percent_probe_gate"
                ]
            ],
        },
    }
    return result


def main() -> None:
    os.environ.setdefault("LINPROGX_DS_EXPORT_BASIS", "1")
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
