"""Read-only fresh-factor census for the remaining greenbea solve frontier.

The diagnostic reconstructs four actual native dual-simplex bases, measures
top-level off-diagonal ranks of their dense inverses under two deterministic
orderings, and applies the campaign's favorable inverse-update traffic model.
It also records the exact four-vCPU ideal-scaling arithmetic for an
ElasticDivide-style triangular-solve scheduler against the local live-FT DAG
measurements.

It does not modify solver state, write artifacts, use a competing solver, or
read any external solver source.  JSON is written to stdout only.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

# Set deterministic single-threaded linear algebra before importing NumPy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LINPROGX_DS_EXPORT_BASIS", "1")

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import splu

from experiments.greenbea_pivot_gap_probe import prepare
from experiments.rich_inverse_falsifier import basis_column, native_prefix, ruiz_scaled

CHECKPOINTS = (512, 1536, 3072, 4096)
RELATIVE_RANK_TOLERANCES = (2e-5, 1e-9)

CURRENT_PAIR_US = 47.425
CURRENT_SOLVE_SHARE = 0.3462
BOARD_GAP = 0.177013
PROBE_WALL_GATE = 0.20
FAVORABLE_BANDWIDTH_GBPS = 52.7
FAVORABLE_BYTES_PER_GENERATOR_COEFFICIENT = 16
TRAJECTORY_PIVOTS = 4399
COMPARABLE_EQUIVALENT_PAIRS = 4873

# Exact local live-FT measurements from lsa_level_sched_2026_07_19.md.
LIVE_U_LEVELS_MEAN = 57.68
LIVE_U_LEVELS_MAX = 154
LIVE_U_FTRAN_LABEL_DRIFT = 0.3418
LIVE_U_BTRAN_LABEL_DRIFT = 0.1353
L_OFFDIAGONAL_NNZ_RANGE = (1547.79, 1983.57)
L_SCHEDULE_COLD_REGRESSION = 0.2108
L_SCHEDULE_BSTAR_REGRESSION = 0.2285

# Exact local trajectory accounting from the retained solve-slice reports.
ORDINARY_REFACTORIZATIONS = 33
FTRAN_CALLS = 5313
BTRAN_CALLS = 4433
ELASTIC_SYNC_STAGES = 146


def numerical_rank(singular_values: np.ndarray, relative_tolerance: float) -> int:
    """Return strict relative numerical rank, matching the original screen."""
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular_values > singular_values[0] * relative_tolerance))


def top_level_rank_record(
    inverse: np.ndarray,
    row_order: np.ndarray,
    column_order: np.ndarray,
) -> dict[str, Any]:
    """Measure both off-diagonal blocks at the root of a binary hierarchy."""
    dimension = inverse.shape[0]
    midpoint = dimension // 2
    ordered = inverse[np.ix_(row_order, column_order)]
    blocks = (
        ordered[:midpoint, midpoint:],
        ordered[midpoint:, :midpoint],
    )
    singular_values = [np.linalg.svd(block, compute_uv=False) for block in blocks]

    by_tolerance: dict[str, Any] = {}
    for tolerance in RELATIVE_RANK_TOLERANCES:
        ranks = [numerical_rank(values, tolerance) for values in singular_values]
        rank_sum = sum(ranks)
        generator_coefficients = rank_sum * dimension
        bytes_per_pivot = FAVORABLE_BYTES_PER_GENERATOR_COEFFICIENT * generator_coefficients
        update_us_per_pivot = bytes_per_pivot / (FAVORABLE_BANDWIDTH_GBPS * 1e9) * 1e6
        equivalent_pair_charge_us = (
            update_us_per_pivot * TRAJECTORY_PIVOTS / COMPARABLE_EQUIVALENT_PAIRS
        )
        by_tolerance[format(tolerance, ".0e")] = {
            "block_ranks": ranks,
            "rank_sum": rank_sum,
            "top_level_generator_coefficients": generator_coefficients,
            "favorable_bytes_per_pivot": bytes_per_pivot,
            "favorable_update_us_per_pivot": update_us_per_pivot,
            "favorable_equivalent_pair_charge_us": equivalent_pair_charge_us,
            "exceeds_probe_pair_ceiling_before_other_costs": (
                equivalent_pair_charge_us > probe_pair_ceiling_us()
            ),
        }

    return {
        "split": [midpoint, dimension - midpoint],
        "by_relative_tolerance": by_tolerance,
    }


def board_pair_ceiling_us() -> float:
    """Largest replacement pair that closes the certified board gap."""
    return CURRENT_PAIR_US * (1.0 - BOARD_GAP / CURRENT_SOLVE_SHARE)


def probe_pair_ceiling_us() -> float:
    """Largest replacement pair that clears the campaign's 20% gate."""
    return CURRENT_PAIR_US * (1.0 - PROBE_WALL_GATE / CURRENT_SOLVE_SHARE)


def inverse_rank_census() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay actual bases and measure LU-permuted and RCM inverse ranks."""
    _, prepared = prepare()
    raw_matrix = prepared["A_scipy"].tocsc().astype(np.float64)
    scaled_matrix, _, _ = ruiz_scaled(raw_matrix)
    dimension, structural_columns = scaled_matrix.shape

    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        native = native_prefix(prepared, checkpoint)
        native_iterations = int(native["iterations"])
        if native_iterations != checkpoint:
            raise RuntimeError(
                f"checkpoint {checkpoint}: native replay stopped at {native_iterations}"
            )

        basis_index = np.asarray(native["basis"], dtype=np.int64)
        basis = sparse.hstack(
            [basis_column(scaled_matrix, int(column)) for column in basis_index],
            format="csc",
        )
        factor = splu(basis)
        exact_inverse = factor.solve(np.eye(dimension))

        # Rows of B^-1 correspond to basis columns; columns correspond to
        # constraint rows.  Use the corresponding SuperLU permutations.
        lu_row_order = np.argsort(factor.perm_c)
        lu_column_order = np.argsort(factor.perm_r)

        # A separate deterministic graph ordering for each inverse dimension.
        rcm_row_order = reverse_cuthill_mckee((basis.T @ basis).tocsr(), symmetric_mode=True)
        rcm_column_order = reverse_cuthill_mckee((basis @ basis.T).tocsr(), symmetric_mode=True)

        rows.append(
            {
                "checkpoint": checkpoint,
                "native_iterations": native_iterations,
                "basis_int64_sha256": hashlib.sha256(
                    basis_index.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "basis_nnz": int(basis.nnz),
                "structural_basic_columns": int(np.count_nonzero(basis_index < structural_columns)),
                "logical_basic_columns": int(np.count_nonzero(basis_index >= structural_columns)),
                "orderings": {
                    "lu_permuted": top_level_rank_record(
                        exact_inverse, lu_row_order, lu_column_order
                    ),
                    "independent_rcm": top_level_rank_record(
                        exact_inverse, rcm_row_order, rcm_column_order
                    ),
                },
            }
        )

    minimum_lu_charge = min(
        row["orderings"]["lu_permuted"]["by_relative_tolerance"]["2e-05"][
            "favorable_equivalent_pair_charge_us"
        ]
        for row in rows
    )
    summary = {
        "all_native_prefixes_exact_length": all(
            row["native_iterations"] == row["checkpoint"] for row in rows
        ),
        "minimum_lu_permuted_2e_5_top_level_pair_charge_us": minimum_lu_charge,
        "minimum_exceeds_probe_pair_ceiling_before_deeper_levels": (
            minimum_lu_charge > probe_pair_ceiling_us()
        ),
        "traffic_model": (
            "For the dynamically recompressed fixed-rank construction under test, charge "
            "only one 8-byte read and one 8-byte write per retained top-level generator "
            "coefficient per pivot. A lazy uncompressed update chain is excluded because "
            "it restores the previously killed product-form mechanism."
        ),
        "costs_deliberately_omitted_from_lower_bound": [
            "deeper hierarchy levels",
            "FTRAN and BTRAN applications",
            "rank-one input reads and arithmetic",
            "recompression",
            "factor-to-hierarchy refresh",
            "indices and metadata",
            "residual and Harris authority checks",
            "exact fallback",
            "final original-space certificate",
        ],
        "verdict": "KILL_DYNAMIC_HIERARCHICAL_WHOLE_INVERSE_IN_TESTED_ORDERINGS",
    }
    return rows, summary


def elasticdivide_census() -> dict[str, Any]:
    """Apply exact ideal-scaling economics to the local changing FT DAG."""
    probe_ceiling = probe_pair_ceiling_us()
    core_models: dict[str, Any] = {}
    for cores in (2, 3, 4):
        ideal_pair = CURRENT_PAIR_US / cores
        ideal_whole_wall_reduction = CURRENT_SOLVE_SHARE * (1.0 - 1.0 / cores)
        core_models[str(cores)] = {
            "ideal_pair_us_with_zero_overhead": ideal_pair,
            "ideal_whole_wall_reduction": ideal_whole_wall_reduction,
            "closes_board_gap_with_zero_overhead": ideal_whole_wall_reduction >= BOARD_GAP,
            "overhead_budget_under_20_percent_pair_gate_us": probe_ceiling - ideal_pair,
        }

    three_core_overhead_us = core_models["3"]["overhead_budget_under_20_percent_pair_gate_us"]
    return {
        "literature_mechanism_under_audit": (
            "exact stale-synchronous SpTRSV scheduling; reported gains grow with core count"
        ),
        "literature_scheduler_amortization_solves": [23, 54],
        "local_accounting": {
            "ordinary_refactorizations": ORDINARY_REFACTORIZATIONS,
            "pivots_per_refactorization": TRAJECTORY_PIVOTS / ORDINARY_REFACTORIZATIONS,
            "ftran_calls": FTRAN_CALLS,
            "btran_calls": BTRAN_CALLS,
            "solve_calls_per_refactorization": (
                (FTRAN_CALLS + BTRAN_CALLS) / ORDINARY_REFACTORIZATIONS
            ),
            "live_u_levels_mean": LIVE_U_LEVELS_MEAN,
            "live_u_levels_max": LIVE_U_LEVELS_MAX,
            "live_u_ftran_label_drift_fraction": LIVE_U_FTRAN_LABEL_DRIFT,
            "live_u_btran_label_drift_fraction": LIVE_U_BTRAN_LABEL_DRIFT,
            "elastic_sync_stages": ELASTIC_SYNC_STAGES,
            "three_core_overhead_budget_ns_per_sync_stage": (
                three_core_overhead_us * 1000.0 / ELASTIC_SYNC_STAGES
            ),
            "immutable_l_offdiagonal_nnz_range": list(L_OFFDIAGONAL_NNZ_RANGE),
            "existing_immutable_l_schedule_regression_fraction": [
                L_SCHEDULE_COLD_REGRESSION,
                L_SCHEDULE_BSTAR_REGRESSION,
            ],
        },
        "ideal_scaling_by_cores": core_models,
        "scheduler_construction_disposition": (
            "Fresh-factor construction could amortize over the observed solves per refactor, "
            "but live U' changes every pivot; its schedule requires patch/rebuild. Freezing U "
            "instead restores the killed product-form update chain."
        ),
        "execution_disposition": (
            "Two ideal cores cannot close the board gap. Three ideal cores leave only the "
            "reported overhead budget for synchronization, dynamic schedule maintenance, "
            "and nonparallel FT-chain work."
        ),
        "verdict": "KILL_ELASTIC_SCHEDULING_UNDER_FOUR_VCPU_V3",
    }


def run() -> dict[str, Any]:
    """Return the complete deterministic census."""
    rows, hierarchy_summary = inverse_rank_census()
    return {
        "verdict": "KILL_NO_FUNDED_FRESH_FACTOR_MECHANISM",
        "fixture": "/tmp/lpsuite/lp_greenbea.mat",
        "fixed_policy": {
            "checkpoints": list(CHECKPOINTS),
            "eps": 2e-5,
            "tol": 1e-8,
            "leaving_rule": 1,
            "expand": 1,
            "bfrt": 0,
            "relative_rank_tolerances": list(RELATIVE_RANK_TOLERANCES),
            "rank_rule": "count(s_i > s_0 * relative_tolerance)",
        },
        "funding": {
            "current_comparable_pair_us": CURRENT_PAIR_US,
            "current_solve_share": CURRENT_SOLVE_SHARE,
            "board_gap": BOARD_GAP,
            "board_pair_ceiling_us": board_pair_ceiling_us(),
            "probe_whole_wall_gate": PROBE_WALL_GATE,
            "probe_pair_ceiling_us": probe_pair_ceiling_us(),
            "favorable_bandwidth_gbps": FAVORABLE_BANDWIDTH_GBPS,
            "favorable_bytes_per_generator_coefficient": (
                FAVORABLE_BYTES_PER_GENERATOR_COEFFICIENT
            ),
            "trajectory_pivots": TRAJECTORY_PIVOTS,
            "comparable_equivalent_pairs": COMPARABLE_EQUIVALENT_PAIRS,
        },
        "hierarchical_inverse": {
            "enabling_algebra": {
                "basis_update": "B' = B E, E = I + (d - e_r) e_r^T, d = B^-1 a_q",
                "inverse_update": ("B'^-1 = B^-1 - ((d - e_r) / d_r) (e_r^T B^-1)"),
            },
            "rows": rows,
            "summary": hierarchy_summary,
        },
        "elasticdivide": elasticdivide_census(),
        "integrity": {
            "writes_files": False,
            "uses_network": False,
            "uses_external_solver": False,
            "changes_production_solver": False,
        },
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
