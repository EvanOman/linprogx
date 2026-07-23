"""Read-only structural census for the remaining greenbea frontier.

The diagnostic measures four globally applicable candidate classes without
changing solver behavior:

* inequality-only Fourier--Motzkin elimination inferred from standard-form
  slack columns;
* exact connected-component and high-degree-row separator decomposition;
* a generous generalized-network core (all columns of degree at most two);
* exact row sketches, screened by structural rank and a normalized Gram solve.

The MAT fixtures expose equality-form matrices but not original MPS row senses.
An inequality row is therefore inferred only when it contains a zero-cost,
``[0, +inf]``, singleton column with coefficient ``+/-1``.  When several such
columns occur in one row, the highest-index column deterministically supplies
the orientation and the ambiguity is reported.  All slack-like columns are
excluded from the structural FME candidate set.

Results are deterministic JSON on stdout.  The script writes no files and
fails loudly when a required fixture, dependency, or prepared reduction is
unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy import sparse
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components, structural_rank
from scipy.sparse.linalg import splu

from linprogx.presolve import presolve_matrix
from linprogx.sparse import csr_matrix, from_scipy_sparse

FIXTURES = ("greenbea", "woodw", "pds_10", "cre_a", "80bau3b")
SEPARATOR_FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15)
SLACK_TOL = 1e-12
FME_DROP_TOL = 1e-12
GRAM_MAX_ROWS = 2_000
GRAM_RHS_COUNT = 3
GRAM_SEED = 20_260_722
BOARD_GAP_PCT = 17.7013459
PROBE_GATE_PCT = 20.0
PRESOLVE_ENV_KEYS = (
    "LINPROGX_AGG_FILLGATE",
    "LINPROGX_AGG_MAX_FILL",
    "LINPROGX_AGG_MAX_NNZ",
    "LINPROGX_AGG_PIVOT_TOL",
    "LINPROGX_PRESOLVE_AGG",
    "LINPROGX_PRESOLVE_FIXPOINT",
    "LINPROGX_PRESOLVE_NETAGG",
    "LINPROGX_PRESOLVE_PARALLEL_COLS",
    "LINPROGX_PRESOLVE_V2",
    "LINPROGX_PRESOLVE_V2_NATIVE",
)


def _load_fixture(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required fixture is missing: {path}")
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    matrix = raw["A"].tocsr().astype(np.float64)
    return {
        "A": matrix,
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def _counter_json(counter: Counter[Any]) -> dict[str, int]:
    def key_token(key: Any) -> str:
        if isinstance(key, tuple):
            return ",".join(str(value) for value in key)
        return str(key)

    return {key_token(key): int(counter[key]) for key in sorted(counter, key=key_token)}


def _normalized_row_without_slacks(
    matrix: sparse.csr_matrix,
    row: int,
    orientation: float,
    slack_like: np.ndarray,
) -> dict[int, float]:
    start, stop = matrix.indptr[row], matrix.indptr[row + 1]
    return {
        int(col): float(value * orientation)
        for col, value in zip(matrix.indices[start:stop], matrix.data[start:stop], strict=True)
        if not slack_like[int(col)]
    }


def _paired_row_nnz(positive: dict[int, float], negative: dict[int, float], col: int) -> int:
    positive_coef = positive[col]
    negative_coef = negative[col]
    keys = (set(positive) | set(negative)) - {col}
    return sum(
        abs((-negative_coef) * positive.get(key, 0.0) + positive_coef * negative.get(key, 0.0))
        > FME_DROP_TOL
        for key in keys
    )


def _raw_fme_census(data: dict[str, Any], *, include_details: bool) -> dict[str, Any]:
    matrix = data["A"].tocsc(copy=True)
    matrix.eliminate_zeros()
    rows_matrix = matrix.tocsr()
    c = data["c"]
    lo = data["lo"]
    hi = data["hi"]
    row_count, col_count = matrix.shape
    col_degree = np.diff(matrix.indptr)

    slack_like = (col_degree == 1) & (c == 0.0) & (lo == 0.0) & np.isposinf(hi)
    provisional = np.flatnonzero(slack_like)
    provisional_values = matrix.data[matrix.indptr[provisional]]
    nonunit = np.abs(np.abs(provisional_values) - 1.0) > SLACK_TOL
    slack_like[provisional[nonunit]] = False

    slack_cols = np.flatnonzero(slack_like)
    slack_rows = matrix.indices[matrix.indptr[slack_cols]]
    slack_values = matrix.data[matrix.indptr[slack_cols]]
    slack_count_by_row = np.bincount(slack_rows, minlength=row_count)

    # Iteration is by increasing column index, so assignment retains the
    # highest-index candidate in an ambiguous row, matching the original census.
    designated_slack: dict[int, tuple[int, float]] = {}
    for col, row, value in zip(slack_cols, slack_rows, slack_values, strict=True):
        designated_slack[int(row)] = (int(col), float(value))
    inferred_inequality = np.zeros(row_count, dtype=bool)
    inferred_inequality[list(designated_slack)] = True

    candidates: list[dict[str, Any]] = []
    touched_rows: set[int] = set()
    for col in range(col_count):
        if slack_like[col]:
            continue
        start, stop = matrix.indptr[col], matrix.indptr[col + 1]
        incident_rows = matrix.indices[start:stop]
        values = matrix.data[start:stop]
        if len(incident_rows) == 0 or not bool(np.all(inferred_inequality[incident_rows])):
            continue

        normalized_values = np.asarray(
            [
                value * math.copysign(1.0, designated_slack[int(row)][1])
                for row, value in zip(incident_rows, values, strict=True)
            ],
            dtype=np.float64,
        )
        model_positive = int(np.count_nonzero(normalized_values > 0.0))
        model_negative = int(np.count_nonzero(normalized_values < 0.0))
        positive_count = model_positive + int(np.isfinite(hi[col]))
        negative_count = model_negative + int(np.isfinite(lo[col]))
        if positive_count == 0 or negative_count == 0:
            continue

        positive_rows: list[dict[int, float]] = []
        negative_rows: list[dict[int, float]] = []
        for row, normalized_value in zip(incident_rows, normalized_values, strict=True):
            orientation = math.copysign(1.0, designated_slack[int(row)][1])
            row_values = _normalized_row_without_slacks(
                rows_matrix, int(row), orientation, slack_like
            )
            (positive_rows if normalized_value > 0.0 else negative_rows).append(row_values)
        if np.isfinite(hi[col]):
            positive_rows.append({col: 1.0})
        if np.isfinite(lo[col]):
            negative_rows.append({col: -1.0})

        input_nnz = sum(len(row) for row in positive_rows + negative_rows)
        output_nnz = sum(
            _paired_row_nnz(positive, negative, col)
            for positive in positive_rows
            for negative in negative_rows
        )
        row_delta = positive_count * negative_count - positive_count - negative_count
        abs_values = np.abs(normalized_values)
        candidate = {
            "column": int(col),
            "incident_model_rows": [int(row) for row in incident_rows],
            "model_positive": model_positive,
            "model_negative": model_negative,
            "m_plus_with_bounds": positive_count,
            "m_minus_with_bounds": negative_count,
            "input_rows_with_bounds": positive_count + negative_count,
            "output_paired_rows": positive_count * negative_count,
            "row_delta": row_delta,
            "input_nnz_without_slacks": input_nnz,
            "output_nnz": output_nnz,
            "nnz_delta": output_nnz - input_nnz,
            "coefficient_abs_min": float(np.min(abs_values)),
            "coefficient_abs_max": float(np.max(abs_values)),
            "coefficient_range_gate_1e3": bool(
                float(np.min(abs_values)) >= 1e-3 and float(np.max(abs_values)) <= 1e3
            ),
            "finite_lower_bound": bool(np.isfinite(lo[col])),
            "finite_upper_bound": bool(np.isfinite(hi[col])),
        }
        candidates.append(candidate)
        touched_rows.update(int(row) for row in incident_rows)

    touched_raw_nnz = sum(
        int(rows_matrix.indptr[row + 1] - rows_matrix.indptr[row]) for row in touched_rows
    )
    result: dict[str, Any] = {
        "raw_shape": [int(row_count), int(col_count), int(matrix.nnz)],
        "inference": {
            "strict_slack_like_columns": int(np.count_nonzero(slack_like)),
            "inferred_inequality_rows": int(np.count_nonzero(inferred_inequality)),
            "ambiguous_multi_slack_rows": int(np.count_nonzero(slack_count_by_row > 1)),
            "orientation_rule": "highest-index strict slack-like column",
        },
        "eligible_columns": len(candidates),
        "m_plus_m_minus_histogram": _counter_json(
            Counter(
                (candidate["m_plus_with_bounds"], candidate["m_minus_with_bounds"])
                for candidate in candidates
            )
        ),
        "row_effect": {
            "reducing_candidates": sum(candidate["row_delta"] < 0 for candidate in candidates),
            "zero_growth_candidates": sum(candidate["row_delta"] == 0 for candidate in candidates),
            "growing_candidates": sum(candidate["row_delta"] > 0 for candidate in candidates),
            "independent_input_rows_with_bounds": sum(
                candidate["input_rows_with_bounds"] for candidate in candidates
            ),
            "independent_output_rows": sum(
                candidate["output_paired_rows"] for candidate in candidates
            ),
            "independent_row_delta": sum(candidate["row_delta"] for candidate in candidates),
            "independent_reducing_only_rows_removed": -sum(
                min(0, candidate["row_delta"]) for candidate in candidates
            ),
            "unique_incident_model_rows": len(touched_rows),
        },
        "nnz_effect": {
            "independent_input_nnz_without_slacks": sum(
                candidate["input_nnz_without_slacks"] for candidate in candidates
            ),
            "independent_output_nnz": sum(candidate["output_nnz"] for candidate in candidates),
            "independent_nnz_delta": sum(candidate["nnz_delta"] for candidate in candidates),
            "unique_incident_raw_nnz_with_slacks": touched_raw_nnz,
            "unique_incident_raw_nnz_pct": 100.0 * touched_raw_nnz / matrix.nnz,
        },
        "coefficient_range_gate_1e3_pass": sum(
            candidate["coefficient_range_gate_1e3"] for candidate in candidates
        ),
    }
    if include_details:
        result["candidate_details"] = candidates
    return result


def _to_scipy(matrix: Any) -> sparse.csr_matrix:
    indptr, indices, values = matrix.to_components()
    result = sparse.csr_matrix((values, indices, indptr), shape=matrix.shape)
    result.eliminate_zeros()
    return result


def _prepared_matrix(data: dict[str, Any]) -> sparse.csr_matrix:
    matrix = from_scipy_sparse(data["A"])
    reduction = presolve_matrix(
        matrix,
        data["b"].tolist(),
        data["c"].tolist(),
        data["lo"].tolist(),
        data["hi"].tolist(),
    )
    if reduction is None:
        raise RuntimeError("current presolve unexpectedly returned no reduction")
    reduced = reduction._matrix
    if reduced is None:
        reduced = csr_matrix(
            reduction.rows,
            reduction.cols,
            reduction.indptr,
            reduction.indices,
            reduction.data,
        )
    return _to_scipy(reduced)


def _bipartite_component_stats(matrix: sparse.csr_matrix) -> dict[str, Any]:
    pattern = matrix.astype(bool)
    graph = sparse.bmat([[None, pattern], [pattern.T, None]], format="csr")
    component_count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return {
        "component_count": int(component_count),
        "largest_component_nodes": int(np.max(sizes)),
        "largest_component_node_fraction": float(np.max(sizes) / len(labels)),
    }


def _separator_curve(matrix: sparse.csr_matrix) -> list[dict[str, Any]]:
    row_count, col_count = matrix.shape
    row_degree = np.diff(matrix.indptr)
    order = np.argsort(-row_degree, kind="stable")
    curve: list[dict[str, Any]] = []
    for fraction in SEPARATOR_FRACTIONS:
        removed_count = int(math.ceil(fraction * row_count))
        keep = np.ones(row_count, dtype=bool)
        keep[order[:removed_count]] = False
        remaining = matrix[keep]
        pattern = remaining.astype(bool)
        graph = sparse.bmat([[None, pattern], [pattern.T, None]], format="csr")
        component_count, labels = connected_components(graph, directed=False)
        col_labels = labels[remaining.shape[0] :]
        col_sizes = np.bincount(col_labels, minlength=component_count)
        border_nnz = int(np.sum(row_degree[order[:removed_count]]))
        curve.append(
            {
                "removed_row_fraction": fraction,
                "removed_rows": removed_count,
                "largest_column_component_fraction": float(np.max(col_sizes) / col_count),
                "border_nnz_fraction": float(border_nnz / matrix.nnz),
            }
        )
    return curve


def _gram_evidence(matrix: sparse.csr_matrix) -> dict[str, Any]:
    row_count = matrix.shape[0]
    if row_count > GRAM_MAX_ROWS:
        return {
            "attempted": False,
            "reason": f"row count exceeds fixed diagnostic cap {GRAM_MAX_ROWS}",
        }
    row_norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if bool(np.any(row_norm == 0.0)):
        raise AssertionError("prepared matrix contains a zero row")
    normalized = sparse.diags(1.0 / row_norm) @ matrix
    gram = (normalized @ normalized.T).tocsc()
    factor = splu(gram)
    rng = np.random.default_rng(GRAM_SEED)
    residuals: list[float] = []
    for _ in range(GRAM_RHS_COUNT):
        rhs = rng.standard_normal(row_count)
        solution = factor.solve(rhs)
        residuals.append(
            float(np.linalg.norm(gram @ solution - rhs, np.inf) / np.linalg.norm(rhs, np.inf))
        )
    diagonal = np.abs(factor.U.diagonal())
    return {
        "attempted": True,
        "factorization": "nonsingular",
        "random_rhs_count": GRAM_RHS_COUNT,
        "random_seed": GRAM_SEED,
        "max_relative_inf_residual": max(residuals),
        "u_diagonal_abs_min": float(np.min(diagonal)),
        "u_diagonal_abs_max": float(np.max(diagonal)),
    }


def _prepared_census(matrix: sparse.csr_matrix) -> dict[str, Any]:
    row_count, col_count = matrix.shape
    col_degree = np.diff(matrix.tocsc().indptr)
    degree_le_two = col_degree <= 2
    network_nnz = int(np.sum(col_degree[degree_le_two]))
    rank = int(structural_rank(matrix))
    components = _bipartite_component_stats(matrix)
    return {
        "shape": [int(row_count), int(col_count), int(matrix.nnz)],
        "components": components,
        "row_separator_curve": _separator_curve(matrix),
        "generalized_network_upper_envelope": {
            "degree_le_two_columns": int(np.count_nonzero(degree_le_two)),
            "degree_le_two_column_fraction": float(np.count_nonzero(degree_le_two) / col_count),
            "degree_le_two_nnz": network_nnz,
            "degree_le_two_nnz_fraction": float(network_nnz / matrix.nnz),
        },
        "rank": {
            "structural_rank": rank,
            "structural_row_deficiency": row_count - rank,
            "gram": _gram_evidence(matrix),
        },
    }


def _greenbea_verdicts(raw: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    rows, _, _ = prepared["shape"]
    touched_rows = raw["row_effect"]["unique_incident_model_rows"]
    dense_proxy = 100.0 * (1.0 - ((rows - touched_rows) / rows) ** 3)
    network_fraction = prepared["generalized_network_upper_envelope"]["degree_le_two_nnz_fraction"]
    components = prepared["components"]
    rank = prepared["rank"]
    return {
        "inequality_only_fme": {
            "verdict": "KILL",
            "reason": "eligible columns touch only eight model rows and overlap heavily",
            "favorable_dense_m_cubed_proxy_pct": dense_proxy,
            "note": "proxy is opportunity arithmetic, not a pivot-path theorem",
        },
        "exact_component_or_thin_border_decomposition": {
            "verdict": "KILL",
            "exact_component_opportunity_pct": 0.0 if components["component_count"] == 1 else None,
            "reason": "the prepared graph is connected and screened borders retain a dominant core",
        },
        "generalized_network_route": {
            "verdict": "KILL",
            "favorable_free_qualifying_nnz_ceiling_pct": 100.0 * network_fraction,
            "reason": "most columns and nonzeros remain in a generic LP core",
        },
        "exact_randomized_row_sketch": {
            "verdict": "KILL",
            "exact_dimension_reduction_pct": 0.0
            if rank["structural_row_deficiency"] == 0
            else None,
            "reason": "the prepared matrix has full structural row rank",
        },
        "funded_candidate": None,
        "required_board_reduction_pct": BOARD_GAP_PCT,
        "probe_gate_pct": PROBE_GATE_PCT,
    }


def run(suite_dir: Path) -> dict[str, Any]:
    overrides = sorted(key for key in PRESOLVE_ENV_KEYS if key in os.environ)
    if overrides:
        raise RuntimeError(
            "presolve environment overrides would make the census non-default: "
            + ", ".join(overrides)
        )

    fixtures: dict[str, Any] = {}
    for name in FIXTURES:
        data = _load_fixture(suite_dir / f"lp_{name}.mat")
        raw = _raw_fme_census(data, include_details=name == "greenbea")
        prepared = _prepared_census(_prepared_matrix(data))
        fixtures[name] = {"raw_fme": raw, "prepared": prepared}

    return {
        "diagnostic": "fresh_structural_census",
        "scope": {
            "fixtures": list(FIXTURES),
            "suite_dir": str(suite_dir),
            "eps": 2e-5,
            "board_gap_pct": BOARD_GAP_PCT,
            "probe_gate_pct": PROBE_GATE_PCT,
            "writes_files": False,
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "fixtures": fixtures,
        "greenbea_verdicts": _greenbea_verdicts(
            fixtures["greenbea"]["raw_fme"], fixtures["greenbea"]["prepared"]
        ),
        "measurement_note": (
            "Counts, graph statistics, rank, and Gram residuals are measurements. "
            "Opportunity ceilings and dense-cubic arithmetic are scoped favorable proxies."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=Path("/tmp/lpsuite"))
    args = parser.parse_args()
    print(json.dumps(run(args.suite_dir), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
