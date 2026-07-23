"""Two-way basis-transfer falsifier for greenbea.

HiGHS remains a black box: this uses only highspy's public getBasis/setBasis
interfaces and runtime iteration-limit/log output. The linprogx side requires
the diagnostic-only C hook enabled by LINPROGX_DS_WARM_START=1 and exports its
final basis when LINPROGX_DS_EXPORT_BASIS=1.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.greenbea_pivot_gap_probe import _pass_highs_model, prepare
from linprogx.presolve import postsolve_x

OUT_DIR = Path("/tmp/greenbea-basis-transfer")
RESULTS = OUT_DIR / "results.json"
PHASE1_BOUNDARY = 1_655
PHASE1_PLUS_ONE = 1_656
OBJECTIVE_REL_LIMIT = 2e-5


def _status_name(value: Any) -> str:
    return str(value).rsplit(".", 1)[-1]


def _basis_summary(basis: Any) -> dict[str, Any]:
    return {
        "valid": bool(basis.valid),
        "col_status": dict(sorted(Counter(map(_status_name, basis.col_status)).items())),
        "row_status": dict(sorted(Counter(map(_status_name, basis.row_status)).items())),
    }


def _basis_columns(basis: Any, num_cols: int, num_rows: int) -> list[int]:
    import highspy  # ty: ignore[unresolved-import]

    columns = [
        j for j, status in enumerate(basis.col_status) if status == highspy.HighsBasisStatus.kBasic
    ]
    columns.extend(
        num_cols + i
        for i, status in enumerate(basis.row_status)
        if status == highspy.HighsBasisStatus.kBasic
    )
    if len(columns) != num_rows:
        raise RuntimeError(f"HiGHS basis has {len(columns)} basics; expected {num_rows}")
    return columns


def _basis_bound_status(basis: Any, num_cols: int, num_rows: int) -> list[int]:
    """Map public HiGHS statuses to linprogx's diagnostic status codes."""
    import highspy  # ty: ignore[unresolved-import]

    status_map = {
        highspy.HighsBasisStatus.kLower: 0,
        highspy.HighsBasisStatus.kUpper: 1,
        highspy.HighsBasisStatus.kZero: 2,
        highspy.HighsBasisStatus.kNonbasic: 0,
        highspy.HighsBasisStatus.kBasic: 4,
    }
    statuses = [status_map[status] for status in basis.col_status]
    statuses.extend(
        4 if status == highspy.HighsBasisStatus.kBasic else 3 for status in basis.row_status
    )
    if len(statuses) != num_cols + num_rows:
        raise RuntimeError("mapped HiGHS status vector has the wrong length")
    return statuses


def _new_highs(
    model: dict[str, Any],
    tag: str,
    *,
    iteration_limit: int | None = None,
    basis: Any | None = None,
    verbose: bool = True,
) -> Any:
    import highspy  # ty: ignore[unresolved-import]

    log_path = OUT_DIR / f"{tag}.log"
    log_path.unlink(missing_ok=True)
    h = highspy.Highs()
    options: dict[str, Any] = {
        "solver": "simplex",
        "presolve": "off",
        "simplex_strategy": 1,
        "output_flag": True,
        "log_to_console": False,
        "log_file": str(log_path),
        "log_dev_level": 3 if verbose else 0,
    }
    if iteration_limit is not None:
        options["simplex_iteration_limit"] = iteration_limit
    for name, value in options.items():
        status = h.setOptionValue(name, value)
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"setOptionValue({name}={value!r}) failed: {status}")
    _pass_highs_model(h, model)
    if basis is not None:
        status = h.setBasis(basis)
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"setBasis failed: {status}")
    return h


def _solution_metrics(h: Any, original: dict[str, Any], reduction: Any) -> dict[str, float]:
    x = np.asarray(postsolve_x(h.getSolution().col_value, reduction), dtype=np.float64)
    return {
        "objective_original": float(np.dot(original["c"], x)),
        "max_equality_residual_original": float(
            np.max(np.abs(original["A_scipy"] @ x - original["b"]))
        ),
        "max_bound_violation_original": float(
            max(
                np.max(np.maximum(original["lo"] - x, 0.0)),
                np.max(np.maximum(x - original["hi"], 0.0)),
            )
        ),
    }


def extract_highs_bases(
    model: dict[str, Any], original: dict[str, Any], reduction: Any
) -> tuple[dict[str, Any], Any, Any, Any]:
    boundary_h = _new_highs(
        model, "highs_phase1_boundary", iteration_limit=PHASE1_BOUNDARY, verbose=True
    )
    first_status = boundary_h.run()
    boundary_info = boundary_h.getInfo()
    boundary_basis = boundary_h.getBasis()
    phase1 = {
        "method": "simplex_iteration_limit=1655 followed by getBasis()",
        "run_status": str(first_status),
        "model_status": boundary_h.modelStatusToString(boundary_h.getModelStatus()),
        "simplex_iterations": int(boundary_info.simplex_iteration_count),
        "basis": _basis_summary(boundary_basis),
        "mapped_basis_columns": len(
            _basis_columns(boundary_basis, model["A_scipy"].shape[1], model["A_scipy"].shape[0])
        ),
        "log": str(OUT_DIR / "highs_phase1_boundary.log"),
    }

    # Audit interruption semantics by resuming the same Highs object. HiGHS
    # reconstructs from the useful basis after an iteration-limit return, so
    # this records any recovery Phase-1 work rather than assuming continuity.
    boundary_h.setOptionValue("simplex_iteration_limit", 2_147_483_647)
    resumed_status = boundary_h.run()
    phase1["resume"] = {
        "run_status": str(resumed_status),
        "model_status": boundary_h.modelStatusToString(boundary_h.getModelStatus()),
        "final_simplex_iterations": int(boundary_h.getInfo().simplex_iteration_count),
        **_solution_metrics(boundary_h, original, reduction),
    }

    plus_one_h = _new_highs(
        model, "highs_phase1_plus_one", iteration_limit=PHASE1_PLUS_ONE, verbose=True
    )
    plus_one_status = plus_one_h.run()
    plus_one_basis = plus_one_h.getBasis()
    plus_one = {
        "method": (
            "simplex_iteration_limit=1656 followed by getBasis(); runtime log "
            "shows DuPh1=1655 and exactly one DuPh2 pivot"
        ),
        "run_status": str(plus_one_status),
        "model_status": plus_one_h.modelStatusToString(plus_one_h.getModelStatus()),
        "simplex_iterations": int(plus_one_h.getInfo().simplex_iteration_count),
        "basis": _basis_summary(plus_one_basis),
        "mapped_basis_columns": len(
            _basis_columns(plus_one_basis, model["A_scipy"].shape[1], model["A_scipy"].shape[0])
        ),
        "log": str(OUT_DIR / "highs_phase1_plus_one.log"),
    }
    plus_one_h.setOptionValue("simplex_iteration_limit", 2_147_483_647)
    plus_one_resume_status = plus_one_h.run()
    plus_one["resume"] = {
        "run_status": str(plus_one_resume_status),
        "model_status": plus_one_h.modelStatusToString(plus_one_h.getModelStatus()),
        "remaining_simplex_iterations": int(plus_one_h.getInfo().simplex_iteration_count),
        **_solution_metrics(plus_one_h, original, reduction),
    }

    optimal_h = _new_highs(model, "highs_optimal_basis", verbose=True)
    optimal_status = optimal_h.run()
    optimal_basis = optimal_h.getBasis()
    optimal = {
        "method": "full presolve-off solve followed by getBasis()",
        "run_status": str(optimal_status),
        "model_status": optimal_h.modelStatusToString(optimal_h.getModelStatus()),
        "simplex_iterations": int(optimal_h.getInfo().simplex_iteration_count),
        "basis": _basis_summary(optimal_basis),
        **_solution_metrics(optimal_h, original, reduction),
        "log": str(OUT_DIR / "highs_optimal_basis.log"),
    }
    return (
        {"phase1_boundary": phase1, "phase1_plus_one": plus_one, "optimal": optimal},
        boundary_basis,
        plus_one_basis,
        optimal_basis,
    )


def run_linprogx(
    prepared: dict[str, Any],
    original: dict[str, Any],
    basis: list[int] | None,
    bound_status: list[int] | None,
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    if basis is not None:
        kwargs["initial_basis"] = basis
    if bound_status is not None:
        kwargs["initial_bound_status"] = bound_status
    started = time.perf_counter()
    result = prepared["matrix"].solve_eq_box_dual_simplex(
        prepared["c"].tolist(),
        prepared["b"].tolist(),
        prepared["lo"].tolist(),
        prepared["hi"].tolist(),
        max_iter=50_000,
        tol=1e-8,
        expand=1,
        leaving_rule=1,
        bfrt=0,
        **kwargs,
    )
    wall = time.perf_counter() - started
    x = np.asarray(postsolve_x(result["x"], prepared["reduction"]), dtype=np.float64)
    metrics = {
        "tag": tag,
        "status": result["status"],
        "iterations": int(result["iterations"]),
        "bound_flips": int(result["bound_flips"]),
        "artificial_ejections": int(result["artificial_ejections"]),
        "ftran_mean_density": float(result["ftran_mean_density"]),
        "btran_mean_density": float(result["btran_mean_density"]),
        "phase_us": result["phase_us"],
        "wall_seconds": wall,
        "objective_original": float(np.dot(original["c"], x)),
        "max_equality_residual_original": float(
            np.max(np.abs(original["A_scipy"] @ x - original["b"]))
        ),
        "max_bound_violation_original": float(
            max(
                np.max(np.maximum(original["lo"] - x, 0.0)),
                np.max(np.maximum(x - original["hi"], 0.0)),
            )
        ),
        "warm_start": result.get("warm_start"),
        "basis_size": len(result["basis"]),
    }
    export = {"basis": result["basis"], "bound_status": result["bound_status"]}
    return metrics, export


def _highs_basis_from_linprogx(export: dict[str, Any], num_cols: int, num_rows: int) -> Any:
    import highspy  # ty: ignore[unresolved-import]

    basis_columns = set(export["basis"])
    bound_status = export["bound_status"]
    col_status = []
    for j in range(num_cols):
        if j in basis_columns:
            status = highspy.HighsBasisStatus.kBasic
        elif bound_status[j] == 1:
            status = highspy.HighsBasisStatus.kUpper
        elif bound_status[j] == 2:
            status = highspy.HighsBasisStatus.kZero
        else:
            status = highspy.HighsBasisStatus.kLower
        col_status.append(status)
    row_status = [
        highspy.HighsBasisStatus.kBasic
        if num_cols + i in basis_columns
        else highspy.HighsBasisStatus.kLower
        for i in range(num_rows)
    ]
    basis = highspy.HighsBasis()
    basis.col_status = col_status
    basis.row_status = row_status
    basis.valid = True
    return basis


def run_highs_from_linprogx_basis(
    model: dict[str, Any], original: dict[str, Any], reduction: Any, export: dict[str, Any]
) -> dict[str, Any]:
    basis = _highs_basis_from_linprogx(export, model["A_scipy"].shape[1], model["A_scipy"].shape[0])
    h = _new_highs(model, "highs_from_linprogx_optimal", basis=basis, verbose=True)
    started = time.perf_counter()
    status = h.run()
    wall = time.perf_counter() - started
    return {
        "set_basis_summary": _basis_summary(basis),
        "run_status": str(status),
        "model_status": h.modelStatusToString(h.getModelStatus()),
        "simplex_iterations": int(h.getInfo().simplex_iteration_count),
        "wall_seconds": wall,
        **_solution_metrics(h, original, reduction),
        "log": str(OUT_DIR / "highs_from_linprogx_optimal.log"),
    }


def _objective_ok(value: float, reference: float) -> bool:
    return abs(value - reference) / max(1.0, abs(reference)) <= OBJECTIVE_REL_LIMIT


def main() -> int:
    if os.environ.get("LINPROGX_DS_WARM_START") != "1":
        raise RuntimeError("set LINPROGX_DS_WARM_START=1")
    if os.environ.get("LINPROGX_DS_EXPORT_BASIS") is None:
        raise RuntimeError("set LINPROGX_DS_EXPORT_BASIS=1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    original, prepared = prepare()
    reduced = {key: prepared[key] for key in ("A_scipy", "b", "c", "lo", "hi")}

    highs, phase1_basis, phase1_plus_one_basis, highs_optimal_basis = extract_highs_bases(
        reduced, original, prepared["reduction"]
    )
    num_rows, num_cols = reduced["A_scipy"].shape
    phase1_columns = _basis_columns(phase1_basis, num_cols, num_rows)
    phase1_plus_one_columns = _basis_columns(phase1_plus_one_basis, num_cols, num_rows)
    phase1_plus_one_status = _basis_bound_status(phase1_plus_one_basis, num_cols, num_rows)
    optimal_columns = _basis_columns(highs_optimal_basis, num_cols, num_rows)
    optimal_status = _basis_bound_status(highs_optimal_basis, num_cols, num_rows)

    baseline, linprogx_export = run_linprogx(prepared, original, None, None, "native_crash")
    from_phase1_basis_only, _ = run_linprogx(
        prepared, original, phase1_columns, None, "highs_phase1_boundary_basis_only"
    )
    from_phase1_plus_one, _ = run_linprogx(
        prepared,
        original,
        phase1_plus_one_columns,
        phase1_plus_one_status,
        "highs_phase1_plus_one_basis_and_status",
    )
    from_highs_optimal, _ = run_linprogx(
        prepared,
        original,
        optimal_columns,
        optimal_status,
        "highs_optimal_basis_and_status",
    )
    reverse = run_highs_from_linprogx_basis(
        reduced, original, prepared["reduction"], linprogx_export
    )

    reference = baseline["objective_original"]
    objective_gates = {
        "linprogx_baseline": _objective_ok(baseline["objective_original"], reference),
        "linprogx_from_highs_phase1_boundary_basis_only": _objective_ok(
            from_phase1_basis_only["objective_original"], reference
        ),
        "linprogx_from_highs_phase1_plus_one": _objective_ok(
            from_phase1_plus_one["objective_original"], reference
        ),
        "linprogx_from_highs_optimal": _objective_ok(
            from_highs_optimal["objective_original"], reference
        ),
        "highs_from_linprogx_optimal": _objective_ok(reverse["objective_original"], reference),
    }
    if not all(objective_gates.values()):
        raise RuntimeError(f"objective gate failed: {objective_gates}")

    phase1_pivots = from_phase1_plus_one["iterations"]
    if phase1_pivots < 2_600:
        verdict = "LIVE"
    elif phase1_pivots > 3_200:
        verdict = "KILLED"
    else:
        verdict = "INCONCLUSIVE_BAND"
    payload = {
        "fixture": "/tmp/lpsuite/lp_greenbea.mat",
        "shape": [num_rows, num_cols, int(reduced["A_scipy"].nnz)],
        "highs_version": __import__("highspy").Highs().version(),
        "phase1_boundary_iteration": PHASE1_BOUNDARY,
        "certified_phase1_plus_one_iteration": PHASE1_PLUS_ONE,
        "highs_basis_extraction": highs,
        "linprogx_runs": {
            "native_crash": baseline,
            "from_highs_phase1_boundary_basis_only": from_phase1_basis_only,
            "from_highs_phase1_plus_one": from_phase1_plus_one,
            "from_highs_optimal": from_highs_optimal,
        },
        "reverse_highs_from_linprogx_optimal": reverse,
        "objective_relative_limit": OBJECTIVE_REL_LIMIT,
        "objective_gates": objective_gates,
        "verdict": verdict,
    }
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
