"""Black-box measurement probe for the greenbea HiGHS pivot gap.

This experiment deliberately treats HiGHS as a black box: it uses only the
public highspy model/options/info interfaces and runtime logs. It does not read
HiGHS source code.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from experiments.suite_bench import bounds_of, load_instance
from linprogx.presolve import postsolve_x, presolve_matrix
from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

FIXTURE = Path("/tmp/lpsuite/lp_greenbea.mat")
OUT_DIR = Path("/tmp/greenbea-pivot-gap")
RESULTS = OUT_DIR / "results.json"
TOL = 1e-8


def prepare() -> tuple[dict[str, Any], dict[str, Any]]:
    data = load_instance(FIXTURE)
    matrix = from_scipy_sparse(data["A_scipy"])
    reduction = presolve_matrix(
        matrix,
        data["b"].tolist(),
        data["c"].tolist(),
        data["lo"].tolist(),
        data["hi"].tolist(),
        algorithm="dual_simplex",
    )
    if reduction is None:
        raise RuntimeError("greenbea unexpectedly had no linprogx presolve reduction")
    reduced = reduction._matrix
    if reduced is None:
        reduced = type(matrix)(
            reduction.rows,
            reduction.cols,
            reduction.indptr,
            reduction.indices,
            reduction.data,
        )
    indptr, indices, values = reduced.to_components()
    reduced_scipy = sparse.csr_matrix(
        (np.asarray(values), np.asarray(indices), np.asarray(indptr)),
        shape=reduced.shape,
    ).tocsc()
    prepared = {
        "A_scipy": reduced_scipy,
        "b": np.asarray(reduction.b, dtype=np.float64),
        "c": np.asarray(reduction.c, dtype=np.float64),
        "lo": np.asarray(reduction.lo, dtype=np.float64),
        "hi": np.asarray(reduction.hi, dtype=np.float64),
        "matrix": reduced,
        "reduction": reduction,
    }
    return data, prepared


def _pass_highs_model(h: Any, model: dict[str, Any]) -> None:
    import highspy

    A = model["A_scipy"].tocsc()
    m, n = A.shape
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = np.asarray(model["c"], dtype=np.float64)
    lp.col_lower_ = np.where(np.isneginf(model["lo"]), -inf, model["lo"])
    lp.col_upper_ = np.where(np.isposinf(model["hi"]), inf, model["hi"])
    lp.row_lower_ = np.asarray(model["b"], dtype=np.float64)
    lp.row_upper_ = np.asarray(model["b"], dtype=np.float64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32).tolist()
    lp.a_matrix_.index_ = A.indices.astype(np.int32).tolist()
    lp.a_matrix_.value_ = A.data.astype(np.float64).tolist()
    lp.a_matrix_.num_col_ = n
    lp.a_matrix_.num_row_ = m
    status = h.passModel(lp)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"passModel failed: {status}")


def run_highs(
    tag: str,
    model: dict[str, Any],
    original: dict[str, Any],
    *,
    postsolve: Any | None = None,
    presolve: str = "off",
    options: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    import highspy

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / f"{tag}.log"
    log_path.unlink(missing_ok=True)
    h = highspy.Highs()
    base_options: dict[str, Any] = {
        "solver": "simplex",
        "presolve": presolve,
        "simplex_strategy": 1,
        "output_flag": True,
        "log_to_console": False,
        "log_file": str(log_path),
        "log_dev_level": 3 if verbose else 0,
    }
    base_options.update(options or {})
    for name, value in base_options.items():
        status = h.setOptionValue(name, value)
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"setOptionValue({name}={value!r}) failed: {status}")
    _pass_highs_model(h, model)
    started = time.perf_counter()
    status = h.run()
    wall = time.perf_counter() - started
    info = h.getInfo()
    solution = h.getSolution()
    x_reduced = np.asarray(solution.col_value, dtype=np.float64)
    x_original = np.asarray(postsolve_x(x_reduced.tolist(), postsolve)) if postsolve else x_reduced
    objective = float(np.dot(original["c"], x_original))
    equality_residual = float(np.max(np.abs(original["A_scipy"] @ x_original - original["b"])))
    lower_violation = float(np.max(np.maximum(original["lo"] - x_original, 0.0)))
    upper_violation = float(np.max(np.maximum(x_original - original["hi"], 0.0)))
    basis = h.getBasis()
    row_basis = Counter(str(value).rsplit(".", 1)[-1] for value in basis.row_status)
    col_basis = Counter(str(value).rsplit(".", 1)[-1] for value in basis.col_status)
    lp = h.getLp()
    return {
        "tag": tag,
        "status": str(status),
        "model_status": h.modelStatusToString(h.getModelStatus()),
        "simplex_iterations": int(info.simplex_iteration_count),
        "objective_original": objective,
        "max_equality_residual_original": equality_residual,
        "max_bound_violation_original": max(lower_violation, upper_violation),
        "wall_seconds": wall,
        "solved_lp_shape": [int(lp.num_row_), int(lp.num_col_), len(lp.a_matrix_.value_)],
        "basis_valid": bool(basis.valid),
        "final_row_basis_status": dict(sorted(row_basis.items())),
        "final_col_basis_status": dict(sorted(col_basis.items())),
        "options": {name: h.getOptionValue(name)[1] for name in base_options},
        "log": str(log_path),
    }


def run_linprogx(original: dict[str, Any], prepared: dict[str, Any]) -> dict[str, Any]:
    direct: dict[str, Any] = {}
    for bfrt in (0, 1):
        started = time.perf_counter()
        result = prepared["matrix"].solve_eq_box_dual_simplex(
            prepared["c"].tolist(),
            prepared["b"].tolist(),
            prepared["lo"].tolist(),
            prepared["hi"].tolist(),
            max_iter=50_000,
            tol=TOL,
            expand=1,
            leaving_rule=1,
            bfrt=bfrt,
        )
        wall = time.perf_counter() - started
        x = np.asarray(postsolve_x(result["x"], prepared["reduction"]))
        direct[f"dantzig_bfrt_{bfrt}"] = {
            key: result.get(key)
            for key in (
                "status",
                "iterations",
                "bound_flips",
                "artificial_ejections",
                "degenerate_pivots",
                "bland_pivots",
                "max_degenerate_streak",
                "refactorizations",
                "phase_us",
            )
        }
        direct[f"dantzig_bfrt_{bfrt}"].update(
            {
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
            }
        )

    public = SparseSolver(
        algorithm="auto", max_iterations=50_000, eps=2e-5, check_interval=50_000
    ).solve(
        SparseLPProblem(
            c=original["c"].tolist(),
            A_eq=from_scipy_sparse(original["A_scipy"]),
            b_eq=original["b"].tolist(),
            objective="min",
            bounds=bounds_of(original),
        )
    )
    return {
        "direct": direct,
        "public_auto": {
            "backend": public.backend,
            "status": public.solution.status.value,
            "iterations": public.solution.iterations,
            "message": public.solution.message,
            "wall_seconds": public.seconds,
        },
    }


def run_reverse_cross(original: dict[str, Any]) -> dict[str, Any]:
    """Run linprogx DS directly on HiGHS's public presolved LP export."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", "on")
    _pass_highs_model(h, original)
    status = h.presolve()
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS presolve failed: {status}")
    lp = h.getPresolvedLp()
    starts = np.asarray(lp.a_matrix_.start_, dtype=np.int64)
    matrix = sparse.csc_matrix(
        (
            np.asarray(lp.a_matrix_.value_, dtype=np.float64),
            np.asarray(lp.a_matrix_.index_, dtype=np.int64),
            starts,
        ),
        shape=(lp.num_row_, lp.num_col_),
    )
    row_lo = np.asarray(lp.row_lower_, dtype=np.float64)
    row_hi = np.asarray(lp.row_upper_, dtype=np.float64)
    equality_rows = row_lo == row_hi
    col_lo = np.asarray(lp.col_lower_, dtype=np.float64)
    col_hi = np.asarray(lp.col_upper_, dtype=np.float64)
    col_lo = np.where(col_lo <= -0.5 * highspy.kHighsInf, -np.inf, col_lo)
    col_hi = np.where(col_hi >= 0.5 * highspy.kHighsInf, np.inf, col_hi)
    linprogx_matrix = from_scipy_sparse(matrix)
    runs: dict[str, Any] = {}
    for bfrt in (0, 1):
        started = time.perf_counter()
        result = linprogx_matrix.solve_eq_box_dual_simplex(
            list(lp.col_cost_),
            row_lo.tolist(),
            col_lo.tolist(),
            col_hi.tolist(),
            max_iter=50_000,
            tol=TOL,
            expand=1,
            leaving_rule=1,
            bfrt=bfrt,
        )
        wall = time.perf_counter() - started
        x = np.asarray(result["x"], dtype=np.float64)
        runs[f"dantzig_bfrt_{bfrt}"] = {
            key: result.get(key)
            for key in (
                "status",
                "iterations",
                "bound_flips",
                "artificial_ejections",
                "degenerate_pivots",
                "phase_us",
            )
        }
        runs[f"dantzig_bfrt_{bfrt}"].update(
            {
                "wall_seconds": wall,
                "objective_with_highs_offset": float(result["objective"] + lp.offset_),
                "max_equality_residual_reduced": float(np.max(np.abs(matrix @ x - row_lo))),
            }
        )
    return {
        "shape": [int(lp.num_row_), int(lp.num_col_), len(lp.a_matrix_.value_)],
        "equality_rows": int(equality_rows.sum()),
        "non_equality_rows": int((~equality_rows).sum()),
        "objective_offset": float(lp.offset_),
        "runs": runs,
    }


def main() -> int:
    original, prepared = prepare()
    raw = {
        "A_scipy": original["A_scipy"],
        "b": original["b"],
        "c": original["c"],
        "lo": original["lo"],
        "hi": original["hi"],
    }
    reduced = {key: prepared[key] for key in ("A_scipy", "b", "c", "lo", "hi")}
    runs: list[dict[str, Any]] = []

    # Decisive cross and raw log anatomy.
    runs.append(
        run_highs(
            "our_reduction_presolve_off",
            reduced,
            original,
            postsolve=prepared["reduction"],
            presolve="off",
            verbose=True,
        )
    )
    runs.append(run_highs("raw_presolve_on", raw, original, presolve="on", verbose=True))
    runs.append(run_highs("raw_presolve_off", raw, original, presolve="off", verbose=True))
    runs.append(
        run_highs(
            "our_reduction_presolve_on",
            reduced,
            original,
            postsolve=prepared["reduction"],
            presolve="on",
            verbose=True,
        )
    )

    # Publicly documented simplex strategy variants on our fixed reduction.
    for strategy in (0, 2, 3, 4):
        runs.append(
            run_highs(
                f"our_reduction_strategy_{strategy}",
                reduced,
                original,
                postsolve=prepared["reduction"],
                options={"simplex_strategy": strategy},
            )
        )

    # Runtime-exposed crash option: highspy 1.14 accepts the integer range 0..9.
    # Meanings are not published, so report these as opaque numeric ablations.
    for crash in range(1, 10):
        runs.append(
            run_highs(
                f"our_reduction_crash_{crash}",
                reduced,
                original,
                postsolve=prepared["reduction"],
                options={"simplex_crash_strategy": crash},
            )
        )

    # Documented dual edge-weight strategies: choose, Dantzig, Devex, steepest edge.
    for edge_weight in (0, 1, 2):
        runs.append(
            run_highs(
                f"our_reduction_edge_{edge_weight}",
                reduced,
                original,
                postsolve=prepared["reduction"],
                options={"simplex_dual_edge_weight_strategy": edge_weight},
            )
        )

    payload = {
        "fixture": str(FIXTURE),
        "highs_version": __import__("highspy").Highs().version(),
        "raw_shape": [*original["A_scipy"].shape, int(original["A_scipy"].nnz)],
        "linprogx_reduced_shape": [
            *prepared["A_scipy"].shape,
            int(prepared["A_scipy"].nnz),
        ],
        "linprogx_reduction_counts": prepared["reduction"]._reduction_counts,
        "highs_runs": runs,
        "linprogx_runs": run_linprogx(original, prepared),
        "reverse_cross_linprogx_on_highs_reduction": run_reverse_cross(raw),
    }
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
