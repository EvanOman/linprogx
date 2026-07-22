"""Exact single-exchange lookahead for the full-KKT block-PDAS probe.

This is a standalone S0 diagnostic.  It deliberately imports the repaired
crash, scaling, legal-endpoint, candidate-generation, and certificate logic
from :mod:`experiments.block_pdas_probe`; it does not call the production
simplex trajectory.  The only policy change is predeclared lookahead: every
generated forward-valid edge is scored by exact-real single-exchange algebra
before a rank-safe matching is formed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.block_pdas_probe import (
    AT_HI,
    AT_LO,
    BASIC,
    BOARD_TARGET_SECONDS,
    CONTROL_SECONDS,
    EPS,
    FIXED,
    FREE,
    KKT_TOL,
    MAX_ACCEPTED,
    MAX_ATTEMPTED,
    MEDIAN_WIDTH_GATE,
    PIVOT_TOL,
    PROBE_TARGET_SECONDS,
    REFAC_REFERENCE_SECONDS,
    SENTINELS,
    Edge,
    Model,
    State,
    candidate_edges,
    certificate,
    crash_basis,
    initial_status,
    make_model,
    make_state,
    merit_json,
    proposed_state,
    state_key,
)

OUT_DIR = Path("/tmp/pdas-lookahead-falsifier")
RESULTS = OUT_DIR / "results.json"

BATCH_LADDER = (64, 32, 16, 8, 4, 2)
MERIT_FP_MULTIPLIER = 256.0
AUTHORITY_FP_MULTIPLIER = 4096.0


@dataclass(frozen=True)
class ScoredEdge:
    """One candidate plus its algebraic post-exchange merit."""

    edge: Edge
    merit: tuple[float, float, float, float]
    step: float
    dual_step: float


@dataclass
class PredictedState:
    """Full algebraic state retained only for scalar authority comparison."""

    basis: np.ndarray
    status: np.ndarray
    x_basis: np.ndarray
    y: np.ndarray
    reduced_cost: np.ndarray
    merit: tuple[float, float, float, float]


def _endpoint(model: Model, column: int, status: int) -> float:
    if status in (AT_LO, FIXED):
        value = float(model.lo[column])
    elif status == AT_HI:
        value = float(model.hi[column])
    elif status == FREE:
        value = 0.0
    else:
        raise ValueError(f"column {column} has no nonbasic endpoint for status {status}")
    if not np.isfinite(value):
        raise ValueError(f"column {column} status {status} has nonfinite endpoint")
    return value


def _merit_from_algebra(
    model: Model,
    basis: np.ndarray,
    status: np.ndarray,
    x_basis: np.ndarray,
    reduced: np.ndarray,
) -> tuple[float, float, float, float]:
    lo_violation = np.maximum(model.lo[basis] - x_basis, 0.0)
    hi_violation = np.maximum(x_basis - model.hi[basis], 0.0)
    bound_scale = 1.0 + np.maximum(
        np.where(np.isfinite(model.lo[basis]), np.abs(model.lo[basis]), 0.0),
        np.where(np.isfinite(model.hi[basis]), np.abs(model.hi[basis]), 0.0),
    )
    primal = np.maximum(lo_violation, hi_violation) / bound_scale

    dual = np.zeros(model.n + model.m)
    c_scale = 1.0 + np.abs(model.c)
    at_lo = status == AT_LO
    at_hi = status == AT_HI
    free = status == FREE
    dual[at_lo] = np.maximum(0.0, -reduced[at_lo]) / c_scale[at_lo]
    dual[at_hi] = np.maximum(0.0, reduced[at_hi]) / c_scale[at_hi]
    dual[free] = np.abs(reduced[free]) / c_scale[free]

    artificial = basis >= model.n
    return (
        float(np.sum(np.abs(x_basis[artificial]))),
        float(primal.max(initial=0.0)),
        float(dual.max(initial=0.0)),
        float(primal.sum() + dual.sum()),
    )


def merit_strictly_less(
    candidate: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> bool:
    """Tolerance-aware strict lexicographic comparison fixed before the run."""
    epsilon = np.finfo(np.float64).eps
    for after, before in zip(candidate, reference, strict=True):
        tolerance = MERIT_FP_MULTIPLIER * epsilon * (1.0 + abs(after) + abs(before))
        if after < before - tolerance:
            return True
        if after > before + tolerance:
            return False
    return False


def _algebraic_state(
    model: Model,
    state: State,
    edge: Edge,
    d: np.ndarray,
    h: np.ndarray,
    alpha: np.ndarray,
) -> PredictedState:
    p = edge.row
    entering = edge.entering
    leaving = int(state.basis[p])
    entering_endpoint = _endpoint(model, entering, int(state.status[entering]))
    leaving_endpoint = _endpoint(model, leaving, edge.leaving_status)
    pivot = float(d[p])
    if abs(pivot) <= PIVOT_TOL:
        raise ValueError("lookahead edge has a numerically zero pivot")

    step = (float(state.x_basis[p]) - leaving_endpoint) / pivot
    x_basis = state.x_basis - d * step
    x_basis[p] = entering_endpoint + step

    alpha_entering = float(alpha[entering])
    if abs(alpha_entering) <= PIVOT_TOL:
        raise ValueError("lookahead edge has a numerically zero dual pivot")
    dual_step = float(state.reduced_cost[entering]) / alpha_entering
    y = state.y + dual_step * h
    reduced = state.reduced_cost - dual_step * alpha
    reduced[entering] = 0.0

    basis = state.basis.copy()
    basis[p] = entering
    status = state.status.copy()
    status[leaving] = edge.leaving_status
    status[entering] = BASIC
    merit = _merit_from_algebra(model, basis, status, x_basis, reduced)
    return PredictedState(basis, status, x_basis, y, reduced, merit)


def score_edges(
    model: Model, state: State, edges: list[Edge]
) -> tuple[list[ScoredEdge], dict[tuple[int, int], PredictedState], float]:
    """Score all candidates with batched exact solves from the current factor."""
    started = time.perf_counter()
    if not edges:
        return [], {}, time.perf_counter() - started

    enterings = sorted({edge.entering for edge in edges})
    rows = sorted({edge.row for edge in edges})
    entering_slot = {column: slot for slot, column in enumerate(enterings)}
    row_slot = {row: slot for slot, row in enumerate(rows)}

    columns = model.a_aug[:, enterings].toarray()
    d_block = state.factor.solve(columns)
    unit = np.zeros((model.m, len(rows)))
    unit[rows, np.arange(len(rows))] = 1.0
    h_block = state.factor.solve(unit, trans="T")
    alpha_block = np.asarray(model.a_aug.T @ h_block)

    scored: list[ScoredEdge] = []
    predictions: dict[tuple[int, int], PredictedState] = {}
    for edge in edges:
        prediction = _algebraic_state(
            model,
            state,
            edge,
            d_block[:, entering_slot[edge.entering]],
            h_block[:, row_slot[edge.row]],
            alpha_block[:, row_slot[edge.row]],
        )
        predictions[(edge.row, edge.entering)] = prediction
        if merit_strictly_less(prediction.merit, state.merit):
            entering_endpoint = _endpoint(model, edge.entering, int(state.status[edge.entering]))
            scored.append(
                ScoredEdge(
                    edge,
                    prediction.merit,
                    float(prediction.x_basis[edge.row] - entering_endpoint),
                    float(state.reduced_cost[edge.entering])
                    / float(alpha_block[edge.entering, row_slot[edge.row]]),
                )
            )
    scored.sort(
        key=lambda item: (
            item.merit,
            item.edge.row,
            item.edge.entering,
            item.edge.leaving_status,
            item.edge.source_kind,
            item.edge.source_index,
        )
    )
    return scored, predictions, time.perf_counter() - started


def rank_safe_matching(
    model: Model, state: State, scored: list[ScoredEdge], limit: int = 64
) -> tuple[list[ScoredEdge], float]:
    """Greedily match the lookahead order while retaining a full-rank minor."""
    started = time.perf_counter()
    selected: list[ScoredEdge] = []
    used_rows: set[int] = set()
    used_columns: set[int] = set()
    q_cache: dict[int, np.ndarray] = {}
    for item in scored:
        edge = item.edge
        if edge.row in used_rows or edge.entering in used_columns:
            continue
        if edge.entering not in q_cache:
            q_cache[edge.entering] = state.factor.solve(
                model.a_aug[:, edge.entering].toarray().ravel()
            )
        trial = [*selected, item]
        rows = [candidate.edge.row for candidate in trial]
        exchange = np.column_stack([q_cache[candidate.edge.entering][rows] for candidate in trial])
        singular_values = np.linalg.svd(exchange, compute_uv=False)
        if singular_values[-1] <= PIVOT_TOL * max(1.0, singular_values[0]):
            continue
        selected.append(item)
        used_rows.add(edge.row)
        used_columns.add(edge.entering)
        if len(selected) == limit:
            break
    return selected, time.perf_counter() - started


def _edge_json(state: State, item: ScoredEdge) -> dict[str, Any]:
    edge = item.edge
    return {
        "row": edge.row,
        "entering": edge.entering,
        "leaving": int(state.basis[edge.row]),
        "leaving_status": edge.leaving_status,
        "source_kind": edge.source_kind,
        "source_index": edge.source_index,
        "source_rank": edge.source_rank,
        "severity": edge.severity,
        "ratio": edge.ratio,
        "pivot_abs": edge.pivot_abs,
        "predicted_merit": merit_json(item.merit),
        "predicted_primal_step": item.step,
        "predicted_dual_step": item.dual_step,
    }


def _authority_tolerances(state: State, trial: State) -> tuple[float, float]:
    conditioning = math.sqrt(max(1.0, state.growth_proxy) * max(1.0, trial.growth_proxy))
    relative = AUTHORITY_FP_MULTIPLIER * np.finfo(np.float64).eps * conditioning
    return relative, relative


def assert_scalar_authority(
    predicted: PredictedState, state: State, trial: State
) -> dict[str, Any]:
    """Fail closed unless the single-exchange algebra matches fresh LU authority."""
    relative, base_absolute = _authority_tolerances(state, trial)

    def comparison(name: str, expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
        scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
        absolute = base_absolute * scale
        error = float(np.max(np.abs(expected - actual), initial=0.0))
        passed = bool(np.allclose(expected, actual, rtol=relative, atol=absolute))
        if not passed:
            raise AssertionError(
                f"scalar {name} algebra mismatch: max_error={error:.6e}, "
                f"rtol={relative:.6e}, atol={absolute:.6e}"
            )
        return {"max_abs_error": error, "rtol": relative, "atol": absolute, "passed": passed}

    records = {
        "basis_exact": bool(np.array_equal(predicted.basis, trial.basis)),
        "status_exact": bool(np.array_equal(predicted.status, trial.status)),
        "x_basis": comparison("x_basis", predicted.x_basis, trial.x_basis),
        "y": comparison("y", predicted.y, trial.y),
        "reduced_cost": comparison("reduced_cost", predicted.reduced_cost, trial.reduced_cost),
        "merit": comparison("merit", np.asarray(predicted.merit), np.asarray(trial.merit)),
    }
    if not records["basis_exact"] or not records["status_exact"]:
        raise AssertionError("scalar basis/status algebra mismatch")
    return records


def run_greenbea() -> dict[str, Any]:
    model, _ = make_model()
    basis, crash = crash_basis(model.a, model.lo[: model.n], model.hi[: model.n])

    cold_one = model.prepared["matrix"].solve_eq_box_dual_simplex(
        model.prepared["c"].tolist(),
        model.prepared["b"].tolist(),
        model.prepared["lo"].tolist(),
        model.prepared["hi"].tolist(),
        max_iter=1,
        tol=1e-8,
        expand=1,
        leaving_rule=1,
        bfrt=0,
    )
    warm_one = model.prepared["matrix"].solve_eq_box_dual_simplex(
        model.prepared["c"].tolist(),
        model.prepared["b"].tolist(),
        model.prepared["lo"].tolist(),
        model.prepared["hi"].tolist(),
        max_iter=1,
        tol=1e-8,
        expand=1,
        leaving_rule=1,
        bfrt=0,
        initial_basis=basis.tolist(),
    )
    crash["native_plus_one_basis_match"] = bool(cold_one["basis"] == warm_one["basis"])
    crash["native_plus_one_status_match"] = bool(
        cold_one["bound_status"] == warm_one["bound_status"]
    )
    if not (crash["native_plus_one_basis_match"] and crash["native_plus_one_status_match"]):
        raise RuntimeError("Python crash does not reproduce native one-pivot basis and status")

    status = initial_status(model, basis)
    algorithm_started = time.perf_counter()
    state = make_state(model, basis, status)
    initial_merit = state.merit
    initial_factor_seconds = state.factor_seconds
    initial_factor_nnz = state.factor_nnz
    seen = {state_key(state)}
    proposals: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    accepted_widths: list[int] = []
    accepted = 0
    attempted = 0
    generated_total = 0
    improving_total = 0
    candidate_seconds = 0.0
    scoring_seconds = 0.0
    matching_seconds = 0.0
    fresh_factor_seconds = initial_factor_seconds
    cycle = False
    stop_reason = "unknown"

    while accepted < MAX_ACCEPTED and attempted < MAX_ATTEMPTED:
        if state.merit[0] <= KKT_TOL and state.merit[1] <= KKT_TOL and state.merit[2] <= KKT_TOL:
            stop_reason = "kkt_optimal"
            break

        edges, build_seconds = candidate_edges(model, state)
        scored, predictions, score_seconds = score_edges(model, state, edges)
        matching, match_seconds = rank_safe_matching(model, state, scored)
        round_record: dict[str, Any] = {
            "round": len(rounds) + 1,
            "merit_before": merit_json(state.merit),
            "generated_edges": len(edges),
            "predicted_strict_improvers": len(scored),
            "rank_safe_matching_width": len(matching),
            "candidate_generation_seconds": build_seconds,
            "single_exchange_scoring_seconds": score_seconds,
            "rank_matching_seconds": match_seconds,
        }
        rounds.append(round_record)
        generated_total += len(edges)
        improving_total += len(scored)
        candidate_seconds += build_seconds
        scoring_seconds += score_seconds
        matching_seconds += match_seconds
        accepted_this_cycle = False
        attempted_widths: set[int] = set()

        for requested in BATCH_LADDER:
            if attempted >= MAX_ATTEMPTED:
                break
            if len(matching) < requested:
                continue
            trial_items = matching[:requested]
            if len(trial_items) in attempted_widths:
                continue
            attempted_widths.add(len(trial_items))
            attempted += 1
            before = state.merit
            record: dict[str, Any] = {
                "attempt": attempted,
                "kind": "lookahead_matching_prefix",
                "requested_width": requested,
                "width": len(trial_items),
                "generated_edges": len(edges),
                "predicted_improving_edges": len(scored),
                "merit_before": merit_json(before),
                "best_single_predicted_merit": merit_json(trial_items[0].merit),
                "worst_selected_single_predicted_merit": merit_json(trial_items[-1].merit),
                "edges": [_edge_json(state, item) for item in trial_items],
            }
            try:
                trial = proposed_state(model, state, [item.edge for item in trial_items])
            except (RuntimeError, ValueError) as error:
                record.update(
                    {
                        "accepted": False,
                        "reason": f"fail_closed:{type(error).__name__}:{error}",
                    }
                )
                proposals.append(record)
                continue
            fresh_factor_seconds += trial.factor_seconds
            key = state_key(trial)
            repeated = key in seen
            improves = merit_strictly_less(trial.merit, before)
            record.update(
                {
                    "factor_seconds": trial.factor_seconds,
                    "factor_nnz": trial.factor_nnz,
                    "factor_growth_proxy": trial.growth_proxy,
                    "actual_merit": merit_json(trial.merit),
                    "repeated_state": repeated,
                    "accepted": bool(improves and not repeated),
                    "reason": (
                        "strict_tolerance_aware_lexicographic_decrease"
                        if improves and not repeated
                        else "repeated_state"
                        if repeated
                        else "no_strict_tolerance_aware_lexicographic_decrease"
                    ),
                }
            )
            proposals.append(record)
            if repeated:
                cycle = True
            if improves and not repeated:
                state = trial
                seen.add(key)
                accepted += 1
                accepted_widths.append(len(trial_items))
                round_record.update(
                    {
                        "accepted_attempt": attempted,
                        "accepted_width": len(trial_items),
                        "merit_after": merit_json(state.merit),
                    }
                )
                accepted_this_cycle = True
                break

        if accepted_this_cycle:
            continue

        if not scored:
            round_record["hard_stop"] = "no_predicted_strict_single_exchange_improver"
            stop_reason = "no_predicted_strict_single_exchange_improver"
            break
        if attempted >= MAX_ATTEMPTED:
            stop_reason = "attempt_cap"
            break

        best = scored[0]
        attempted += 1
        before = state.merit
        record = {
            "attempt": attempted,
            "kind": "best_predicted_scalar",
            "requested_width": 1,
            "width": 1,
            "generated_edges": len(edges),
            "predicted_improving_edges": len(scored),
            "merit_before": merit_json(before),
            "predicted_merit": merit_json(best.merit),
            "edges": [_edge_json(state, best)],
        }
        try:
            trial = proposed_state(model, state, [best.edge])
            fresh_factor_seconds += trial.factor_seconds
            authority = assert_scalar_authority(
                predictions[(best.edge.row, best.edge.entering)], state, trial
            )
        except (AssertionError, RuntimeError, ValueError) as error:
            record.update(
                {
                    "accepted": False,
                    "reason": f"fail_closed:{type(error).__name__}:{error}",
                }
            )
            proposals.append(record)
            stop_reason = "scalar_prediction_authority_mismatch"
            break
        key = state_key(trial)
        repeated = key in seen
        improves = merit_strictly_less(trial.merit, before)
        record.update(
            {
                "factor_seconds": trial.factor_seconds,
                "factor_nnz": trial.factor_nnz,
                "factor_growth_proxy": trial.growth_proxy,
                "actual_merit": merit_json(trial.merit),
                "authority_match": authority,
                "repeated_state": repeated,
                "accepted": bool(improves and not repeated),
                "reason": (
                    "strict_tolerance_aware_lexicographic_decrease"
                    if improves and not repeated
                    else "repeated_state"
                    if repeated
                    else "predicted_improvement_not_confirmed"
                ),
            }
        )
        proposals.append(record)
        if repeated:
            cycle = True
        if not improves or repeated:
            stop_reason = "scalar_repeated_state" if repeated else "scalar_prediction_not_confirmed"
            break
        state = trial
        seen.add(key)
        accepted += 1
        accepted_widths.append(1)
        round_record.update(
            {
                "accepted_attempt": attempted,
                "accepted_width": 1,
                "merit_after": merit_json(state.merit),
            }
        )
    else:
        stop_reason = "accepted_cap" if accepted >= MAX_ACCEPTED else "attempt_cap"

    algorithm_seconds = time.perf_counter() - algorithm_started
    cert = certificate(model, state)
    median_width = statistics.median(accepted_widths) if accepted_widths else 0.0
    factor_count = 1 + sum("factor_seconds" in proposal for proposal in proposals)
    refactor_floor = factor_count * REFAC_REFERENCE_SECONDS
    projected_complete = algorithm_seconds if cert["passed"] else math.inf
    gates = {
        "accepted_count_le_256": accepted <= MAX_ACCEPTED,
        "attempted_fresh_factor_count_le_384": attempted <= MAX_ATTEMPTED,
        "median_accepted_width_ge_18": median_width >= MEDIAN_WIDTH_GATE,
        "no_repeated_state": not cycle,
        "original_space_certificate": cert["passed"],
        "complete_projected_cost_le_gate": projected_complete <= PROBE_TARGET_SECONDS,
    }
    passed = all(gates.values())
    return {
        "fixture": "/tmp/lpsuite/lp_greenbea.mat",
        "shape": [model.m, model.n, int(model.a.nnz)],
        "verdict": "PASS_LOOKAHEAD_BLOCK_PDAS_CHARACTERIZATION"
        if passed
        else "KILL_LOOKAHEAD_BLOCK_PDAS_CHARACTERIZATION",
        "fixed_policy": {
            "candidate_generation": "repaired deterministic forward-valid full-KKT edges",
            "single_exchange_scoring": (
                "exact B^-1 a_j and B^-T e_p primal/dual post-state algebra"
            ),
            "improver_filter": ("256*FP-epsilon scale-aware strict lexicographic merit decrease"),
            "sort": "predicted post-merit, row, column, leaving status, source ties",
            "matching": "greedy row/column matching with full-rank exchange minor",
            "batch_ladder": list(BATCH_LADDER),
            "scalar_fallback": "best predicted strict improver; never unscored",
            "max_accepted": MAX_ACCEPTED,
            "max_fresh_factor_attempts": MAX_ATTEMPTED,
            "median_width_gate": MEDIAN_WIDTH_GATE,
            "kkt_tol": KKT_TOL,
            "eps": EPS,
            "merit": [
                "artificial_mass",
                "max_scaled_primal_violation",
                "max_scaled_dual_violation",
                "l1_scaled_KKT_violation",
            ],
            "scalar_authority_fp_multiplier": AUTHORITY_FP_MULTIPLIER,
        },
        "timing_contract": {
            "control_seconds": CONTROL_SECONDS,
            "board_target_seconds": BOARD_TARGET_SECONDS,
            "probe_target_seconds": PROBE_TARGET_SECONDS,
            "refactor_reference_seconds": REFAC_REFERENCE_SECONDS,
        },
        "crash": crash,
        "initial_merit": merit_json(initial_merit),
        "final_merit": merit_json(state.merit),
        "initial_factor_seconds": initial_factor_seconds,
        "initial_factor_nnz": initial_factor_nnz,
        "accepted": accepted,
        "attempted_fresh_factors": attempted,
        "accepted_widths": accepted_widths,
        "median_accepted_width": median_width,
        "generated_edges_total": generated_total,
        "predicted_improving_edges_total": improving_total,
        "unique_states": len(seen),
        "cycle_observed": cycle,
        "stop_reason": stop_reason,
        "candidate_generation_seconds": candidate_seconds,
        "single_exchange_scoring_seconds": scoring_seconds,
        "rank_matching_seconds": matching_seconds,
        "fresh_factor_seconds": fresh_factor_seconds,
        "algorithm_seconds": algorithm_seconds,
        "fresh_factor_count": factor_count,
        "refactor_only_projection_floor_seconds": refactor_floor,
        "projected_complete_cost_seconds": projected_complete,
        "certificate": cert,
        "gates": gates,
        "passed": passed,
        "rounds": rounds,
        "proposals": proposals,
    }


def run() -> dict[str, Any]:
    greenbea = run_greenbea()
    sentinels: list[dict[str, Any]] = []
    if greenbea["passed"]:
        raise NotImplementedError(
            "greenbea unexpectedly passed; implement the predeclared sentinel loader before ship review"
        )
    return {
        "verdict": greenbea["verdict"],
        "greenbea": greenbea,
        "sentinel_policy": {
            "fixtures": list(SENTINELS),
            "run_only_after_greenbea_pass": True,
            "executed": False,
        },
        "sentinels": sentinels,
    }


def main() -> None:
    os.environ.setdefault("LINPROGX_DS_EXPORT_BASIS", "1")
    os.environ.setdefault("LINPROGX_DS_WARM_START", "1")
    OUT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(OUT_DIR, 0o700)
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    RESULTS.write_text(payload)
    os.chmod(RESULTS, 0o600)
    print(RESULTS)
    print(hashlib.sha256(payload.encode()).hexdigest())
    summary = result["greenbea"]
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "verdict",
                    "accepted",
                    "attempted_fresh_factors",
                    "accepted_widths",
                    "median_accepted_width",
                    "generated_edges_total",
                    "predicted_improving_edges_total",
                    "stop_reason",
                    "algorithm_seconds",
                    "final_merit",
                    "gates",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
