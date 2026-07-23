"""Bounded exact simultaneous-block-merit characterization for full-KKT PDAS.

The probe reproduces the fixed exact-single-exchange lookahead trajectory to
its round-24 zero-single-improver state, then runs one predeclared bounded
interaction rescue.  It is diagnostic-only and does not alter or call a new
production solver policy.
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
    FREE,
    MAX_ACCEPTED,
    MAX_ATTEMPTED,
    MEDIAN_WIDTH_GATE,
    PIVOT_TOL,
    PROBE_TARGET_SECONDS,
    REFAC_REFERENCE_SECONDS,
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
from experiments.pdas_lookahead_probe import (
    PredictedState,
    _algebraic_state,
    _authority_tolerances,
    _endpoint,
    _merit_from_algebra,
    merit_strictly_less,
    run_greenbea,
)

OUT_DIR = Path("/tmp/pdas-block-merit-falsifier")
RESULTS = OUT_DIR / "results.json"

POOL_LIMIT = 64
SNAPSHOT_WIDTHS = (2, 4, 8, 16, 32, 64)
WIDE_WIDTHS = (32, 64)
DENSE_BACKWARD_MULTIPLIER = 4096.0
JOINT_DIRECTION_FP_MULTIPLIER = 64.0

EXPECTED_ACCEPTED_WIDTHS = (
    32,
    16,
    32,
    8,
    16,
    2,
    1,
    8,
    4,
    4,
    4,
    8,
    1,
    1,
    8,
    4,
    8,
    4,
    4,
    2,
    1,
    2,
    2,
)
EXPECTED_FINAL_MERIT = (
    0.0,
    21463.265411141812,
    17679.010085693157,
    587808.2604128214,
)
EXPECTED_FINAL_STATE_HASH = "58769c2a8748a08a9a8cf064cdd6e8f08ae9d20db3e94bfe0854ca8d2af98211"


@dataclass(frozen=True)
class ScalarPrediction:
    edge: Edge
    state: PredictedState
    edge_id: str


@dataclass
class AlgebraCache:
    enterings: list[int]
    rows: list[int]
    entering_slot: dict[int, int]
    row_slot: dict[int, int]
    d: np.ndarray
    z: np.ndarray
    alpha: np.ndarray


@dataclass
class BlockPrediction:
    edges: tuple[Edge, ...]
    edge_ids: tuple[str, ...]
    basis: np.ndarray
    status: np.ndarray
    x_basis: np.ndarray
    y: np.ndarray
    reduced_cost: np.ndarray
    merit: tuple[float, float, float, float]
    delta: np.ndarray
    dual_step: np.ndarray
    singular_min: float
    singular_max: float
    condition_proxy: float
    primal_backward_residual: float
    dual_backward_residual: float
    max_leaving_endpoint_error: float
    max_new_basic_reduced_cost: float
    min_joint_direction_margin: float


def edge_id(edge: Edge) -> str:
    return (
        f"r{edge.row}:j{edge.entering}:s{edge.leaving_status}:"
        f"{edge.source_kind}{edge.source_index}:k{edge.source_rank}"
    )


def edge_tie(edge: Edge) -> tuple[Any, ...]:
    return (
        edge.row,
        edge.entering,
        edge.leaving_status,
        edge.source_kind,
        edge.source_index,
        edge.source_rank,
        edge.ratio,
        -edge.pivot_abs,
    )


def basis_status_hash(basis: np.ndarray, status: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(basis.astype("<i8", copy=False).tobytes())
    digest.update(status.astype("i1", copy=False).tobytes())
    return digest.hexdigest()


def reconstruct_terminal_state(
    baseline: dict[str, Any], model: Model
) -> tuple[State, set[str], float]:
    """Rebuild the authority terminal state from accepted recorded exchanges."""
    basis, _ = crash_basis(model.a, model.lo[: model.n], model.hi[: model.n])
    status = initial_status(model, basis)
    seen = {basis_status_hash(basis, status)}
    accepted_widths: list[int] = []
    for proposal in baseline["proposals"]:
        if not proposal["accepted"]:
            continue
        accepted_widths.append(int(proposal["width"]))
        for record in proposal["edges"]:
            row = int(record["row"])
            entering = int(record["entering"])
            leaving = int(basis[row])
            if leaving != int(record["leaving"]):
                raise AssertionError("accepted-path leaving column mismatch")
            status[leaving] = int(record["leaving_status"])
            status[entering] = BASIC
            basis[row] = entering
        key = basis_status_hash(basis, status)
        if key in seen:
            raise AssertionError("accepted-path basis/status cycle")
        seen.add(key)

    if tuple(accepted_widths) != EXPECTED_ACCEPTED_WIDTHS:
        raise AssertionError("lookahead accepted-width authority changed")
    started = time.perf_counter()
    state = make_state(model, basis, status)
    reconstruction_seconds = time.perf_counter() - started
    if tuple(baseline["final_merit"]) != EXPECTED_FINAL_MERIT:
        raise AssertionError("lookahead terminal-merit authority changed")
    if state.merit != EXPECTED_FINAL_MERIT:
        raise AssertionError("reconstructed terminal merit does not exactly match authority")
    if state_key(state) != EXPECTED_FINAL_STATE_HASH:
        raise AssertionError("reconstructed terminal basis/status hash changed")
    if len(baseline["rounds"]) != 24:
        raise AssertionError("lookahead did not stop at authority round 24")
    final_round = baseline["rounds"][-1]
    if not (
        final_round["generated_edges"] == 512
        and final_round["predicted_strict_improvers"] == 0
        and baseline["stop_reason"] == "no_predicted_strict_single_exchange_improver"
    ):
        raise AssertionError("lookahead terminal zero-single-improver authority changed")
    return state, seen, reconstruction_seconds


def deduplicate_edges(edges: list[Edge]) -> list[Edge]:
    """Choose one deterministic source record per actual basis exchange."""
    chosen: dict[tuple[int, int, int], Edge] = {}
    for edge in edges:
        identity = (edge.row, edge.entering, edge.leaving_status)
        incumbent = chosen.get(identity)
        source_tie = (
            edge.source_rank,
            edge.ratio,
            -edge.pivot_abs,
            edge.source_kind,
            edge.source_index,
        )
        if incumbent is None:
            chosen[identity] = edge
            continue
        incumbent_tie = (
            incumbent.source_rank,
            incumbent.ratio,
            -incumbent.pivot_abs,
            incumbent.source_kind,
            incumbent.source_index,
        )
        if source_tie < incumbent_tie:
            chosen[identity] = edge
    return list(chosen.values())


def make_cache(model: Model, state: State, edges: list[Edge]) -> AlgebraCache:
    enterings = sorted({edge.entering for edge in edges})
    rows = sorted({edge.row for edge in edges})
    entering_slot = {column: slot for slot, column in enumerate(enterings)}
    row_slot = {row: slot for slot, row in enumerate(rows)}
    d = state.factor.solve(model.a_aug[:, enterings].toarray())
    unit = np.zeros((model.m, len(rows)))
    unit[rows, np.arange(len(rows))] = 1.0
    z = state.factor.solve(unit, trans="T")
    alpha = np.asarray(model.a_aug.T @ z)
    return AlgebraCache(enterings, rows, entering_slot, row_slot, d, z, alpha)


def scalar_pool(
    model: Model, state: State, edges: list[Edge]
) -> tuple[list[ScalarPrediction], AlgebraCache, dict[str, Any]]:
    started = time.perf_counter()
    unique = deduplicate_edges(edges)
    cache = make_cache(model, state, unique)
    predictions: list[ScalarPrediction] = []
    for edge in unique:
        column_slot = cache.entering_slot[edge.entering]
        row_slot = cache.row_slot[edge.row]
        prediction = _algebraic_state(
            model,
            state,
            edge,
            cache.d[:, column_slot],
            cache.z[:, row_slot],
            cache.alpha[:, row_slot],
        )
        predictions.append(ScalarPrediction(edge, prediction, edge_id(edge)))
    predictions.sort(key=lambda item: (item.state.merit, edge_tie(item.edge)))
    pool = predictions[:POOL_LIMIT]
    return (
        pool,
        cache,
        {
            "generated_count": len(edges),
            "deduplicated_count": len(unique),
            "pool_count": len(pool),
            "strict_single_improvers": sum(
                merit_strictly_less(item.state.merit, state.merit) for item in predictions
            ),
            "seconds": time.perf_counter() - started,
        },
    )


def dense_backward_residual(matrix: np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> float:
    numerator = float(np.linalg.norm(matrix @ solution - rhs, ord=np.inf))
    denominator = (
        1.0
        + float(np.linalg.norm(matrix, ord=np.inf)) * float(np.linalg.norm(solution, ord=np.inf))
        + float(np.linalg.norm(rhs, ord=np.inf))
    )
    return numerator / denominator


def predict_block(
    model: Model,
    state: State,
    cache: AlgebraCache,
    edges: tuple[Edge, ...],
) -> tuple[BlockPrediction | None, str, dict[str, float]]:
    """Return the exact simultaneous post-state, or a fixed fail-closed reason."""
    rows = [edge.row for edge in edges]
    enterings = [edge.entering for edge in edges]
    if len(set(rows)) != len(rows):
        return None, "duplicate_row", {}
    if len(set(enterings)) != len(enterings):
        return None, "duplicate_entering", {}

    d_slots = [cache.entering_slot[column] for column in enterings]
    z_slots = [cache.row_slot[row] for row in rows]
    d = cache.d[:, d_slots]
    z = cache.z[:, z_slots]
    alpha = cache.alpha[:, z_slots]
    h = d[np.ix_(rows, range(len(edges)))]
    singular_values = np.linalg.svd(h, compute_uv=False)
    singular_min = float(singular_values[-1])
    singular_max = float(singular_values[0])
    rank_floor = PIVOT_TOL * max(1.0, singular_max)
    if singular_min <= rank_floor:
        return (
            None,
            "rank_unsafe",
            {
                "singular_min": singular_min,
                "singular_max": singular_max,
                "rank_floor": rank_floor,
            },
        )

    leaving = [int(state.basis[row]) for row in rows]
    leaving_endpoints = np.asarray(
        [
            _endpoint(model, column, edge.leaving_status)
            for column, edge in zip(leaving, edges, strict=True)
        ]
    )
    entering_endpoints = np.asarray(
        [_endpoint(model, edge.entering, int(state.status[edge.entering])) for edge in edges]
    )
    primal_rhs = state.x_basis[rows] - leaving_endpoints
    dual_rhs = state.reduced_cost[enterings]
    try:
        delta = np.linalg.solve(h, primal_rhs)
        dual_step = np.linalg.solve(h.T, dual_rhs)
    except np.linalg.LinAlgError:
        return (
            None,
            "dense_solve_failure",
            {
                "singular_min": singular_min,
                "singular_max": singular_max,
            },
        )

    primal_backward = dense_backward_residual(h, delta, primal_rhs)
    dual_backward = dense_backward_residual(h.T, dual_step, dual_rhs)
    backward_limit = float(
        DENSE_BACKWARD_MULTIPLIER * np.finfo(np.float64).eps * max(1, len(edges))
    )
    if max(primal_backward, dual_backward) > backward_limit:
        return (
            None,
            "dense_backward_residual",
            {
                "primal_backward_residual": primal_backward,
                "dual_backward_residual": dual_backward,
                "backward_limit": backward_limit,
            },
        )

    old_basic_after = state.x_basis - d @ delta
    leaving_error = float(np.max(np.abs(old_basic_after[rows] - leaving_endpoints), initial=0.0))
    endpoint_scale = max(
        1.0,
        float(np.max(np.abs(state.x_basis[rows]), initial=0.0)),
        float(np.max(np.abs(leaving_endpoints), initial=0.0)),
    )
    endpoint_limit = float(backward_limit * endpoint_scale)
    if leaving_error > endpoint_limit:
        return (
            None,
            "leaving_endpoint_residual",
            {
                "max_leaving_endpoint_error": leaving_error,
                "endpoint_limit": endpoint_limit,
            },
        )

    direction_margins: list[float] = []
    direction_valid = True
    for edge, movement in zip(edges, delta, strict=True):
        status = int(state.status[edge.entering])
        if status == AT_LO:
            margin = float(movement)
        elif status == AT_HI:
            margin = -float(movement)
        elif status == FREE:
            margin = math.inf
        else:
            return None, "invalid_entering_status", {}
        tolerance = (
            JOINT_DIRECTION_FP_MULTIPLIER * np.finfo(np.float64).eps * (1.0 + abs(float(movement)))
        )
        direction_margins.append(margin)
        if margin < -tolerance:
            direction_valid = False
    min_direction_margin = min(direction_margins, default=math.inf)
    if not direction_valid:
        return (
            None,
            "joint_direction_reversal",
            {"min_joint_direction_margin": min_direction_margin},
        )

    x_basis = old_basic_after
    x_basis[rows] = entering_endpoints + delta
    y = state.y + z @ dual_step
    reduced = state.reduced_cost - alpha @ dual_step
    new_basic_reduced_error = float(np.max(np.abs(reduced[enterings]), initial=0.0))
    reduced[enterings] = 0.0

    basis = state.basis.copy()
    status = state.status.copy()
    for row, entering, old_column, edge in zip(rows, enterings, leaving, edges, strict=True):
        basis[row] = entering
        status[old_column] = edge.leaving_status
        status[entering] = BASIC
    merit = _merit_from_algebra(model, basis, status, x_basis, reduced)
    return (
        BlockPrediction(
            edges=edges,
            edge_ids=tuple(edge_id(edge) for edge in edges),
            basis=basis,
            status=status,
            x_basis=x_basis,
            y=y,
            reduced_cost=reduced,
            merit=merit,
            delta=delta,
            dual_step=dual_step,
            singular_min=singular_min,
            singular_max=singular_max,
            condition_proxy=singular_max / singular_min,
            primal_backward_residual=primal_backward,
            dual_backward_residual=dual_backward,
            max_leaving_endpoint_error=leaving_error,
            max_new_basic_reduced_cost=new_basic_reduced_error,
            min_joint_direction_margin=min_direction_margin,
        ),
        "valid",
        {},
    )


def merit_margin(
    candidate: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> dict[str, Any]:
    epsilon = np.finfo(np.float64).eps
    for index, (after, before) in enumerate(zip(candidate, reference, strict=True)):
        tolerance = 256.0 * epsilon * (1.0 + abs(after) + abs(before))
        difference = before - after
        if abs(difference) > tolerance:
            return {
                "decisive_component": index,
                "signed_improvement_margin": difference,
                "tolerance": tolerance,
                "strict_improvement": difference > 0.0,
            }
    return {
        "decisive_component": None,
        "signed_improvement_margin": 0.0,
        "tolerance": 0.0,
        "strict_improvement": False,
    }


def prediction_json(prediction: BlockPrediction, reference: State) -> dict[str, Any]:
    return {
        "width": len(prediction.edges),
        "edge_ids": list(prediction.edge_ids),
        "merit": merit_json(prediction.merit),
        "merit_margin": merit_margin(prediction.merit, reference.merit),
        "joint_direction_valid": True,
        "min_joint_direction_margin": prediction.min_joint_direction_margin,
        "singular_min": prediction.singular_min,
        "singular_max": prediction.singular_max,
        "condition_proxy": prediction.condition_proxy,
        "primal_backward_residual": prediction.primal_backward_residual,
        "dual_backward_residual": prediction.dual_backward_residual,
        "max_leaving_endpoint_error": prediction.max_leaving_endpoint_error,
        "max_new_basic_reduced_cost": prediction.max_new_basic_reduced_cost,
    }


def exact_interaction_path(
    model: Model,
    state: State,
    pool: list[ScalarPrediction],
    cache: AlgebraCache,
) -> tuple[dict[int, BlockPrediction], dict[str, Any]]:
    pair_started = time.perf_counter()
    reason_counts: dict[str, int] = {}
    compatible_pairs = 0
    valid_pairs: list[BlockPrediction] = []
    closest_invalid_direction_margin = -math.inf
    for first_index, first in enumerate(pool):
        for second in pool[first_index + 1 :]:
            if first.edge.row == second.edge.row or first.edge.entering == second.edge.entering:
                reason_counts["matching_conflict"] = reason_counts.get("matching_conflict", 0) + 1
                continue
            compatible_pairs += 1
            prediction, reason, diagnostics = predict_block(
                model, state, cache, (first.edge, second.edge)
            )
            if prediction is None:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if reason == "joint_direction_reversal":
                    closest_invalid_direction_margin = max(
                        closest_invalid_direction_margin,
                        diagnostics["min_joint_direction_margin"],
                    )
                continue
            valid_pairs.append(prediction)
    pair_seconds = time.perf_counter() - pair_started
    if not valid_pairs:
        return {}, {
            "possible_pairs": len(pool) * (len(pool) - 1) // 2,
            "compatible_pairs": compatible_pairs,
            "valid_pairs": 0,
            "pair_reason_counts": reason_counts,
            "closest_invalid_direction_margin": closest_invalid_direction_margin,
            "pair_scoring_seconds": pair_seconds,
            "augmentation_seconds": 0.0,
            "path_stop_reason": "no_valid_pair",
            "path": [],
        }

    valid_pairs.sort(key=lambda prediction: (prediction.merit, prediction.edge_ids))
    selected = list(valid_pairs[0].edges)
    snapshots: dict[int, BlockPrediction] = {2: valid_pairs[0]}
    path = [prediction_json(valid_pairs[0], state)]
    augmentation_started = time.perf_counter()
    augmentation_evaluations = 0
    augmentation_reason_counts: dict[str, int] = {}
    path_stop_reason = "pool_exhausted"
    while len(selected) < POOL_LIMIT:
        used_rows = {edge.row for edge in selected}
        used_enterings = {edge.entering for edge in selected}
        candidates: list[BlockPrediction] = []
        for item in pool:
            edge = item.edge
            if edge in selected:
                continue
            if edge.row in used_rows or edge.entering in used_enterings:
                augmentation_reason_counts["matching_conflict"] = (
                    augmentation_reason_counts.get("matching_conflict", 0) + 1
                )
                continue
            augmentation_evaluations += 1
            prediction, reason, diagnostics = predict_block(
                model, state, cache, tuple([*selected, edge])
            )
            if prediction is None:
                augmentation_reason_counts[reason] = augmentation_reason_counts.get(reason, 0) + 1
                if reason == "joint_direction_reversal":
                    closest_invalid_direction_margin = max(
                        closest_invalid_direction_margin,
                        diagnostics["min_joint_direction_margin"],
                    )
                continue
            candidates.append(prediction)
        if not candidates:
            path_stop_reason = "no_compatible_valid_augmentation"
            break
        candidates.sort(key=lambda prediction: (prediction.merit, prediction.edge_ids))
        best = candidates[0]
        selected = list(best.edges)
        path.append(prediction_json(best, state))
        if len(selected) in SNAPSHOT_WIDTHS:
            snapshots[len(selected)] = best
    augmentation_seconds = time.perf_counter() - augmentation_started
    top_pairs = [prediction_json(prediction, state) for prediction in valid_pairs[:10]]
    return snapshots, {
        "possible_pairs": len(pool) * (len(pool) - 1) // 2,
        "compatible_pairs": compatible_pairs,
        "valid_pairs": len(valid_pairs),
        "strict_improving_pairs": sum(
            merit_strictly_less(prediction.merit, state.merit) for prediction in valid_pairs
        ),
        "pair_reason_counts": reason_counts,
        "closest_invalid_direction_margin": closest_invalid_direction_margin,
        "pair_scoring_seconds": pair_seconds,
        "best_pair": prediction_json(valid_pairs[0], state),
        "top_ten_pairs": top_pairs,
        "augmentation_evaluations": augmentation_evaluations,
        "augmentation_reason_counts": augmentation_reason_counts,
        "augmentation_seconds": augmentation_seconds,
        "path_stop_reason": path_stop_reason,
        "path": path,
    }


def compare_authority(
    predicted: BlockPrediction, old_state: State, actual: State
) -> dict[str, Any]:
    relative, base_absolute = _authority_tolerances(old_state, actual)

    def compare(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
        scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
        absolute = base_absolute * scale
        error = float(np.max(np.abs(expected - observed), initial=0.0))
        return {
            "max_abs_error": error,
            "rtol": relative,
            "atol": absolute,
            "passed": bool(np.allclose(expected, observed, rtol=relative, atol=absolute)),
        }

    result = {
        "basis_exact": bool(np.array_equal(predicted.basis, actual.basis)),
        "status_exact": bool(np.array_equal(predicted.status, actual.status)),
        "x_basis": compare(predicted.x_basis, actual.x_basis),
        "y": compare(predicted.y, actual.y),
        "reduced_cost": compare(predicted.reduced_cost, actual.reduced_cost),
        "merit": compare(np.asarray(predicted.merit), np.asarray(actual.merit)),
    }
    result["passed"] = bool(
        result["basis_exact"]
        and result["status_exact"]
        and result["x_basis"]["passed"]
        and result["y"]["passed"]
        and result["reduced_cost"]["passed"]
        and result["merit"]["passed"]
    )
    return result


def run() -> dict[str, Any]:
    os.environ.setdefault("LINPROGX_DS_EXPORT_BASIS", "1")
    os.environ.setdefault("LINPROGX_DS_WARM_START", "1")
    whole_started = time.perf_counter()
    baseline = run_greenbea()
    model, _ = make_model()
    state, seen, reconstruction_seconds = reconstruct_terminal_state(baseline, model)

    edges, candidate_seconds = candidate_edges(model, state)
    pool, cache, scalar_record = scalar_pool(model, state, edges)
    if scalar_record["strict_single_improvers"] != 0:
        raise AssertionError("round-24 state unexpectedly has a strict scalar improver")
    snapshots, interaction = exact_interaction_path(model, state, pool, cache)
    snapshot_json = {
        str(width): prediction_json(prediction, state)
        for width, prediction in sorted(snapshots.items())
    }
    eligible = [
        prediction
        for width, prediction in snapshots.items()
        if width in WIDE_WIDTHS and merit_strictly_less(prediction.merit, state.merit)
    ]
    eligible.sort(key=lambda prediction: (prediction.merit, prediction.edge_ids))

    authority: dict[str, Any] | None = None
    authority_state = state
    authority_factor_attempts = 0
    authority_factor_seconds = 0.0
    cycle = False
    local_pass = False
    if eligible:
        best = eligible[0]
        authority_factor_attempts = 1
        trial = proposed_state(model, state, list(best.edges))
        authority_factor_seconds = trial.factor_seconds
        comparison = compare_authority(best, state, trial)
        actual_improves = merit_strictly_less(trial.merit, state.merit)
        key = state_key(trial)
        cycle = key in seen
        local_pass = bool(comparison["passed"] and actual_improves and not cycle)
        authority = {
            "predicted": prediction_json(best, state),
            "actual_merit": merit_json(trial.merit),
            "actual_strict_improvement": actual_improves,
            "comparison": comparison,
            "state_hash": key,
            "repeated_state": cycle,
            "factor_seconds": trial.factor_seconds,
            "factor_nnz": trial.factor_nnz,
            "factor_growth_proxy": trial.growth_proxy,
            "passed": local_pass,
        }
        if local_pass:
            authority_state = trial

    cert = certificate(model, authority_state)
    accepted_widths = list(EXPECTED_ACCEPTED_WIDTHS)
    if local_pass and authority is not None:
        accepted_widths.append(int(authority["predicted"]["width"]))
    total_accepted = len(accepted_widths)
    total_proposal_attempts = int(baseline["attempted_fresh_factors"]) + authority_factor_attempts
    median_width = statistics.median(accepted_widths)
    whole_seconds = time.perf_counter() - whole_started
    projected_complete = whole_seconds if cert["passed"] else math.inf
    gates = {
        "accepted_count_le_256": total_accepted <= MAX_ACCEPTED,
        "attempted_fresh_factor_count_le_384": total_proposal_attempts <= MAX_ATTEMPTED,
        "median_accepted_width_ge_18": median_width >= MEDIAN_WIDTH_GATE,
        "no_repeated_state": not baseline["cycle_observed"] and not cycle,
        "original_space_certificate": cert["passed"],
        "complete_projected_cost_le_gate": projected_complete <= PROBE_TARGET_SECONDS,
    }
    passed = bool(local_pass and all(gates.values()))
    return {
        "verdict": "PASS_EXACT_SIMULTANEOUS_BLOCK_MERIT_S0"
        if passed
        else "KILL_EXACT_SIMULTANEOUS_BLOCK_MERIT_S0",
        "passed": passed,
        "local_wide_block_pass": local_pass,
        "fixture": "/tmp/lpsuite/lp_greenbea.mat",
        "shape": [model.m, model.n, int(model.a.nnz)],
        "fixed_policy": {
            "activation": "round-24 zero-single-improver state only",
            "deduplication": "(row, entering, leaving_status), fixed source ties",
            "pool": "first 64 by exact scalar old-merit order and fixed ties",
            "pair": "all compatible rank-safe pairs; lex-best exact merit even if worse",
            "augmentation": (
                "one edge per depth, lex-best compatible rank-safe exact simultaneous "
                "old merit regardless interim worsening"
            ),
            "snapshot_widths": list(SNAPSHOT_WIDTHS),
            "s0_pass_widths": list(WIDE_WIDTHS),
            "dense_backward_multiplier": DENSE_BACKWARD_MULTIPLIER,
            "joint_direction_fp_multiplier": JOINT_DIRECTION_FP_MULTIPLIER,
            "eps": EPS,
            "median_width_gate": MEDIAN_WIDTH_GATE,
        },
        "lookahead_authority": {
            "accepted": baseline["accepted"],
            "attempted_fresh_factors": baseline["attempted_fresh_factors"],
            "accepted_widths": baseline["accepted_widths"],
            "final_merit": baseline["final_merit"],
            "round_count": len(baseline["rounds"]),
            "final_round": baseline["rounds"][-1],
            "expected_state_hash": EXPECTED_FINAL_STATE_HASH,
            "reconstructed_state_hash": state_key(state),
            "accepted_state_hash_agreement": state_key(state) == EXPECTED_FINAL_STATE_HASH,
            "reconstruction_factor_seconds": reconstruction_seconds,
        },
        "candidate_generation_seconds": candidate_seconds,
        "scalar_pool": {
            **scalar_record,
            "edge_ids": [item.edge_id for item in pool],
            "merits": [merit_json(item.state.merit) for item in pool],
        },
        "interaction": interaction,
        "snapshots": snapshot_json,
        "eligible_wide_blocks": [prediction_json(prediction, state) for prediction in eligible],
        "authority": authority,
        "authority_factor_attempts": authority_factor_attempts,
        "authority_factor_seconds": authority_factor_seconds,
        "total_accepted": total_accepted,
        "total_proposal_fresh_factor_attempts": total_proposal_attempts,
        "median_accepted_width": median_width,
        "cycle_observed": bool(baseline["cycle_observed"] or cycle),
        "certificate": cert,
        "gates": gates,
        "timing_contract": {
            "control_seconds": CONTROL_SECONDS,
            "board_target_seconds": BOARD_TARGET_SECONDS,
            "probe_target_seconds": PROBE_TARGET_SECONDS,
            "refactor_reference_seconds": REFAC_REFERENCE_SECONDS,
        },
        "component_seconds": {
            "lookahead_reproduction": baseline["algorithm_seconds"],
            "terminal_reconstruction_factor": reconstruction_seconds,
            "rescue_candidate_generation": candidate_seconds,
            "rescue_scalar_pool": scalar_record["seconds"],
            "rescue_pair_scoring": interaction["pair_scoring_seconds"],
            "rescue_augmentation": interaction["augmentation_seconds"],
            "rescue_authority_factor": authority_factor_seconds,
            "whole_probe": whole_seconds,
        },
        "actual_fresh_factor_count": int(baseline["fresh_factor_count"])
        + 1
        + authority_factor_attempts,
        "refactor_only_projection_floor_seconds": (
            (int(baseline["fresh_factor_count"]) + 1 + authority_factor_attempts)
            * REFAC_REFERENCE_SECONDS
        ),
        "projected_complete_cost_seconds": projected_complete,
        "sentinels": {"executed": False, "reason": "bounded S0 ends after one rescue"},
    }


def main() -> None:
    OUT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(OUT_DIR, 0o700)
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    RESULTS.write_text(payload)
    os.chmod(RESULTS, 0o600)
    print(RESULTS)
    print(hashlib.sha256(payload.encode()).hexdigest())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "verdict",
                    "local_wide_block_pass",
                    "eligible_wide_blocks",
                    "median_accepted_width",
                    "certificate",
                    "gates",
                    "component_seconds",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
