"""Block-row uplook gate falsifier probe.

Compares the public sparse auto route with the block-row uplook gate left
automatic against LINPROGX_UPLOOK_BLOCK=4 forced.  The C knob is cached on
first read inside a process, so the driver runs every measured solve in a
fresh worker subprocess and records only the solver wall time reported by
SparseSolver.

Usage:
    PYTHONPATH=. uv run python experiments/blockgate_probe.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from experiments.suite_bench import bounds_of, load_instance
from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

INSTANCES = ("80bau3b", "pilot87", "cre_a", "woodw")
CONFIGS = ("auto", "block4")
SAVEABLE_FRACTIONS = {
    "cre_a": 0.317,
    "woodw": 0.366,
    "80bau3b": 0.446,
    "pilot87": 0.455,
}
DEFAULT_LPSUITE = Path("/tmp/lpsuite")
DEFAULT_OUT = Path("probe_out/blockgate-probe.json")
TRIALS = 9
MAX_ITER = 50_000
EPS = 2e-5
CHECK_INTERVAL = 50_000
TIMEOUT_SECONDS = 180.0
OBJECTIVE_REL_LIMIT = 1e-6
RESIDUAL_LIMIT = EPS


def _worker(path: Path) -> dict[str, Any]:
    data = load_instance(path)
    matrix = from_scipy_sparse(data["A_scipy"])
    result = SparseSolver(
        algorithm="auto",
        max_iterations=MAX_ITER,
        eps=EPS,
        check_interval=CHECK_INTERVAL,
    ).solve(
        SparseLPProblem(
            c=data["c"].tolist(),
            A_eq=matrix,
            b_eq=data["b"].tolist(),
            objective="min",
            bounds=bounds_of(data),
            name=path.stem,
        )
    )
    x = np.array(result.solution.x, dtype=float)
    residual = float(np.max(np.abs(data["A_scipy"] @ x - data["b"]))) if x.size else None
    return {
        "status": result.solution.status.value,
        "backend": result.backend,
        "iterations": result.solution.iterations,
        "objective": result.solution.objective_value,
        "max_residual": residual,
        "wall": result.seconds,
        "message": result.solution.message,
    }


def _run_worker(directory: Path, instance: str, config: str, trial: int) -> dict[str, Any]:
    env = os.environ.copy()
    if config == "auto":
        env.pop("LINPROGX_UPLOOK_BLOCK", None)
    elif config == "block4":
        env["LINPROGX_UPLOOK_BLOCK"] = "4"
    else:  # pragma: no cover - argparse/driver only passes known configs
        msg = f"unknown config {config!r}"
        raise ValueError(msg)

    proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            str(directory / f"lp_{instance}.mat"),
            "--config",
            config,
            "--trial",
            str(trial),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "config": config,
            "trial": trial,
            "status": "worker_failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-2000:],
        }
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row["config"] = config
    row["trial"] = trial
    return row


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _relative_delta(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a))


def _summarize_instance(instance: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_config = {config: [row for row in runs if row["config"] == config] for config in CONFIGS}
    summary: dict[str, Any] = {}
    for config, rows in by_config.items():
        walls = [float(row["wall"]) for row in rows if row.get("wall") is not None]
        statuses = sorted({row.get("status") for row in rows})
        backends = sorted({row.get("backend") for row in rows})
        iterations = sorted({row.get("iterations") for row in rows})
        objectives = [float(row["objective"]) for row in rows if row.get("objective") is not None]
        residuals = [
            float(row["max_residual"]) for row in rows if row.get("max_residual") is not None
        ]
        summary[config] = {
            "trials": len(rows),
            "statuses": statuses,
            "backends": backends,
            "iterations": iterations,
            "wall_median": _median(walls) if walls else None,
            "wall_min": min(walls) if walls else None,
            "wall_max": max(walls) if walls else None,
            "objectives": {
                "min": min(objectives) if objectives else None,
                "max": max(objectives) if objectives else None,
            },
            "max_residual": max(residuals) if residuals else None,
        }

    auto = summary["auto"]
    block = summary["block4"]
    comparable = (
        auto["statuses"] == ["optimal"]
        and block["statuses"] == ["optimal"]
        and len(auto["iterations"]) == 1
        and auto["iterations"] == block["iterations"]
    )
    objective_rel_delta = None
    residual_max = None
    objective_match = False
    residual_match = False
    if comparable:
        auto_objs = [float(row["objective"]) for row in by_config["auto"]]
        block_objs = [float(row["objective"]) for row in by_config["block4"]]
        objective_rel_delta = max(
            _relative_delta(a, b) for a, b in zip(auto_objs, block_objs, strict=True)
        )
        residual_max = max(
            float(row["max_residual"]) for config in CONFIGS for row in by_config[config]
        )
        objective_match = objective_rel_delta <= OBJECTIVE_REL_LIMIT
        residual_match = residual_max <= RESIDUAL_LIMIT

    speedup = None
    flips = False
    if comparable and auto["wall_median"] and block["wall_median"]:
        speedup = 1.0 - float(block["wall_median"]) / float(auto["wall_median"])
        flips = speedup > 0.03 and objective_match and residual_match

    return {
        "instance": instance,
        "saveable_fraction": SAVEABLE_FRACTIONS[instance],
        "configs": summary,
        "comparability": {
            "identical_iterations": auto["iterations"] == block["iterations"],
            "objective_rel_delta_max": objective_rel_delta,
            "objective_match": objective_match,
            "residual_max": residual_max,
            "residual_match": residual_match,
        },
        "forced_block_median_speedup": speedup,
        "verdict": "FLIP" if flips else "KEEP_OFF",
    }


def evaluate(results: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [_summarize_instance(row["instance"], row["runs"]) for row in results]
    abort_reasons = []
    for summary in summaries:
        comp = summary["comparability"]
        if not comp["identical_iterations"]:
            abort_reasons.append(
                f"{summary['instance']}: iterations differ "
                f"{summary['configs']['auto']['iterations']} vs "
                f"{summary['configs']['block4']['iterations']}"
            )
    flips = [summary["instance"] for summary in summaries if summary["verdict"] == "FLIP"]
    near_threshold_flips = [name for name in flips if name in {"80bau3b", "pilot87"}]
    losers = [summary for summary in summaries if summary["verdict"] != "FLIP"]
    threshold = None
    margin_to_losers = None
    if near_threshold_flips:
        threshold = min(SAVEABLE_FRACTIONS[name] for name in near_threshold_flips)
        loser_above = [
            summary["saveable_fraction"]
            for summary in losers
            if summary["saveable_fraction"] < threshold
        ]
        margin_to_losers = None if not loser_above else threshold - max(loser_above)

    return {
        "summaries": summaries,
        "abort_reasons": abort_reasons,
        "flips": flips,
        "near_threshold_flips": near_threshold_flips,
        "recommendation": {
            "change_gate": bool(near_threshold_flips) and not abort_reasons,
            "implied_threshold": threshold,
            "margin_to_highest_gated_off_loser": margin_to_losers,
            "text": _recommendation_text(near_threshold_flips, threshold, margin_to_losers),
        },
    }


def _recommendation_text(
    near_threshold_flips: list[str], threshold: float | None, margin: float | None
) -> str:
    if not near_threshold_flips:
        return "Keep the saveable-fraction gate at 0.5."
    if threshold is None:
        return "Do not change the gate until a threshold can be inferred."
    margin_text = "unknown loser margin" if margin is None else f"{margin:.3f} margin to losers"
    return (
        f"Lower the saveable-fraction gate to at most {threshold:.3f} "
        f"to admit {', '.join(near_threshold_flips)} ({margin_text})."
    )


def run_driver(directory: Path, out: Path, trials: int) -> int:
    results = []
    for instance in INSTANCES:
        runs: list[dict[str, Any]] = []
        for trial in range(1, trials + 1):
            order = CONFIGS if trial % 2 else tuple(reversed(CONFIGS))
            for config in order:
                row = _run_worker(directory, instance, config, trial)
                runs.append(row)
                print(
                    f"{instance:>8} trial={trial} {config:<6} "
                    f"status={row.get('status')} iter={row.get('iterations')} "
                    f"wall={row.get('wall', 0.0):.6f}",
                    flush=True,
                )
                if row.get("status") == "worker_failed":
                    payload = _payload(results + [{"instance": instance, "runs": runs}], trials)
                    payload["evaluation"]["abort_reasons"].append(
                        f"{instance}: worker failed for {config} trial {trial}"
                    )
                    _write(out, payload)
                    return 1
        results.append({"instance": instance, "runs": runs})
        instance_eval = _summarize_instance(instance, runs)
        if not instance_eval["comparability"]["identical_iterations"]:
            payload = _payload(results, trials)
            _write(out, payload)
            print(json.dumps(payload["evaluation"], indent=2, sort_keys=True))
            return 1

    payload = _payload(results, trials)
    _write(out, payload)
    print(json.dumps(payload["evaluation"], indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


def _payload(results: list[dict[str, Any]], trials: int) -> dict[str, Any]:
    return {
        "probe": "block-row-uplook-gate-falsifier",
        "settings": {
            "instances": list(INSTANCES),
            "configs": {
                "auto": {"LINPROGX_UPLOOK_BLOCK": None},
                "block4": {"LINPROGX_UPLOOK_BLOCK": "4"},
            },
            "trials_per_config": trials,
            "route": "SparseSolver algorithm=auto",
            "max_iterations": MAX_ITER,
            "eps": EPS,
            "check_interval": CHECK_INTERVAL,
            "flip_threshold_median_speedup": 0.03,
            "objective_relative_match_limit": OBJECTIVE_REL_LIMIT,
            "residual_match_limit": RESIDUAL_LIMIT,
            "saveable_fractions": SAVEABLE_FRACTIONS,
            "env_cache_note": (
                "LINPROGX_UPLOOK_BLOCK is cached statically by chol_block_size; "
                "each solve is isolated in a fresh subprocess."
            ),
        },
        "results": results,
        "evaluation": evaluate(results),
    }


def _write(out: Path, payload: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_LPSUITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--trial", type=int)
    args = parser.parse_args()

    if args.worker is not None:
        print(json.dumps(_worker(args.worker), sort_keys=True))
        return 0

    return run_driver(args.directory, args.out, args.trials)


if __name__ == "__main__":
    raise SystemExit(main())
