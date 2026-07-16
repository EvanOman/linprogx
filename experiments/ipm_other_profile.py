"""Attribute the IPM wall slice outside loop refactor + Newton timers.

Read-only with respect to solver sources.  The parent process runs isolated
workers so C-level debug stderr can be parsed without mutating global env in a
long-lived process.

Usage:
    PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/ipm_other_profile.py \
        --instances lp_degen3,lp_cre_a,lp_woodw,lp_80bau3b,lp_stocfor3 --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_INSTANCES = (
    "lp_degen3",
    "lp_cre_a",
    "lp_stocfor3",
    "lp_80bau3b",
    "lp_woodw",
    "lp_cre_d",
    "lp_maros_r7",
)


def load_instance(path: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A_scipy": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def bounds_of(data: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    return [
        (
            None if low == float("-inf") else float(low),
            None if up == float("inf") else float(up),
        )
        for low, up in zip(data["lo"], data["hi"], strict=True)
    ]


def prepare(data: dict[str, Any]) -> dict[str, Any]:
    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import csr_matrix, from_scipy_sparse

    t0 = time.perf_counter()
    matrix = from_scipy_sparse(data["A_scipy"])
    matrix_s = time.perf_counter() - t0

    c = [float(v) for v in data["c"]]
    b = [float(v) for v in data["b"]]
    lo = [float(v) for v in data["lo"]]
    hi = [float(v) for v in data["hi"]]

    t0 = time.perf_counter()
    reduction = presolve_matrix(matrix, b, c, lo, hi)
    presolve_s = time.perf_counter() - t0
    if reduction is not None:
        if reduction._matrix is not None:
            matrix = reduction._matrix
        else:
            matrix = csr_matrix(
                reduction.rows,
                reduction.cols,
                reduction.indptr,
                reduction.indices,
                reduction.data,
            )
        c, b, lo, hi = reduction.c, reduction.b, reduction.lo, reduction.hi
    return {
        "matrix": matrix,
        "c": c,
        "b": b,
        "lo": lo,
        "hi": hi,
        "matrix_s": matrix_s,
        "presolve_s": presolve_s,
        "reduced_shape": matrix.shape,
        "reduced_nnz": matrix.nnz,
    }


def run_public(data: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    matrix = from_scipy_sparse(data["A_scipy"])
    problem = SparseLPProblem(
        c=data["c"].tolist(),
        A_eq=matrix,
        b_eq=data["b"].tolist(),
        objective="min",
        bounds=bounds_of(data),
    )
    t0 = time.perf_counter()
    result = SparseSolver(
        algorithm="auto", max_iterations=50_000, eps=2e-5, check_interval=50_000
    ).solve(problem)
    wall = time.perf_counter() - t0
    x = np.array(result.solution.x, dtype=float)
    return {
        "wall_s": wall,
        "status": result.solution.status.value,
        "backend": result.backend,
        "iterations": result.solution.iterations,
        "residual": float(np.max(np.abs(data["A_scipy"] @ x - data["b"]))),
    }


def run_direct(data: dict[str, Any], *, max_iter: int, debug: bool) -> dict[str, Any]:
    prep = prepare(data)
    t0 = time.perf_counter()
    result = prep["matrix"].solve_eq_box_ipm(
        prep["c"],
        prep["b"],
        prep["lo"],
        prep["hi"],
        max_iter=max_iter,
        tol=1e-9,
        debug=debug,
        threads=0,
        feas_tol=2e-5,
    )
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "status": result["status"],
        "iterations": int(result["iterations"]),
        "objective": float(result["objective"]),
        "matrix_s": prep["matrix_s"],
        "presolve_s": prep["presolve_s"],
        "reduced_shape": prep["reduced_shape"],
        "reduced_nnz": prep["reduced_nnz"],
    }


def run_fingerprint(data: dict[str, Any]) -> dict[str, Any]:
    prep = prepare(data)
    return prep["matrix"].cholesky_symbolic_fingerprint()


def worker(args: argparse.Namespace) -> int:
    data = load_instance(args.data_dir / f"{args.instance}.mat")
    if args.mode == "public":
        out = run_public(data)
    elif args.mode == "fingerprint":
        out = run_fingerprint(data)
    else:
        out = run_direct(data, max_iter=args.max_iter, debug=args.debug)
    print(json.dumps(out, sort_keys=True))
    return 0


SETUP_RE = re.compile(r"chol_setup\s+([a-z-]+)\s+([0-9.]+)s")
SETUP_PROFILE_RE = re.compile(r"chol_setup_profile\s+([a-z0-9-]+)\s+([0-9.]+)")
MCC_GATE_RE = re.compile(r"ipm mcc gate: .* ratio=([0-9.]+) budget=([0-9-]+)")
MCC_RE = re.compile(r"ipm mcc: budget=([0-9-]+) accepted_rounds=([0-9-]+)")
SAFE_RE = re.compile(r"ipm safeguard: shrinks=([0-9-]+) breaks=([0-9-]+)")
LAG_RE = re.compile(r"ipm lag: attempts=([0-9-]+) accepts=([0-9-]+) redos=([0-9-]+)")
TIMERS_RE = re.compile(r"ipm timers: refactor=([0-9.]+)s newton_solves=([0-9.]+)s")
LOOP_PROFILE_RE = re.compile(r"ipm loop profile:\s+(.*)")
EXIT_RE = re.compile(
    r"ipm exit: status=([a-z_]+) best_gap=([0-9.e+-]+|inf) best_pres=([0-9.e+-]+|inf) "
    r"best_raw=([0-9.e+-]+|inf) best_dres=([0-9.e+-]+|inf) best_mu=([0-9.e+-]+|inf)"
)
REFAC_RE = re.compile(
    r"refac phases: assemble=([0-9.]+) uplook=([0-9.]+) dpotrf=([0-9.]+) "
    r"copyback=([0-9.]+) solve_tail=([0-9.]+)"
)
TAIL_RE = re.compile(r"chol_setup tail: m=([0-9]+) tail_start=([0-9]+) tail_len=([0-9]+)")
FILL_RE = re.compile(r"chol_setup fill: nnzL=([0-9.]+) flops=([0-9.e+-]+)")


def parse_stderr(stderr: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"setup_phases_s": {}, "setup_profile_s": {}}
    for label, value in SETUP_RE.findall(stderr):
        parsed["setup_phases_s"][label] = float(value)
    for label, value in SETUP_PROFILE_RE.findall(stderr):
        parsed["setup_profile_s"][label] = float(value)
    if m := MCC_GATE_RE.search(stderr):
        parsed["mcc_gate_ratio"] = float(m.group(1))
        parsed["mcc_gate_budget"] = int(m.group(2))
    if m := MCC_RE.search(stderr):
        parsed["mcc_budget"] = int(m.group(1))
        parsed["mcc_accepted_rounds"] = int(m.group(2))
    if m := SAFE_RE.search(stderr):
        parsed["safeguard_shrinks"] = int(m.group(1))
        parsed["safeguard_breaks"] = int(m.group(2))
    if m := LAG_RE.search(stderr):
        parsed["lag_attempts"] = int(m.group(1))
        parsed["lag_accepts"] = int(m.group(2))
        parsed["lag_redos"] = int(m.group(3))
    if m := TIMERS_RE.search(stderr):
        parsed["timer_refactor_s"] = float(m.group(1))
        parsed["timer_newton_s"] = float(m.group(2))
    if m := LOOP_PROFILE_RE.search(stderr):
        loop_profile: dict[str, int | float] = {}
        for part in m.group(1).split():
            key, value = part.split("=", 1)
            if key in {"iterations", "best_updates", "safeguard_checks"}:
                loop_profile[key] = int(value)
            else:
                loop_profile[key] = float(value)
        parsed["loop_profile_s"] = loop_profile
    if m := EXIT_RE.search(stderr):
        parsed["exit_status"] = m.group(1)
        parsed["best_gap"] = float(m.group(2))
        parsed["best_pres"] = float(m.group(3))
        parsed["best_raw"] = float(m.group(4))
        parsed["best_dres"] = float(m.group(5))
        parsed["best_mu"] = float(m.group(6))
    if m := REFAC_RE.search(stderr):
        parsed["refac_phases_s"] = {
            "assemble": float(m.group(1)),
            "uplook": float(m.group(2)),
            "dpotrf": float(m.group(3)),
            "copyback": float(m.group(4)),
            "solve_tail": float(m.group(5)),
        }
    if m := TAIL_RE.search(stderr):
        parsed["chol_m"] = int(m.group(1))
        parsed["tail_start"] = int(m.group(2))
        parsed["tail_len"] = int(m.group(3))
    if m := FILL_RE.search(stderr):
        parsed["nnzL"] = float(m.group(1))
        parsed["factor_flops"] = float(m.group(2))
    return parsed


def invoke(
    instance: str,
    mode: str,
    data_dir: Path,
    *,
    max_iter: int = 200,
    debug: bool = False,
    env: dict[str, str] | None = None,
) -> tuple[dict[str, Any], str]:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--instance",
        instance,
        "--mode",
        mode,
        "--max-iter",
        str(max_iter),
        "--data-dir",
        str(data_dir),
    ]
    if debug:
        cmd.append("--debug")
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    proc = subprocess.run(cmd, text=True, capture_output=True, check=True, env=child_env)
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def instance_profile(instance: str, data_dir: Path, runs: int) -> dict[str, Any]:
    public_runs = [invoke(instance, "public", data_dir)[0] for _ in range(runs)]
    direct_runs = [invoke(instance, "direct", data_dir, max_iter=200)[0] for _ in range(runs)]
    zero_runs = [invoke(instance, "direct", data_dir, max_iter=0)[0] for _ in range(runs)]
    one_runs = [invoke(instance, "direct", data_dir, max_iter=1)[0] for _ in range(runs)]

    debug_env = {
        "LINPROGX_CHOL_DEBUG": "1",
        "LINPROGX_CHOL_SETUP_PROFILE": "1",
        "LINPROGX_IPM_LOOP_PROFILE": "1",
        "LINPROGX_REFAC_PROFILE": "1",
    }
    debug_full, debug_full_stderr = invoke(
        instance, "direct", data_dir, max_iter=200, debug=True, env=debug_env
    )
    debug_zero, debug_zero_stderr = invoke(
        instance, "direct", data_dir, max_iter=0, debug=True, env=debug_env
    )
    parsed_full = parse_stderr(debug_full_stderr)
    parsed_zero = parse_stderr(debug_zero_stderr)
    fingerprint = invoke(instance, "fingerprint", data_dir)[0]

    public_wall = median([r["wall_s"] for r in public_runs])
    direct_wall = median([r["wall_s"] for r in direct_runs])
    zero_wall = median([r["wall_s"] for r in zero_runs])
    one_wall = median([r["wall_s"] for r in one_runs])
    presolve = median([r.get("presolve_s", 0.0) for r in direct_runs])
    matrix_build = median([r.get("matrix_s", 0.0) for r in direct_runs])

    setup_sum = sum(parsed_full.get("setup_phases_s", {}).values())
    setup_zero_sum = sum(parsed_zero.get("setup_phases_s", {}).values())
    refactor_timer = parsed_full.get("timer_refactor_s", 0.0)
    newton_timer = parsed_full.get("timer_newton_s", 0.0)

    refac_phases = parsed_full.get("refac_phases_s", {})
    refac_zero_phases = parsed_zero.get("refac_phases_s", {})
    startup_refac = max(
        0.0,
        sum(refac_zero_phases.get(k, 0.0) for k in ("assemble", "uplook", "dpotrf", "copyback")),
    )
    startup_solve_tail = refac_zero_phases.get("solve_tail", 0.0)
    loop_refac_phases = {
        k: max(0.0, refac_phases.get(k, 0.0) - refac_zero_phases.get(k, 0.0))
        for k in ("assemble", "uplook", "dpotrf", "copyback")
    }
    loop_solve_tail = max(
        0.0, refac_phases.get("solve_tail", 0.0) - refac_zero_phases.get("solve_tail", 0.0)
    )
    wrapper = max(0.0, public_wall - direct_wall)
    preloop_other = max(0.0, zero_wall - setup_zero_sum - startup_refac - startup_solve_tail)
    first_iter_increment = max(0.0, one_wall - zero_wall)

    known_native = refactor_timer + newton_timer
    native_other = max(0.0, direct_wall - known_native)
    loop_misc = max(0.0, direct_wall - zero_wall - refactor_timer - newton_timer)

    return {
        "instance": instance,
        "public_wall_s": public_wall,
        "direct_wall_s": direct_wall,
        "direct_wall_runs_s": [r["wall_s"] for r in direct_runs],
        "public_wall_runs_s": [r["wall_s"] for r in public_runs],
        "status": direct_runs[-1]["status"],
        "iterations": direct_runs[-1]["iterations"],
        "reduced_shape": direct_runs[-1].get("reduced_shape"),
        "reduced_nnz": direct_runs[-1].get("reduced_nnz"),
        "matrix_build_s": matrix_build,
        "presolve_s": presolve,
        "setup_measured_s": setup_sum,
        "setup_zero_measured_s": setup_zero_sum,
        "startup_refac_s": startup_refac,
        "startup_solve_tail_s": startup_solve_tail,
        "zero_wall_s": zero_wall,
        "one_wall_s": one_wall,
        "first_iter_increment_s": first_iter_increment,
        "timer_refactor_s": refactor_timer,
        "timer_newton_s": newton_timer,
        "native_other_s": native_other,
        "loop_misc_est_s": loop_misc,
        "preloop_other_est_s": preloop_other,
        "wrapper_route_est_s": wrapper,
        "loop_refac_phases_s": loop_refac_phases,
        "loop_solve_tail_s": loop_solve_tail,
        "debug_full": parsed_full,
        "debug_zero": parsed_zero,
        "symbolic_fingerprint": fingerprint,
    }


def write_tables(rows: list[dict[str, Any]]) -> None:
    components = [
        ("python route/presolve/postsolve", "wrapper_route_est_s"),
        ("native setup/order", "setup_measured_s"),
        ("scaling+initial point+marshal est", "preloop_other_est_s"),
        ("loop refactor timer", "timer_refactor_s"),
        ("loop Newton timer", "timer_newton_s"),
        ("loop residual/gap/step/update est", "loop_misc_est_s"),
    ]
    print("| component | " + " | ".join(r["instance"] for r in rows) + " |")
    print("| --- | " + " | ".join("---:" for _ in rows) + " |")
    for label, key in components:
        cells = []
        for row in rows:
            wall = row["direct_wall_s"]
            if key == "wrapper_route_est_s":
                wall = row["public_wall_s"]
            value = row[key]
            cells.append(f"{value * 1e6:.0f} us ({(value / wall * 100.0 if wall else 0):.1f}%)")
        print("| " + label + " | " + " | ".join(cells) + " |")
    print()
    loop_labels = [
        ("residual matvecs", "resid_matvec"),
        ("residual scans/norms", "resid_scan"),
        ("best-iterate gap scan", "best_gap"),
        ("best-iterate copies", "best_copy"),
        ("certificate/exit gates", "exit_gate"),
        ("H/D assembly", "hessian"),
        ("Mehrotra RHS assembly", "rhs"),
        ("sigma computation", "sigma"),
        ("step ratio tests", "step_ratio"),
        ("Gondzio corrector misc", "mcc"),
        ("mu safeguard", "mu_safeguard"),
        ("iterate update", "update"),
    ]
    print("| loop component | " + " | ".join(r["instance"] for r in rows) + " |")
    print("| --- | " + " | ".join("---:" for _ in rows) + " |")
    for label, key in loop_labels:
        cells = []
        for row in rows:
            value = row["debug_full"].get("loop_profile_s", {}).get(key, 0.0)
            wall = row["direct_wall_s"]
            cells.append(
                f"{float(value) * 1e6:.0f} us ({(float(value) / wall * 100.0 if wall else 0):.1f}%)"
            )
        print("| " + label + " | " + " | ".join(cells) + " |")
    print(
        "| profile counts | "
        + " | ".join(
            f"best={row['debug_full'].get('loop_profile_s', {}).get('best_updates', 0)}, "
            f"safe={row['debug_full'].get('loop_profile_s', {}).get('safeguard_checks', 0)}"
            for row in rows
        )
        + " |"
    )
    print()
    print("| instance | wall | top non-refactor/Newton component | evidence |")
    print("| --- | ---: | --- | --- |")
    for row in rows:
        candidates = [
            ("native setup/order", row["setup_measured_s"]),
            ("scaling+initial point+marshal est", row["preloop_other_est_s"]),
            ("loop residual/gap/step/update est", row["loop_misc_est_s"]),
            ("python route/presolve/postsolve", row["wrapper_route_est_s"]),
        ]
        top_name, top_value = max(candidates, key=lambda item: item[1])
        print(
            f"| {row['instance']} | {row['direct_wall_s']:.6f}s native / "
            f"{row['public_wall_s']:.6f}s public | {top_name} "
            f"{top_value * 1e6:.0f} us | iters={row['iterations']}, "
            f"tail={row['debug_full'].get('tail_len', 'n/a')}, "
            f"mcc={row['debug_full'].get('mcc_budget', 'n/a')}/"
            f"{row['debug_full'].get('mcc_accepted_rounds', 'n/a')} |"
        )
    print()
    labels = sorted(
        {
            label
            for row in rows
            for label in row["debug_full"].get("setup_profile_s", {})
            if label != "total"
        }
    )
    print("| setup phase | " + " | ".join(r["instance"] for r in rows) + " |")
    print("| --- | " + " | ".join("---:" for _ in rows) + " |")
    for label in labels:
        cells = []
        for row in rows:
            value = row["debug_full"].get("setup_profile_s", {}).get(label, 0.0)
            cells.append(f"{value * 1e3:.2f} ms")
        print("| " + label + " | " + " | ".join(cells) + " |")
    totals = [
        f"{row['debug_full'].get('setup_profile_s', {}).get('total', 0.0) * 1e3:.2f} ms"
        for row in rows
    ]
    print("| total | " + " | ".join(totals) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/lpsuite"))
    parser.add_argument("--instances", default=",".join(DEFAULT_INSTANCES))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/ipm_other_profile_results.json")
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--instance", default="")
    parser.add_argument("--mode", choices=("public", "direct", "fingerprint"), default="direct")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args)

    instances = [
        name if name.startswith("lp_") else f"lp_{name}" for name in args.instances.split(",")
    ]
    rows = [instance_profile(instance, args.data_dir, args.runs) for instance in instances]
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_tables(rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
