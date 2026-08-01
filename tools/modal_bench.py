"""Modal-based dedicated benchmarking harness for linprogx.

Runs the LPnetlib suite benchmark on a clean, reproducible, no-other-load
CPU container (Modal) instead of the busy dev machine. Absolute wall times
differ across CPUs; the apples-to-apples product is the RATIO of linprogx
wall to HiGHS wall on the same box.

-----------------------------------------------------------------------------
ENVIRONMENT / PINS (documented for reproducibility)
-----------------------------------------------------------------------------
Container image (see `IMAGE` below):
  * base:            modal debian_slim, python 3.12
  * apt:             build-essential, git, libopenblas-dev, pkg-config,
                     curl, ca-certificates
  * pip:             uv (used to build+install the repo from its own uv.lock)
Solver stack pins come from the repo's committed `uv.lock` (uv sync --extra
dev installs the exact locked versions), which currently resolves:
  * scipy   >= 1.14  (HiGHS via scipy.optimize.linprog method="highs")
  * clarabel>= 0.11.1
  * numpy   (locked)
The linprogx C extension `_csparse` links OpenBLAS (libraries=["openblas"],
-DLINPROGX_HAVE_BLAS), hence libopenblas-dev in the image.

Resources: cpu=4.0 (dedicated), memory=8 GiB, timeout=3600s. CPU-ONLY. Never
requests a GPU.

-----------------------------------------------------------------------------
SOURCE SELECTION
-----------------------------------------------------------------------------
Two ways to get linprogx source into the container:

  1. snapshot (default for the perf validation): the local entrypoint runs
     `git archive HEAD` on a local worktree and uploads the clean tarball to
     the `linprogx-src` Modal Volume keyed by the HEAD sha. The container
     extracts + builds it. REQUIRED here because the current perf-branch HEAD
     is NOT pushed to the public GitHub repo (local worktree is ahead of
     origin), so `git clone` cannot reach it.

  2. git clone (for already-public commits): the container clones
     https://github.com/EvanOman/linprogx and checks out git_ref.

-----------------------------------------------------------------------------
FIXTURES
-----------------------------------------------------------------------------
The LPnetlib `lp_*.mat` fixtures are stored in the `linprogx-lpsuite`
Modal Volume (uploaded once from local /tmp/lpsuite via `--upload-fixtures`).
This is more reliable than re-downloading from sparse.tamu.edu per run.

-----------------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------------
  # one-time: upload fixtures + the current worktree HEAD source snapshot
  uvx modal run tools/modal_bench.py --action upload-fixtures
  uvx modal run tools/modal_bench.py --action upload-src

  # smoke test (one small instance, 3 pairs)
  uvx modal run tools/modal_bench.py --action bench --mode paired \
      --ref <sha> --instances lp_woodw --pairs 3

  # full expanded suite (39 instances, lx/highs/clarabel) on 3 hosts
  uvx modal run tools/modal_bench.py --action bench --mode suite --ref <sha> \
      --hosts 3

  # certified knife-edge paired verdicts
  uvx modal run tools/modal_bench.py --action bench --mode paired --ref <sha> \
      --instances lp_degen3,lp_osa_14,lp_stocfor3,lp_80bau3b,lp_cre_a,lp_greenbea,lp_cre_b \
      --pairs 7

  # protocol v3: same paired certification, concurrently across 3 hosts
  uvx modal run tools/modal_bench.py --action bench --mode paired --ref <sha> \
      --instances lp_osa_14,lp_osa_60,lp_pds_10,lp_pds_20,lp_pilot87,lp_woodw,lp_greenbea \
      --pairs 7 --hosts 3

  # protocol v3: on-host environment A/B, concurrently across 3 hosts
  uvx modal run tools/modal_bench.py --action bench --mode envab --ref <sha> \
      --instances lp_greenbea --pairs 7 --hosts 3 \
      --env-a "" --env-b "LINPROGX_DS_FT_DENSE_U=1"

The local entrypoint prints the JSON blob to stdout and saves it to
/tmp/modal_bench_<ref>_<mode>.json locally.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

# --------------------------------------------------------------------------
# App / volumes / image
# --------------------------------------------------------------------------
app = modal.App("linprogx-bench")

FIXTURES_VOL = modal.Volume.from_name("linprogx-lpsuite", create_if_missing=True)
SRC_VOL = modal.Volume.from_name("linprogx-src", create_if_missing=True)

FIXTURES_DIR = "/fixtures"
SRC_DIR = "/src"

PUBLIC_REPO = "https://github.com/EvanOman/linprogx"

# The original 24-cell board, retained so historical artifacts remain legible.
ORIGINAL_INSTANCES = [
    "lp_80bau3b",
    "lp_cre_a",
    "lp_cre_b",
    "lp_cre_d",
    "lp_d2q06c",
    "lp_degen3",
    "lp_fit2p",
    "lp_greenbea",
    "lp_ken_07",
    "lp_ken_11",
    "lp_ken_13",
    "lp_ken_18",
    "lp_maros_r7",
    "lp_osa_14",
    "lp_osa_30",
    "lp_osa_60",
    "lp_pds_10",
    "lp_pds_20",
    "lp_pilot87",
    "lp_qap12",
    "lp_qap15",
    "lp_stocfor3",
    "lp_truss",
    "lp_woodw",
]

# Fifteen additional fixtures selected for route and structural diversity.
# Eight exercise the dual-simplex class that the original board barely
# represented; the remaining seven broaden the IPM and degenerate-network mix.
EXPANDED_INSTANCES = [
    "lp_25fv47",
    "lp_agg2",
    "lp_agg3",
    "lp_bnl2",
    "lp_cycle",
    "lp_degen2",
    "lp_fffff800",
    "lp_fit1p",
    "lp_ganges",
    "lp_greenbeb",
    "lp_israel",
    "lp_pilot",
    "lp_sierra",
    "lp_stocfor2",
    "lp_tuff",
]

ALL_INSTANCES = ORIGINAL_INSTANCES + EXPANDED_INSTANCES

# Certified knife-edge set for paired verdicts (from docs/HANDOFF.md).
CERTIFIED_SET = [
    "lp_degen3",
    "lp_osa_14",
    "lp_stocfor3",
    "lp_80bau3b",
    "lp_cre_a",
    "lp_greenbea",
    "lp_cre_b",
]

SOLVERS = ("highs", "clarabel", "linprogx")
CELL_TIMEOUT = 200.0  # per (instance, solver) worker subprocess

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "git",
        "libopenblas-dev",
        "pkg-config",
        "curl",
        "ca-certificates",
    )
    .pip_install("uv")
)


# --------------------------------------------------------------------------
# Helpers (run inside the container)
# --------------------------------------------------------------------------
def _norm(inst: str) -> str:
    inst = inst.strip()
    if not inst:
        return inst
    return inst if inst.startswith("lp_") else f"lp_{inst}"


def _parse_env_overrides(spec: str) -> dict[str, str]:
    """Parse a comma-separated K=V list; an empty string means no overrides."""
    if not spec:
        return {}

    overrides: dict[str, str] = {}
    for assignment in spec.split(","):
        assignment = assignment.strip()
        if not assignment or "=" not in assignment:
            raise ValueError(f"invalid environment override {assignment!r}; expected K=V")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("environment override key must not be empty")
        if key in overrides:
            raise ValueError(f"duplicate environment override key {key!r}")
        overrides[key] = value
    return overrides


def _machine_info() -> dict[str, Any]:
    model = "unknown"
    ncpu = 0
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
            if line.startswith("processor"):
                ncpu += 1
    except Exception:
        pass
    load = None
    try:
        load = Path("/proc/loadavg").read_text().strip()
    except Exception:
        pass
    memkb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                memkb = int(line.split()[1])
                break
    except Exception:
        pass
    return {
        "cpu_model": model,
        "cpu_count": ncpu,
        "mem_total_kb": memkb,
        "loadavg": load,
        "modal_region": os.environ.get("MODAL_REGION"),
        "modal_cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }


def _prepare_source(git_ref: str, use_snapshot: bool) -> Path:
    """Materialize + build linprogx source, return the build dir."""
    workdir = Path("/root/build")
    if workdir.exists():
        subprocess.run(["rm", "-rf", str(workdir)], check=True)
    workdir.mkdir(parents=True)

    if use_snapshot:
        tar_path = Path(SRC_DIR) / f"{git_ref}.tar"
        if not tar_path.exists():
            raise RuntimeError(
                f"snapshot tarball {tar_path} not found in linprogx-src volume; "
                f"run `--action upload-src` first"
            )
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(workdir)], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "50", PUBLIC_REPO, str(workdir)], check=True)
        subprocess.run(["git", "-C", str(workdir), "checkout", git_ref], check=True)

    # Build: uv sync --extra dev (installs pinned deps from uv.lock + builds C ext).
    t0 = time.perf_counter()
    subprocess.run(
        ["uv", "sync", "--extra", "dev"],
        cwd=str(workdir),
        check=True,
        env={**os.environ, "UV_NO_PROGRESS": "1"},
    )
    build_secs = time.perf_counter() - t0
    print(f"[build] uv sync --extra dev completed in {build_secs:.1f}s", flush=True)
    return workdir


def _run_cell(
    workdir: Path,
    fixture: Path,
    solver: str,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one (instance, solver) worker in an isolated subprocess."""
    started = time.perf_counter()
    row: dict[str, Any] = {"solver": solver}
    worker_env = {**os.environ, "PYTHONPATH": "."}
    if solver == "linprogx" and env_overrides:
        worker_env.update(env_overrides)
    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "experiments/suite_bench.py",
                "--worker",
                str(fixture),
                solver,
            ],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=CELL_TIMEOUT,
            env=worker_env,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
        else:
            row.update(
                {
                    "status": "crashed",
                    "seconds": time.perf_counter() - started,
                    "error": proc.stderr.strip()[-400:],
                }
            )
    except subprocess.TimeoutExpired:
        row.update({"status": "timeout", "seconds": CELL_TIMEOUT})
    return row


def aggregate_protocol_v3_hosts(host_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent paired-protocol host results into v3 verdicts."""
    if not host_results:
        raise ValueError("host_results must not be empty")

    instances = sorted(
        {inst for host_result in host_results for inst in host_result.get("paired", {}).keys()}
    )
    paired: dict[str, Any] = {}

    for inst in instances:
        per_host: list[dict[str, Any]] = []
        ratios: list[float] = []
        wins: list[int] = []
        pairs_by_host: list[int] = []

        for idx, host_result in enumerate(host_results):
            entry = host_result.get("paired", {}).get(inst)
            if entry is None:
                continue

            ratio = entry.get("ratio_median")
            if ratio is not None:
                ratios.append(float(ratio))

            lx_wins = entry.get("lx_wins")
            if lx_wins is not None:
                wins.append(int(lx_wins))

            pairs = entry.get("pairs")
            if pairs is not None:
                pairs_by_host.append(int(pairs))

            per_host.append(
                {
                    "host_index": host_result.get("host_index", idx),
                    "machine_info": host_result.get("machine_info"),
                    "load_checks": host_result.get("load_checks"),
                    "pairs": pairs,
                    "lx": entry.get("lx"),
                    "hx": entry.get("hx"),
                    "lx_wins": lx_wins,
                    "ratio_median": ratio,
                    "ratio_min": entry.get("ratio_min"),
                    "verdict": entry.get("verdict"),
                }
            )

        ratio_median = statistics.median(ratios) if ratios else None
        verdict = None
        if ratio_median is not None:
            verdict = "lx_faster" if ratio_median < 1.0 else "highs_faster"

        paired[inst] = {
            "hosts_observed": len(per_host),
            "hosts_with_ratio": len(ratios),
            "ratio_median_of_hosts": ratio_median,
            "ratio_min_host": min(ratios) if ratios else None,
            "ratio_max_host": max(ratios) if ratios else None,
            "lx_wins_by_host": wins,
            "lx_wins_total": sum(wins),
            "pairs_by_host": pairs_by_host,
            "verdict": verdict,
            "per_host": per_host,
        }

    return {
        "protocol": "v3",
        "hosts": len(host_results),
        "paired": paired,
    }


def aggregate_suite_v3_hosts(host_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate three-solver suite rows across independent clean hosts."""
    if not host_results:
        raise ValueError("host_results must not be empty")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for host_result in host_results:
        for row in host_result.get("rows", []):
            grouped.setdefault(row["instance"], {}).setdefault(row["solver"], []).append(row)

    instances: dict[str, Any] = {}
    for instance, solver_rows in sorted(grouped.items()):
        solvers: dict[str, Any] = {}
        for solver in SOLVERS:
            rows = solver_rows.get(solver, [])
            optimal_rows = [row for row in rows if row.get("status") == "optimal"]
            seconds = [float(row["seconds"]) for row in optimal_rows]
            solvers[solver] = {
                "hosts_observed": len(rows),
                "hosts_optimal": len(optimal_rows),
                "status": "optimal" if len(optimal_rows) == len(host_results) else "incomplete",
                "seconds_median_of_hosts": statistics.median(seconds) if seconds else None,
                "seconds_min_host": min(seconds) if seconds else None,
                "seconds_max_host": max(seconds) if seconds else None,
                "objective": optimal_rows[0].get("objective") if optimal_rows else None,
                "max_residual": max(
                    (
                        float(row["residual"])
                        for row in optimal_rows
                        if row.get("residual") is not None
                    ),
                    default=None,
                ),
                "backend": optimal_rows[0].get("backend") if optimal_rows else None,
                "iterations": optimal_rows[0].get("iterations") if optimal_rows else None,
            }

        lx_seconds = solvers["linprogx"]["seconds_median_of_hosts"]
        highs_seconds = solvers["highs"]["seconds_median_of_hosts"]
        clarabel_seconds = solvers["clarabel"]["seconds_median_of_hosts"]
        instances[instance] = {
            "solvers": solvers,
            "linprogx_over_highs": (
                lx_seconds / highs_seconds if lx_seconds and highs_seconds else None
            ),
            "linprogx_over_clarabel": (
                lx_seconds / clarabel_seconds if lx_seconds and clarabel_seconds else None
            ),
        }

    return {
        "protocol": "suite-v3",
        "hosts": len(host_results),
        "instances": instances,
    }


def aggregate_envab_v3_hosts(host_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent env-A/B host results without changing paired v3."""
    if not host_results:
        raise ValueError("host_results must not be empty")

    instances = sorted(
        {inst for host_result in host_results for inst in host_result.get("envab", {}).keys()}
    )
    envab: dict[str, Any] = {}

    for inst in instances:
        per_host: list[dict[str, Any]] = []
        ratios: list[float] = []
        wins: list[int] = []
        pairs_by_host: list[int] = []

        for idx, host_result in enumerate(host_results):
            entry = host_result.get("envab", {}).get(inst)
            if entry is None:
                continue

            ratio = entry.get("ratio_median")
            if ratio is not None:
                ratios.append(float(ratio))

            lx_b_wins = entry.get("lxB_wins")
            if lx_b_wins is not None:
                wins.append(int(lx_b_wins))

            pairs = entry.get("pairs")
            if pairs is not None:
                pairs_by_host.append(int(pairs))

            per_host.append(
                {
                    "host_index": host_result.get("host_index", idx),
                    "machine_info": host_result.get("machine_info"),
                    "load_checks": host_result.get("load_checks"),
                    "pairs": pairs,
                    "lxA": entry.get("lxA"),
                    "lxB": entry.get("lxB"),
                    "lxB_wins": lx_b_wins,
                    "ratio_median": ratio,
                    "ratio_min": entry.get("ratio_min"),
                    "verdict": entry.get("verdict"),
                }
            )

        ratio_median = statistics.median(ratios) if ratios else None
        verdict = None
        if ratio_median is not None:
            verdict = "lxB_faster" if ratio_median < 1.0 else "lxA_faster"

        envab[inst] = {
            "hosts_observed": len(per_host),
            "hosts_with_ratio": len(ratios),
            "ratio_median_of_hosts": ratio_median,
            "ratio_min_host": min(ratios) if ratios else None,
            "ratio_max_host": max(ratios) if ratios else None,
            "lxB_wins_by_host": wins,
            "lxB_wins_total": sum(wins),
            "pairs_by_host": pairs_by_host,
            "verdict": verdict,
            "per_host": per_host,
        }

    return {
        "protocol": "v3",
        "hosts": len(host_results),
        "envab": envab,
    }


# --------------------------------------------------------------------------
# Remote function
# --------------------------------------------------------------------------
@app.function(
    image=IMAGE,
    volumes={FIXTURES_DIR: FIXTURES_VOL, SRC_DIR: SRC_VOL},
    cpu=4.0,
    memory=8192,
    timeout=3600,
    # Scoreboard-of-record host class: margins are host-conditional
    # (HANDOFF 2026-07-16) — certifications pin cloud AND region; the
    # canonical class is AWS us-west-2 (the 2026-07-14 baseline host).
    # "us-west" alone spans GCP us-west2/us-west4 with different
    # memory bandwidth (osa/pilot87/pds verdicts flip across them).
    cloud="aws",
    region="us-west-2",
)
def bench(
    git_ref: str,
    instances: list[str] | None = None,
    pairs: int = 7,
    mode: str = "suite",
    use_snapshot: bool = True,
    include_raw_pairs: bool = False,
    env_a: dict[str, str] | None = None,
    env_b: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build linprogx at git_ref and benchmark on a clean CPU container.

    mode="suite":  run highs/clarabel/linprogx once per instance; rows match
                   experiments/suite_bench.py output shape.
    mode="paired": interleaved lx/HiGHS pairs per instance (pairs repeats);
                   report median/min/wins per side + ratios + verdict.
    mode="envab":  interleaved linprogx A/B pairs with per-arm environment
                   overrides; report B/A ratios and B wins.
    """
    info_start = _machine_info()
    workdir = _prepare_source(git_ref, use_snapshot)

    if instances:
        insts = [_norm(i) for i in instances]
    else:
        insts = ALL_INSTANCES if mode == "suite" else CERTIFIED_SET

    # verify fixtures present
    missing = [i for i in insts if not (Path(FIXTURES_DIR) / f"{i}.mat").exists()]
    if missing:
        raise RuntimeError(f"missing fixtures in volume: {missing}")

    out: dict[str, Any] = {
        "ref": git_ref,
        "mode": mode,
        "use_snapshot": use_snapshot,
        "machine_info": info_start,
        "load_checks": {
            "loadavg_at_start": info_start.get("loadavg"),
        },
    }

    if mode == "suite":
        rows: list[dict[str, Any]] = []
        for inst in insts:
            fixture = Path(FIXTURES_DIR) / f"{inst}.mat"
            for solver in SOLVERS:
                cell = _run_cell(workdir, fixture, solver)
                cell["instance"] = inst
                rows.append(cell)
                print(
                    f"{inst:>14} {solver:>9}: {cell.get('status'):<10} "
                    f"{cell.get('seconds', 0.0):8.2f}s {cell.get('backend', '')}",
                    flush=True,
                )
        out["rows"] = rows

    elif mode == "paired":
        env_a = env_a or {}
        out["env_a"] = env_a
        paired: dict[str, Any] = {}
        for inst in insts:
            fixture = Path(FIXTURES_DIR) / f"{inst}.mat"
            lx_secs: list[float] = []
            hx_secs: list[float] = []
            pair_results: list[dict[str, Any]] = []
            lx_wins = 0
            lx_status = hx_status = "optimal"
            lx_backend = None
            for pair_index in range(pairs):
                # interleaved: lx then HiGHS, back to back
                lx = _run_cell(workdir, fixture, "linprogx", env_a)
                hx = _run_cell(workdir, fixture, "highs")
                if lx.get("status") == "optimal":
                    lx_secs.append(float(lx["seconds"]))
                    lx_backend = lx.get("backend")
                else:
                    lx_status = lx.get("status", "err")
                if hx.get("status") == "optimal":
                    hx_secs.append(float(hx["seconds"]))
                else:
                    hx_status = hx.get("status", "err")
                if lx.get("status") == "optimal" and hx.get("status") == "optimal":
                    if float(lx["seconds"]) < float(hx["seconds"]):
                        lx_wins += 1
                if include_raw_pairs:
                    pair_results.append(
                        {
                            "pair": pair_index + 1,
                            "lx": lx,
                            "hx": hx,
                            "lx_won": (
                                lx.get("status") == "optimal"
                                and hx.get("status") == "optimal"
                                and float(lx["seconds"]) < float(hx["seconds"])
                            ),
                        }
                    )

            def _stat(xs: list[float]) -> dict[str, Any]:
                if not xs:
                    return {"median": None, "min": None, "n": 0}
                return {"median": statistics.median(xs), "min": min(xs), "n": len(xs)}

            lx_st = _stat(lx_secs)
            hx_st = _stat(hx_secs)
            ratio_median = (
                lx_st["median"] / hx_st["median"] if lx_st["median"] and hx_st["median"] else None
            )
            ratio_min = lx_st["min"] / hx_st["min"] if lx_st["min"] and hx_st["min"] else None
            verdict = None
            if ratio_median is not None:
                verdict = "lx_faster" if ratio_median < 1.0 else "highs_faster"
            entry = {
                "pairs": pairs,
                "lx": {**lx_st, "status": lx_status, "backend": lx_backend},
                "hx": {**hx_st, "status": hx_status},
                "lx_wins": lx_wins,
                "ratio_median": ratio_median,
                "ratio_min": ratio_min,
                "verdict": verdict,
            }
            if include_raw_pairs:
                entry["pair_results"] = pair_results
            paired[inst] = entry
            print(
                f"{inst:>14}: lx med {lx_st['median']} hx med {hx_st['median']} "
                f"ratio {ratio_median} wins {lx_wins}/{pairs} -> {verdict}",
                flush=True,
            )
        out["paired"] = paired
    elif mode == "envab":
        env_a = env_a or {}
        env_b = env_b or {}
        out["env_a"] = env_a
        out["env_b"] = env_b
        envab: dict[str, Any] = {}
        for inst in insts:
            fixture = Path(FIXTURES_DIR) / f"{inst}.mat"
            a_secs: list[float] = []
            b_secs: list[float] = []
            pair_results: list[dict[str, Any]] = []
            lx_b_wins = 0
            a_status = b_status = "optimal"
            a_backend = b_backend = None
            for pair_index in range(pairs):
                # Interleaved on one host: arm A then arm B, back to back.
                lx_a = _run_cell(workdir, fixture, "linprogx", env_a)
                lx_b = _run_cell(workdir, fixture, "linprogx", env_b)
                if lx_a.get("status") == "optimal":
                    a_secs.append(float(lx_a["seconds"]))
                    a_backend = lx_a.get("backend")
                else:
                    a_status = lx_a.get("status", "err")
                if lx_b.get("status") == "optimal":
                    b_secs.append(float(lx_b["seconds"]))
                    b_backend = lx_b.get("backend")
                else:
                    b_status = lx_b.get("status", "err")
                both_optimal = lx_a.get("status") == "optimal" and lx_b.get("status") == "optimal"
                b_won = both_optimal and float(lx_b["seconds"]) < float(lx_a["seconds"])
                if b_won:
                    lx_b_wins += 1
                if include_raw_pairs:
                    pair_results.append(
                        {
                            "pair": pair_index + 1,
                            "lxA": lx_a,
                            "lxB": lx_b,
                            "lxB_won": b_won,
                        }
                    )

            def _stat(xs: list[float]) -> dict[str, Any]:
                if not xs:
                    return {"median": None, "min": None, "n": 0}
                return {"median": statistics.median(xs), "min": min(xs), "n": len(xs)}

            a_st = _stat(a_secs)
            b_st = _stat(b_secs)
            ratio_median = (
                b_st["median"] / a_st["median"] if b_st["median"] and a_st["median"] else None
            )
            ratio_min = b_st["min"] / a_st["min"] if b_st["min"] and a_st["min"] else None
            verdict = None
            if ratio_median is not None:
                verdict = "lxB_faster" if ratio_median < 1.0 else "lxA_faster"
            entry = {
                "pairs": pairs,
                "lxA": {**a_st, "status": a_status, "backend": a_backend},
                "lxB": {**b_st, "status": b_status, "backend": b_backend},
                "lxB_wins": lx_b_wins,
                "ratio_median": ratio_median,
                "ratio_min": ratio_min,
                "verdict": verdict,
            }
            if include_raw_pairs:
                entry["pair_results"] = pair_results
            envab[inst] = entry
            print(
                f"{inst:>14}: A med {a_st['median']} B med {b_st['median']} "
                f"B/A {ratio_median} B wins {lx_b_wins}/{pairs} -> {verdict}",
                flush=True,
            )
        out["envab"] = envab
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out["load_checks"]["loadavg_at_end"] = _machine_info().get("loadavg")
    return out


# --------------------------------------------------------------------------
# Fixture / source upload functions
# --------------------------------------------------------------------------
@app.function(image=IMAGE, volumes={FIXTURES_DIR: FIXTURES_VOL})
def list_fixtures() -> list[str]:
    return sorted(p.name for p in Path(FIXTURES_DIR).glob("*.mat"))


@app.function(image=IMAGE, volumes={SRC_DIR: SRC_VOL})
def list_src() -> list[str]:
    return sorted(p.name for p in Path(SRC_DIR).glob("*.tar"))


# --------------------------------------------------------------------------
# Local entrypoint
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    action: str = "bench",
    ref: str = "",
    mode: str = "suite",
    instances: str = "",
    pairs: int = 7,
    hosts: int = 1,
    env_a: str = "",
    env_b: str = "",
    use_snapshot: bool = True,
    worktree: str = "/home/evan/dev/linprogx-perf-worktree",
    local_fixtures: str = "/tmp/lpsuite",
):
    """Actions: upload-fixtures | upload-src | bench | list."""
    if action == "upload-fixtures":
        mats = sorted(Path(local_fixtures).glob("lp_*.mat"))
        if not mats:
            raise SystemExit(f"no lp_*.mat in {local_fixtures}")
        print(f"uploading {len(mats)} fixtures to linprogx-lpsuite volume ...")
        with FIXTURES_VOL.batch_upload(force=True) as batch:
            for m in mats:
                batch.put_file(str(m), f"/{m.name}")
        print("done. fixtures in volume:")
        print("\n".join(list_fixtures.remote()))
        return

    if action == "upload-src":
        sha = subprocess.run(
            ["git", "-C", worktree, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        tar_local = Path(f"/tmp/linprogx_src_{sha}.tar")
        # clean archive of tracked files at HEAD, under a top-level dir stripped
        subprocess.run(
            ["git", "-C", worktree, "archive", "--format=tar", "-o", str(tar_local), "HEAD"],
            check=True,
        )
        print(f"archived HEAD {sha} -> {tar_local} ({tar_local.stat().st_size} bytes)")
        with SRC_VOL.batch_upload(force=True) as batch:
            batch.put_file(str(tar_local), f"/{sha}.tar")
        print(f"uploaded snapshot as /{sha}.tar")
        print("snapshots in volume:")
        print("\n".join(list_src.remote()))
        return

    if action == "list":
        print("fixtures:", list_fixtures.remote())
        print("src snapshots:", list_src.remote())
        return

    if action == "bench":
        if not ref:
            raise SystemExit("--ref required for bench")
        if hosts < 1:
            raise SystemExit("--hosts must be >= 1")
        if hosts > 1 and mode not in {"suite", "paired", "envab"}:
            raise SystemExit("--hosts > 1 requires --mode suite, paired, or envab")
        if mode in {"paired", "envab"}:
            try:
                env_a_overrides = _parse_env_overrides(env_a)
                env_b_overrides = _parse_env_overrides(env_b) if mode == "envab" else {}
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        else:
            env_a_overrides = env_b_overrides = {}
        inst_list = [s for s in (i.strip() for i in instances.split(",")) if s] or None
        short = ref[:12]
        if hosts > 1:
            if mode == "envab":
                calls = [
                    (
                        ref,
                        inst_list,
                        pairs,
                        mode,
                        use_snapshot,
                        True,
                        env_a_overrides,
                        env_b_overrides,
                    )
                    for _ in range(hosts)
                ]
            else:
                calls = [
                    (
                        ref,
                        inst_list,
                        pairs,
                        mode,
                        use_snapshot,
                        True,
                        env_a_overrides,
                        {},
                    )
                    for _ in range(hosts)
                ]
            host_results: list[dict[str, Any]] = []
            for idx, host_result in enumerate(bench.starmap(calls)):
                host_result["host_index"] = idx
                host_results.append(host_result)

            if mode == "suite":
                v3 = aggregate_suite_v3_hosts(host_results)
            elif mode == "envab":
                v3 = aggregate_envab_v3_hosts(host_results)
            else:
                v3 = aggregate_protocol_v3_hosts(host_results)
            default_instances = ALL_INSTANCES if mode == "suite" else CERTIFIED_SET
            result = {
                "ref": ref,
                "mode": mode,
                "protocol": "v3",
                "hosts": hosts,
                "pairs": pairs,
                "use_snapshot": use_snapshot,
                "instances": [_norm(i) for i in inst_list] if inst_list else default_instances,
                "host_results": host_results,
                "v3": v3,
            }
            if mode in {"paired", "envab"}:
                result["env_a"] = env_a_overrides
            if mode == "envab":
                result["env_b"] = env_b_overrides
            blob = json.dumps(result, indent=2, default=str)
            print(blob)
            outp = Path(f"/tmp/modal_bench_{short}_{mode}_hosts{hosts}.json")
            outp.write_text(blob)
            print(f"\n[saved] {outp}", file=__import__("sys").stderr)
            return

        bench_kwargs = {
            "git_ref": ref,
            "instances": inst_list,
            "pairs": pairs,
            "mode": mode,
            "use_snapshot": use_snapshot,
        }
        if mode == "paired":
            bench_kwargs["env_a"] = env_a_overrides
        if mode == "envab":
            bench_kwargs["env_a"] = env_a_overrides
            bench_kwargs["env_b"] = env_b_overrides
        result = bench.remote(**bench_kwargs)
        blob = json.dumps(result, indent=2, default=str)
        print(blob)
        outp = Path(f"/tmp/modal_bench_{short}_{mode}.json")
        outp.write_text(blob)
        print(f"\n[saved] {outp}", file=__import__("sys").stderr)
        return

    raise SystemExit(f"unknown action {action!r}")
