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
The 24 LPnetlib `lp_*.mat` fixtures are stored in the `linprogx-lpsuite`
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

  # full suite single-shot (all 24, lx/highs/clarabel)
  uvx modal run tools/modal_bench.py --action bench --mode suite --ref <sha>

  # certified knife-edge paired verdicts
  uvx modal run tools/modal_bench.py --action bench --mode paired --ref <sha> \
      --instances lp_degen3,lp_osa_14,lp_stocfor3,lp_80bau3b,lp_cre_a,lp_greenbea,lp_cre_b \
      --pairs 7

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

# The 24 LPnetlib fixtures (stems), used as the default suite set.
ALL_INSTANCES = [
    "lp_80bau3b", "lp_cre_a", "lp_cre_b", "lp_cre_d", "lp_d2q06c", "lp_degen3",
    "lp_fit2p", "lp_greenbea", "lp_ken_07", "lp_ken_11", "lp_ken_13", "lp_ken_18",
    "lp_maros_r7", "lp_osa_14", "lp_osa_30", "lp_osa_60", "lp_pds_10", "lp_pds_20",
    "lp_pilot87", "lp_qap12", "lp_qap15", "lp_stocfor3", "lp_truss", "lp_woodw",
]

# Certified knife-edge set for paired verdicts (from docs/HANDOFF.md).
CERTIFIED_SET = [
    "lp_degen3", "lp_osa_14", "lp_stocfor3", "lp_80bau3b", "lp_cre_a",
    "lp_greenbea", "lp_cre_b",
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


def _run_cell(workdir: Path, fixture: Path, solver: str) -> dict[str, Any]:
    """Run one (instance, solver) worker in an isolated subprocess."""
    started = time.perf_counter()
    row: dict[str, Any] = {"solver": solver}
    try:
        proc = subprocess.run(
            [
                "uv", "run", "python",
                "experiments/suite_bench.py",
                "--worker", str(fixture), solver,
            ],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=CELL_TIMEOUT,
            env={**os.environ, "PYTHONPATH": "."},
        )
        if proc.returncode == 0 and proc.stdout.strip():
            row.update(json.loads(proc.stdout.strip().splitlines()[-1]))
        else:
            row.update({
                "status": "crashed",
                "seconds": time.perf_counter() - started,
                "error": proc.stderr.strip()[-400:],
            })
    except subprocess.TimeoutExpired:
        row.update({"status": "timeout", "seconds": CELL_TIMEOUT})
    return row


# --------------------------------------------------------------------------
# Remote function
# --------------------------------------------------------------------------
@app.function(
    image=IMAGE,
    volumes={FIXTURES_DIR: FIXTURES_VOL, SRC_DIR: SRC_VOL},
    cpu=4.0,
    memory=8192,
    timeout=3600,
)
def bench(
    git_ref: str,
    instances: list[str] | None = None,
    pairs: int = 7,
    mode: str = "suite",
    use_snapshot: bool = True,
) -> dict[str, Any]:
    """Build linprogx at git_ref and benchmark on a clean CPU container.

    mode="suite":  run highs/clarabel/linprogx once per instance; rows match
                   experiments/suite_bench.py output shape.
    mode="paired": interleaved lx/HiGHS pairs per instance (pairs repeats);
                   report median/min/wins per side + ratios + verdict.
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
        paired: dict[str, Any] = {}
        for inst in insts:
            fixture = Path(FIXTURES_DIR) / f"{inst}.mat"
            lx_secs: list[float] = []
            hx_secs: list[float] = []
            lx_wins = 0
            lx_status = hx_status = "optimal"
            lx_backend = None
            for _ in range(pairs):
                # interleaved: lx then HiGHS, back to back
                lx = _run_cell(workdir, fixture, "linprogx")
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

            def _stat(xs: list[float]) -> dict[str, Any]:
                if not xs:
                    return {"median": None, "min": None, "n": 0}
                return {"median": statistics.median(xs), "min": min(xs), "n": len(xs)}

            lx_st = _stat(lx_secs)
            hx_st = _stat(hx_secs)
            ratio_median = (
                lx_st["median"] / hx_st["median"]
                if lx_st["median"] and hx_st["median"]
                else None
            )
            ratio_min = (
                lx_st["min"] / hx_st["min"]
                if lx_st["min"] and hx_st["min"]
                else None
            )
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
            paired[inst] = entry
            print(
                f"{inst:>14}: lx med {lx_st['median']} hx med {hx_st['median']} "
                f"ratio {ratio_median} wins {lx_wins}/{pairs} -> {verdict}",
                flush=True,
            )
        out["paired"] = paired
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
            capture_output=True, text=True, check=True,
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
        inst_list = [s for s in (i.strip() for i in instances.split(",")) if s] or None
        result = bench.remote(
            git_ref=ref,
            instances=inst_list,
            pairs=pairs,
            mode=mode,
            use_snapshot=use_snapshot,
        )
        blob = json.dumps(result, indent=2, default=str)
        print(blob)
        short = ref[:12]
        outp = Path(f"/tmp/modal_bench_{short}_{mode}.json")
        outp.write_text(blob)
        print(f"\n[saved] {outp}", file=__import__("sys").stderr)
        return

    raise SystemExit(f"unknown action {action!r}")
