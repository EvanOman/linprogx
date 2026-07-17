#!/usr/bin/env python3
"""Longitudinal replay benchmark harness for the linprogx perf campaign.

Self-contained (stdlib + sqlite3). Populates a per-commit performance record
of linprogx vs the reference solvers (HiGHS, Clarabel) over the 24-fixture
LPnetlib suite, so the improvement arc across the ship commits can be charted.

It has two modes:

  reference   Run the HiGHS + Clarabel cells for all fixtures ONCE. The
              reference solvers do not depend on linprogx source, so their
              numbers are stored under the sentinel commit_hash 'reference'
              and are valid for every linprogx commit. Run from whatever
              commit is currently checked out in the replay worktree.

  replay      Given a list of commit hashes, for each commit: checkout it
              (DETACHED, replay worktree only), rebuild the extension, and
              run the linprogx cell for all 24 fixtures. Rows are inserted
              into the same table keyed by the real commit hash.

Both modes are IDEMPOTENT: a (commit_hash, instance, solver) cell that is
already present is skipped, so the DB can be extended with future commits
or resumed after an interruption.

Design constraints honored:
  * per-instance wall timeout 300s; a timeout records NULL wall + status.
  * builds/benchmarks happen ONLY in the replay worktree; the harness never
    touches any other worktree and only ever checks out commits (detached).
  * machine load average is recorded per run in a `runs` table for honesty.

Usage (run from the replay worktree root):

  # one-time reference pass (resumable; filter to chunk it)
  python3 tools/replay_bench.py reference
  python3 tools/replay_bench.py reference --instances lp_truss,lp_woodw
  python3 tools/replay_bench.py reference --solvers highs

  # replay the linprogx ship commits (checkout + build + 24 cells each)
  python3 tools/replay_bench.py replay a1a355d 0145c8f 86d7064 ...

  # inspect
  python3 tools/replay_bench.py status

  # ingest Modal/paired benchmark artifacts already saved under assets/
  python3 tools/replay_bench.py artifacts
  python3 tools/replay_bench.py artifacts assets/modal_bench_*.json assets/pin4_chunk*.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# --- fixed locations -------------------------------------------------------

# The replay worktree this harness lives in and drives.
WORKTREE = Path(__file__).resolve().parent.parent
# The suite fixtures.
SUITE_DIR = Path("/tmp/lpsuite")
# The campaign DB lives in the perf worktree's assets (a data file, written
# there deliberately — never a benchmark run in that tree).
DB_PATH = Path("/home/evan/dev/linprogx-perf-worktree/assets/campaign.db")

PER_CELL_TIMEOUT = 300.0
REFERENCE_SOLVERS = ("highs", "clarabel")
REFERENCE_TAG = "reference"

ARTIFACT_DATES = {
    "7e9947a": "2026-07-13",
    "1f4351d": "2026-07-14",
    "ecf94bd": "2026-07-16",
    "99ce9c9": "2026-07-16",
    "82cd31d": "2026-07-16",
    "957347b": "2026-07-16",
    "6ec6e2e": "2026-07-16",
    "c344177": "2026-07-16",
    "b656ef3": "2026-07-16",
    "bda0579": "2026-07-16",
    "928399c": "2026-07-17",
    "70203c4": "2026-07-17",
    "592d2c0": "2026-07-17",
    "c5517a2": "2026-07-17",
}

ARTIFACT_LABELS = {
    "7e9947a": "first clean-box validation",
    "1f4351d": "post-presolve-v2 clean-box certification",
    "ecf94bd": "host-conditional knife-edge precision",
    "99ce9c9": "pinned-region setup fast-path certification",
    "82cd31d": "post-native-port paired certification",
    "957347b": "957347b-era paired artifact",
    "6ec6e2e": "canonical board chunk",
    "c344177": "protocol v3 first certification",
    "b656ef3": "v3 knife-edge certification",
    "bda0579": "dense-U on-host envab A/B",
    "928399c": "H0+H1 census-wave certification",
    "70203c4": "native aggregation certification",
    "592d2c0": "on-host IPM slice census",
    "c5517a2": "a2 refactor certification",
}


def fixtures(subset: list[str] | None) -> list[Path]:
    paths = sorted(SUITE_DIR.glob("lp_*.mat"), key=lambda p: p.stat().st_size)
    if subset:
        want = {s if s.startswith("lp_") else f"lp_{s}" for s in subset}
        paths = [p for p in paths if p.stem in want]
    return paths


# --- DB ---------------------------------------------------------------------


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            commit_hash    TEXT NOT NULL,
            commit_date    TEXT,
            commit_subject TEXT,
            instance       TEXT NOT NULL,
            solver         TEXT NOT NULL,
            wall_seconds   REAL,          -- NULL on timeout / crash
            status         TEXT,
            objective      REAL,
            residual       REAL,
            route          TEXT,          -- linprogx backend route (ipm/pdhg/dual_simplex)
            iterations     INTEGER,
            loadavg_1      REAL,
            measured_at    TEXT,
            PRIMARY KEY (commit_hash, instance, solver)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            commit_hash    TEXT,
            commit_date    TEXT,
            commit_subject TEXT,
            solver_group   TEXT,          -- 'reference' or 'linprogx'
            started_at     TEXT,
            finished_at    TEXT,
            loadavg_1      REAL,
            loadavg_5      REAL,
            loadavg_15     REAL,
            n_cells        INTEGER,
            note           TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bench_artifacts (
            artifact       TEXT PRIMARY KEY,
            ref            TEXT NOT NULL,
            short_ref      TEXT NOT NULL,
            mode           TEXT NOT NULL,
            certification_date TEXT,
            label          TEXT,
            cloud          TEXT,
            region         TEXT,
            cpu_count      INTEGER,
            mem_total_kb   INTEGER,
            load_start     TEXT,
            load_end       TEXT,
            imported_at    TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS modal_results (
            artifact       TEXT NOT NULL,
            ref            TEXT NOT NULL,
            certification_date TEXT,
            instance       TEXT NOT NULL,
            solver         TEXT NOT NULL,
            wall_seconds   REAL,
            status         TEXT,
            objective      REAL,
            residual       REAL,
            route          TEXT,
            iterations     INTEGER,
            PRIMARY KEY (artifact, instance, solver)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS modal_pairs (
            artifact       TEXT NOT NULL,
            ref            TEXT NOT NULL,
            certification_date TEXT,
            label          TEXT,
            cloud          TEXT,
            region         TEXT,
            instance       TEXT NOT NULL,
            pairs          INTEGER,
            lx_median      REAL,
            lx_min         REAL,
            lx_n           INTEGER,
            lx_status      TEXT,
            lx_backend     TEXT,
            highs_median   REAL,
            highs_min      REAL,
            highs_n        INTEGER,
            highs_status   TEXT,
            lx_wins        INTEGER,
            ratio_median   REAL,
            ratio_min      REAL,
            verdict        TEXT,
            PRIMARY KEY (artifact, instance)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS modal_v3_pairs (
            artifact       TEXT NOT NULL,
            ref            TEXT NOT NULL,
            certification_date TEXT,
            label          TEXT,
            cloud          TEXT,
            region         TEXT,
            instance       TEXT NOT NULL,
            hosts_observed INTEGER,
            hosts_with_ratio INTEGER,
            pairs_total    INTEGER,
            lx_wins_total  INTEGER,
            ratio_median_of_hosts REAL,
            ratio_min_host REAL,
            ratio_max_host REAL,
            verdict        TEXT,
            PRIMARY KEY (artifact, instance)
        )
        """
    )
    conn.commit()
    return conn


def already_done(conn: sqlite3.Connection, commit: str, instance: str, solver: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM results WHERE commit_hash=? AND instance=? AND solver=?",
        (commit, instance, solver),
    ).fetchone()
    return row is not None


def insert_result(conn: sqlite3.Connection, **kw: object) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO results
          (commit_hash, commit_date, commit_subject, instance, solver,
           wall_seconds, status, objective, residual, route, iterations,
           loadavg_1, measured_at)
        VALUES
          (:commit_hash, :commit_date, :commit_subject, :instance, :solver,
           :wall_seconds, :status, :objective, :residual, :route, :iterations,
           :loadavg_1, :measured_at)
        """,
        kw,
    )
    conn.commit()


def record_run(conn: sqlite3.Connection, **kw: object) -> None:
    conn.execute(
        """
        INSERT INTO runs
          (commit_hash, commit_date, commit_subject, solver_group,
           started_at, finished_at, loadavg_1, loadavg_5, loadavg_15,
           n_cells, note)
        VALUES
          (:commit_hash, :commit_date, :commit_subject, :solver_group,
           :started_at, :finished_at, :loadavg_1, :loadavg_5, :loadavg_15,
           :n_cells, :note)
        """,
        kw,
    )
    conn.commit()


def artifact_paths(paths: list[str] | None) -> list[Path]:
    if paths:
        return sorted(Path(p) for p in paths)
    return sorted(
        list((WORKTREE / "assets").glob("modal_bench_*.json"))
        + list((WORKTREE / "assets").glob("pin4_chunk*.json"))
        + list((WORKTREE / "assets").glob("knife_chunk*.json"))
    )


def artifact_date(ref: str) -> str | None:
    return ARTIFACT_DATES.get(ref[:7])


def artifact_label(ref: str, artifact: str) -> str:
    label = ARTIFACT_LABELS.get(ref[:7], "Modal benchmark artifact")
    if artifact.startswith("pin4_chunk"):
        return "canonical board chunk"
    if artifact.startswith("knife_chunk"):
        return "host-conditional knife-edge precision"
    return label


def artifact_host_metadata(data: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """Return top-level or first-host machine/load metadata."""
    machine = data.get("machine_info")
    loads = data.get("load_checks")
    host_results = data.get("host_results")
    if isinstance(host_results, list) and host_results:
        first_host_obj: object = host_results[0]
        if isinstance(first_host_obj, dict):
            first_host: dict[str, object] = {str(k): v for k, v in first_host_obj.items()}
            machine = machine or first_host.get("machine_info")
            loads = loads or first_host.get("load_checks")
    machine_out: dict[str, object] = (
        {str(k): v for k, v in machine.items()} if isinstance(machine, dict) else {}
    )
    loads_out: dict[str, object] = (
        {str(k): v for k, v in loads.items()} if isinstance(loads, dict) else {}
    )
    return (machine_out, loads_out)


def do_artifacts(conn: sqlite3.Connection, paths: list[str] | None) -> None:
    imported = 0
    result_rows = 0
    pair_rows = 0
    v3_pair_rows = 0
    for path in artifact_paths(paths):
        if not path.exists():
            print(f"missing {path}", flush=True)
            continue
        data = json.loads(path.read_text())
        ref = str(data["ref"])
        mode = str(data["mode"])
        short = ref[:7]
        machine, loads = artifact_host_metadata(data)
        cert_date = artifact_date(ref)
        label = artifact_label(ref, path.name)
        conn.execute(
            """
            INSERT OR REPLACE INTO bench_artifacts
              (artifact, ref, short_ref, mode, certification_date, label, cloud,
               region, cpu_count, mem_total_kb, load_start, load_end, imported_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                path.name,
                ref,
                short,
                mode,
                cert_date,
                label,
                machine.get("modal_cloud"),
                machine.get("modal_region"),
                machine.get("cpu_count"),
                machine.get("mem_total_kb"),
                loads.get("loadavg_at_start"),
                loads.get("loadavg_at_end"),
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
        imported += 1
        for row in data.get("rows", []):
            conn.execute(
                """
                INSERT OR REPLACE INTO modal_results
                  (artifact, ref, certification_date, instance, solver,
                   wall_seconds, status, objective, residual, route, iterations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path.name,
                    ref,
                    cert_date,
                    row.get("instance"),
                    row.get("solver"),
                    row.get("seconds"),
                    row.get("status"),
                    row.get("objective"),
                    row.get("residual"),
                    row.get("backend"),
                    row.get("iterations"),
                ),
            )
            result_rows += 1
        for instance, entry in data.get("paired", {}).items():
            lx = entry.get("lx", {})
            hx = entry.get("hx", {})
            conn.execute(
                """
                INSERT OR REPLACE INTO modal_pairs
                  (artifact, ref, certification_date, label, cloud, region, instance,
                   pairs, lx_median, lx_min, lx_n, lx_status, lx_backend,
                   highs_median, highs_min, highs_n, highs_status, lx_wins,
                   ratio_median, ratio_min, verdict)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path.name,
                    ref,
                    cert_date,
                    label,
                    machine.get("modal_cloud"),
                    machine.get("modal_region"),
                    instance,
                    entry.get("pairs"),
                    lx.get("median"),
                    lx.get("min"),
                    lx.get("n"),
                    lx.get("status"),
                    lx.get("backend"),
                    hx.get("median"),
                    hx.get("min"),
                    hx.get("n"),
                    hx.get("status"),
                    entry.get("lx_wins"),
                    entry.get("ratio_median"),
                    entry.get("ratio_min"),
                    entry.get("verdict"),
                ),
            )
            pair_rows += 1
        v3_paired = data.get("v3", {}).get("paired", {})
        for instance, entry in v3_paired.items():
            pairs_by_host = entry.get("pairs_by_host", [])
            conn.execute(
                """
                INSERT OR REPLACE INTO modal_v3_pairs
                  (artifact, ref, certification_date, label, cloud, region, instance,
                   hosts_observed, hosts_with_ratio, pairs_total, lx_wins_total,
                   ratio_median_of_hosts, ratio_min_host, ratio_max_host, verdict)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path.name,
                    ref,
                    cert_date,
                    label,
                    machine.get("modal_cloud"),
                    machine.get("modal_region"),
                    instance,
                    entry.get("hosts_observed"),
                    entry.get("hosts_with_ratio"),
                    sum(pairs_by_host) if pairs_by_host else None,
                    entry.get("lx_wins_total"),
                    entry.get("ratio_median_of_hosts"),
                    entry.get("ratio_min_host"),
                    entry.get("ratio_max_host"),
                    entry.get("verdict"),
                ),
            )
            v3_pair_rows += 1
        conn.commit()
        shape_note = ""
        if mode == "envab":
            shape_note = " envab=artifact-only"
        print(
            f"  artifact {path.name}: {mode} ref={short} rows={len(data.get('rows', []))} "
            f"pairs={len(data.get('paired', {}))} v3_pairs={len(v3_paired)}{shape_note}",
            flush=True,
        )
    print(
        f"imported {imported} artifacts, upserted {result_rows} suite rows, "
        f"{pair_rows} paired rows, and {v3_pair_rows} v3 paired rows",
        flush=True,
    )


# --- git / build ------------------------------------------------------------


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(WORKTREE), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def commit_meta(commit: str) -> tuple[str, str, str]:
    """Return (full_hash, iso_date, subject) for a commit."""
    full = git("rev-parse", commit)
    date = git("show", "-s", "--format=%cI", commit)
    subject = git("show", "-s", "--format=%s", commit)
    return full, date, subject


def checkout_and_build(commit: str) -> None:
    git("checkout", "--detach", commit)
    subprocess.run(
        ["uv", "sync", "--extra", "dev", "--reinstall-package", "linprogx"],
        cwd=WORKTREE,
        capture_output=True,
        text=True,
        check=True,
    )


# --- cell runner ------------------------------------------------------------


def run_cell(path: Path, solver: str) -> dict[str, object]:
    """Run one (instance, solver) cell via the checked-out suite_bench worker.

    Applies our own 300s wall timeout; a timeout yields NULL wall + status.
    """
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "experiments/suite_bench.py",
                "--worker",
                str(path),
                solver,
            ],
            cwd=WORKTREE,
            capture_output=True,
            text=True,
            timeout=PER_CELL_TIMEOUT,
            env={**os.environ, "PYTHONPATH": "."},
        )
    except subprocess.TimeoutExpired:
        return {
            "wall_seconds": None,
            "status": "timeout",
            "objective": None,
            "residual": None,
            "route": None,
            "iterations": None,
        }
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "wall_seconds": round(time.perf_counter() - started, 4),
            "status": "crashed",
            "objective": None,
            "residual": None,
            "route": None,
            "iterations": None,
        }
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {
        "wall_seconds": payload.get("seconds"),
        "status": payload.get("status"),
        "objective": payload.get("objective"),
        "residual": payload.get("residual"),
        "route": payload.get("backend"),
        "iterations": payload.get("iterations"),
    }


# --- modes ------------------------------------------------------------------


def do_reference(
    conn: sqlite3.Connection, subset: list[str] | None, solvers: tuple[str, ...]
) -> None:
    paths = fixtures(subset)
    la1, la5, la15 = os.getloadavg()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    n = 0
    for path in paths:
        for solver in solvers:
            if already_done(conn, REFERENCE_TAG, path.stem, solver):
                print(f"  skip  {path.stem:>14} {solver:>9} (done)", flush=True)
                continue
            cell = run_cell(path, solver)
            insert_result(
                conn,
                commit_hash=REFERENCE_TAG,
                commit_date=None,
                commit_subject="reference solvers (commit-independent)",
                instance=path.stem,
                solver=solver,
                loadavg_1=os.getloadavg()[0],
                measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                **cell,
            )
            n += 1
            w = cell["wall_seconds"]
            wtxt = f"{w:8.2f}s" if isinstance(w, (int, float)) else "   TIMEOUT"
            print(
                f"  ref   {path.stem:>14} {solver:>9}: {str(cell['status']):<12} {wtxt}", flush=True
            )
    record_run(
        conn,
        commit_hash=REFERENCE_TAG,
        commit_date=None,
        commit_subject="reference solvers",
        solver_group="reference",
        started_at=started,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        loadavg_1=la1,
        loadavg_5=la5,
        loadavg_15=la15,
        n_cells=n,
        note=f"solvers={','.join(solvers)}",
    )


def do_replay(conn: sqlite3.Connection, commits: list[str], subset: list[str] | None) -> None:
    paths = fixtures(subset)
    for commit in commits:
        full, date, subject = commit_meta(commit)
        # Skip build entirely if every cell for this commit is already present.
        pending = [p for p in paths if not already_done(conn, full, p.stem, "linprogx")]
        if not pending:
            print(
                f"== {full[:9]} {subject[:60]}\n   all {len(paths)} cells present, skip", flush=True
            )
            continue
        print(f"== {full[:9]} {date[:10]} {subject[:60]}", flush=True)
        print(f"   checkout + build ({len(pending)} pending cells)...", flush=True)
        checkout_and_build(full)
        la1, la5, la15 = os.getloadavg()
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        n = 0
        for path in pending:
            cell = run_cell(path, "linprogx")
            insert_result(
                conn,
                commit_hash=full,
                commit_date=date,
                commit_subject=subject,
                instance=path.stem,
                solver="linprogx",
                loadavg_1=os.getloadavg()[0],
                measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                **cell,
            )
            n += 1
            w = cell["wall_seconds"]
            wtxt = f"{w:8.2f}s" if isinstance(w, (int, float)) else "   TIMEOUT"
            print(
                f"   {path.stem:>14}: {str(cell['status']):<12} {wtxt} [{cell['route']}]",
                flush=True,
            )
        record_run(
            conn,
            commit_hash=full,
            commit_date=date,
            commit_subject=subject,
            solver_group="linprogx",
            started_at=started,
            finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            loadavg_1=la1,
            loadavg_5=la5,
            loadavg_15=la15,
            n_cells=n,
            note=f"load {la1:.1f}/{la5:.1f}/{la15:.1f} at start",
        )


def do_status(conn: sqlite3.Connection) -> None:
    n_commits = conn.execute(
        "SELECT COUNT(DISTINCT commit_hash) FROM results WHERE solver='linprogx'"
    ).fetchone()[0]
    n_ref = conn.execute("SELECT COUNT(*) FROM results WHERE commit_hash='reference'").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    print(f"linprogx commits: {n_commits}")
    print(f"reference cells : {n_ref}")
    print(f"total result rows: {total}")
    print("\nper-commit linprogx cell counts:")
    for row in conn.execute(
        """
        SELECT substr(commit_hash,1,9) h, commit_date, COUNT(*) n,
               SUM(status='optimal') ok, substr(commit_subject,1,50) s
        FROM results WHERE solver='linprogx'
        GROUP BY commit_hash ORDER BY commit_date
        """
    ):
        print(f"  {row[0]}  {str(row[1])[:10]}  {row[2]:2d} cells  {row[3]} ok  {row[4]}")
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='bench_artifacts'"
    ).fetchone():
        n_artifacts = conn.execute("SELECT COUNT(*) FROM bench_artifacts").fetchone()[0]
        n_pairs = conn.execute("SELECT COUNT(*) FROM modal_pairs").fetchone()[0]
        n_modal = conn.execute("SELECT COUNT(*) FROM modal_results").fetchone()[0]
        print("\nbenchmark artifacts:")
        print(f"  artifacts     : {n_artifacts}")
        print(f"  suite rows    : {n_modal}")
        print(f"  paired rows   : {n_pairs}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("reference", help="run HiGHS+Clarabel reference cells (idempotent)")
    r.add_argument("--instances", type=str, default=None, help="comma list to restrict fixtures")
    r.add_argument("--solvers", type=str, default=",".join(REFERENCE_SOLVERS))

    p = sub.add_parser("replay", help="checkout+build+run linprogx cells for each commit")
    p.add_argument("commits", nargs="+")
    p.add_argument("--instances", type=str, default=None, help="comma list to restrict fixtures")

    sub.add_parser("status", help="print DB coverage")

    a = sub.add_parser("artifacts", help="ingest saved Modal benchmark JSON artifacts")
    a.add_argument(
        "paths",
        nargs="*",
        help="artifact JSON paths; defaults to assets/modal_bench_*.json, pin4_chunk*.json, knife_chunk*.json",
    )

    args = ap.parse_args()
    if shutil.which("uv") is None:
        print("uv not found on PATH", file=sys.stderr)
        return 2
    conn = connect()
    if args.mode == "reference":
        subset = args.instances.split(",") if args.instances else None
        do_reference(conn, subset, tuple(args.solvers.split(",")))
    elif args.mode == "replay":
        subset = args.instances.split(",") if args.instances else None
        do_replay(conn, args.commits, subset)
    elif args.mode == "artifacts":
        do_artifacts(conn, args.paths or None)
    elif args.mode == "status":
        do_status(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
