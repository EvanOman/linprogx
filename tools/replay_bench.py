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


def do_reference(conn: sqlite3.Connection, subset: list[str] | None, solvers: tuple[str, ...]) -> None:
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
            print(f"  ref   {path.stem:>14} {solver:>9}: {str(cell['status']):<12} {wtxt}", flush=True)
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
            print(f"== {full[:9]} {subject[:60]}\n   all {len(paths)} cells present, skip", flush=True)
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
                f"   {path.stem:>14}: {str(cell['status']):<12} {wtxt} "
                f"[{cell['route']}]",
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
    n_ref = conn.execute(
        "SELECT COUNT(*) FROM results WHERE commit_hash='reference'"
    ).fetchone()[0]
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    r = sub.add_parser("reference", help="run HiGHS+Clarabel reference cells (idempotent)")
    r.add_argument("--instances", type=str, default=None, help="comma list to restrict fixtures")
    r.add_argument("--solvers", type=str, default=",".join(REFERENCE_SOLVERS))

    p = sub.add_parser("replay", help="checkout+build+run linprogx cells for each commit")
    p.add_argument("commits", nargs="+")
    p.add_argument("--instances", type=str, default=None, help="comma list to restrict fixtures")

    sub.add_parser("status", help="print DB coverage")

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
    elif args.mode == "status":
        do_status(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
