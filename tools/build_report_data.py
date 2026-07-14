#!/usr/bin/env python3
"""Extract campaign.db into a JSON blob for the report + a narrative table.

Emits scratchpad/report_data.json and prints a per-instance trajectory table.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

DB = "/home/evan/dev/linprogx-perf-worktree/assets/campaign.db"

# Canonical ship order (baseline first, then chronological ship commits).
ORDER_SHORT = [
    "a1a355d",  # baseline
    "0145c8f",  # Tdense fix
    "86d7064",  # resident panels
    "55cae27",  # size-gated BLAS threading
    "e7186c0",  # dense-tail BLAS threshold 400->256
    "29f77a6",  # linear-merge symbolic + regime routing
    "c1812a7",  # contiguous scalar update kernels
    "f919642",  # two-candidate ordering eval
    "0a20b2e",  # MCC gate 5.5->3.0
    "c33f12f",  # DS rate levers
    "2a73a10",  # explicit DS EXPAND config
    "3d53bee",  # mu safeguard
    "cfed6f6",  # DS LU cadence diag-guard 1e6->1e8
    "459c804",  # PDHG auto threads
    "c7190b5",  # mu-gated round-2 IR + kwarg fix
    "e09d425",  # uplook pattern cache
    "d0e6cb1",  # FT program ship (== HEAD code)
    "2f4a1df",  # block-row uplook kernel + saveable-fraction gate (exp-panel merge)
    "11f4157",  # Suhl bounded pivot search (port from exp-leaving)
]

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Resolve full hashes present, map short->full
rows = conn.execute(
    "SELECT DISTINCT commit_hash, commit_date, commit_subject FROM results WHERE solver='linprogx'"
).fetchall()
byshort = {r["commit_hash"][:7]: r for r in rows}

commits = []
for sh in ORDER_SHORT:
    r = byshort[sh]
    commits.append(
        {
            "short": sh,
            "hash": r["commit_hash"],
            "date": r["commit_date"][:10],
            "subject": r["commit_subject"],
        }
    )
full_order = [c["hash"] for c in commits]

# Reference (highs/clarabel) per instance
ref = {}
for r in conn.execute(
    "SELECT instance, solver, wall_seconds, status FROM results WHERE commit_hash='reference'"
):
    ref.setdefault(r["instance"], {})[r["solver"]] = {
        "wall": r["wall_seconds"],
        "status": r["status"],
    }

# linprogx wall per (instance, commit)
lx = {}
for r in conn.execute(
    "SELECT commit_hash, instance, wall_seconds, status, route FROM results WHERE solver='linprogx'"
):
    lx.setdefault(r["instance"], {})[r["commit_hash"]] = {
        "wall": r["wall_seconds"],
        "status": r["status"],
        "route": r["route"],
    }

instances = sorted(lx.keys())

# Build per-instance series (wall over commits, in order)
series = {}
for inst in instances:
    walls = []
    for h in full_order:
        cell = lx[inst].get(h, {})
        walls.append(cell.get("wall"))
    series[inst] = {
        "walls": walls,
        "route_first": lx[inst][full_order[0]].get("route"),
        "route_last": lx[inst][full_order[-1]].get("route"),
        "highs": ref[inst].get("highs", {}).get("wall"),
        "highs_status": ref[inst].get("highs", {}).get("status"),
        "clarabel": ref[inst].get("clarabel", {}).get("wall"),
        "clarabel_status": ref[inst].get("clarabel", {}).get("status"),
    }

# Aggregates per commit: suite total (sum linprogx wall), geomean ratio vs HiGHS
agg = []
for h in full_order:
    total = 0.0
    ratios = []
    for inst in instances:
        w = lx[inst].get(h, {}).get("wall")
        if w is not None:
            total += w
        hw = ref[inst].get("highs", {}).get("wall")
        hstat = ref[inst].get("highs", {}).get("status")
        if w is not None and hw is not None and hstat == "optimal":
            ratios.append(w / hw)
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
    agg.append(
        {
            "total": round(total, 2),
            "geomean_ratio": round(geo, 3) if geo else None,
            "n_ratio": len(ratios),
        }
    )

# Final ratio (last commit) per instance vs HiGHS, for ordering panels
final_ratio = {}
for inst in instances:
    w = series[inst]["walls"][-1]
    hw = series[inst]["highs"]
    if w is not None and hw and series[inst]["highs_status"] == "optimal":
        final_ratio[inst] = w / hw
    else:
        final_ratio[inst] = None


# order: wins first (ratio<1) then losses ascending by ratio; qap15 (no highs) last
def sortkey(inst):
    r = final_ratio[inst]
    return (1e9, inst) if r is None else (r, inst)


ordered_instances = sorted(instances, key=sortkey)

data = {
    "commits": commits,
    "instances": {inst: series[inst] for inst in instances},
    "ordered_instances": ordered_instances,
    "final_ratio": {k: (round(v, 3) if v else None) for k, v in final_ratio.items()},
    "aggregate": agg,
    "generated": "2026-07-13",
}

out = Path(
    "/tmp/claude-1000/-home-evan-dev-linprogx/c9aaa169-d2a9-450d-ad6e-204790d20e27/scratchpad/report_data.json"
)
out.write_text(json.dumps(data, indent=1))
print(f"wrote {out} ({out.stat().st_size} bytes)")

# ---- narrative table to stdout ----
print("\n=== HEADLINE PER-INSTANCE TRAJECTORY (baseline -> current vs HiGHS) ===")
print(
    f"{'instance':>12} {'base':>8} {'final':>8} {'HiGHS':>8} {'ratio':>7} {'route':>9}  {'W/L':>4}"
)
for inst in ordered_instances:
    s = series[inst]
    base = s["walls"][0]
    fin = s["walls"][-1]
    hw = s["highs"]
    r = final_ratio[inst]
    hs = "TIMEOUT" if s["highs_status"] == "timeout" else (f"{hw:.2f}" if hw else "n/a")
    rt = f"{r:.2f}" if r else "n/a"
    wl = "WIN" if (r and r < 1.0) else ("n/a" if r is None else "loss")
    print(f"{inst:>12} {base:8.2f} {fin:8.2f} {hs:>8} {rt:>7} {s['route_last']:>9}  {wl:>4}")

print("\n=== AGGREGATE OVER COMMITS ===")
print(f"{'#':>2} {'short':>8} {'date':>10} {'total_s':>8} {'geomean':>8}  subject")
for i, (c, a) in enumerate(zip(data["commits"], agg, strict=True)):
    print(
        f"{i:2d} {c['short']:>8} {c['date']:>10} {a['total']:8.2f} {str(a['geomean_ratio']):>8}  {c['subject'][:44]}"
    )
