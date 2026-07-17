#!/usr/bin/env python3
"""Extract campaign.db into the self-contained campaign report.

Emits /tmp/campaign_report_data.json, re-embeds the JSON into docs/campaign_report.html,
and prints the single-shot trajectory plus certification summaries.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

DB = "/home/evan/dev/linprogx-perf-worktree/assets/campaign.db"
ROOT = Path("/home/evan/dev/linprogx-perf-worktree")
REPORT_DATA = Path("/tmp/campaign_report_data.json")
REPORT_HTML = ROOT / "docs/campaign_report.html"

PIN4_BOARD = {
    "date": "2026-07-16",
    "label": "AWS us-west-2 pinned canonical board",
    "artifacts": ["pin4_chunk1.json", "pin4_chunk2.json"],
    "summary": "14W-5L-4P plus qap15 coverage = 15 wins",
    "wins": [
        "qap12",
        "ken_18",
        "d2q06c",
        "fit2p",
        "truss",
        "ken_07",
        "ken_11",
        "ken_13",
        "cre_b",
        "maros_r7",
        "cre_d",
        "degen3",
        "pds_20",
        "osa_30",
    ],
    "coverage_wins": ["qap15"],
    "losses": {
        "greenbea": 1.69,
        "osa_60": 1.50,
        "osa_14": 1.34,
        "pds_10": 1.20,
        "cre_a": "4/7 at 0.966",
    },
    "parity": {
        "woodw": 0.996,
        "pilot87": 0.995,
        "80bau3b": 1.010,
        "stocfor3": 1.010,
    },
}

V3_BOARD = {
    "date": "2026-07-16",
    "label": "Protocol v3 median-of-hosts board (AWS us-west-2, 3 hosts x 7 pairs)",
    "artifacts": [
        "modal_bench_c34417761bb6_paired_hosts3.json",
        "modal_bench_b656ef3f8915_paired_hosts3.json",
    ],
    "summary": "16W-2P-6L",
    "wins": [
        "qap12",
        "ken_18",
        "d2q06c",
        "fit2p",
        "truss",
        "ken_07",
        "ken_11",
        "ken_13",
        "cre_b",
        "maros_r7",
        "cre_d",
        "degen3",
        "pds_20",
        "osa_30",
        "pilot87",
    ],
    "coverage_wins": ["qap15"],
    "confirmed_wins": {
        "pilot87": "0.826 (21/21 wins)",
        "pds_20": "0.824 (20/21 wins)",
    },
    "parity": {
        "cre_a": "1.002 (12/21 wins)",
        "stocfor3": "0.999 (12/21 wins)",
    },
    "losses": {
        "greenbea": 1.69,
        "osa_14": 1.42,
        "osa_60": 1.29,
        "pds_10": 1.26,
        "woodw": 1.20,
        "80bau3b": 1.20,
    },
}

# Prior board of record after the H0+H1 census wave (four flips off the v3 board).
CENSUS_BOARD = {
    "date": "2026-07-17",
    "label": "Protocol v3 census-wave board (2026-07-17)",
    # The census-wave artifact re-certifies the seven instances H0+H1 touched;
    # pds_20/pilot87 (wins) and woodw (loss) carry over from the v3 artifacts.
    "artifacts": [
        "modal_bench_928399cf5fea_paired_hosts3.json",
        "modal_bench_c34417761bb6_paired_hosts3.json",
        "modal_bench_b656ef3f8915_paired_hosts3.json",
    ],
    "census_artifact": "modal_bench_928399cf5fea_paired_hosts3.json",
    "prior_core_artifact": "modal_bench_c34417761bb6_paired_hosts3.json",
    "prior_woodw_artifact": "modal_bench_b656ef3f8915_paired_hosts3.json",
    "summary": "20W-0P-4L",
    "wins": [
        "qap12",
        "ken_18",
        "d2q06c",
        "fit2p",
        "truss",
        "ken_07",
        "ken_11",
        "ken_13",
        "cre_b",
        "maros_r7",
        "cre_d",
        "degen3",
        "pds_20",
        "osa_30",
        "pilot87",
        "osa_60",
        "osa_14",
        "cre_a",
        "stocfor3",
    ],
    "coverage_wins": ["qap15"],
    # The four flips the census wave landed off the v3 board.
    "flips": {
        "osa_60": "1.29 -> 0.280 (21/21 wins)",
        "osa_14": "1.42 -> 0.912 (17/21 wins)",
        "cre_a": "1.002 -> 0.939 (18/21 wins)",
        "stocfor3": "0.999 -> 0.962 (17/21 wins)",
    },
    "losses": {
        "greenbea": 1.69,
        "pds_10": "1.26-1.57 (host-dependent PDHG swing)",
        "woodw": 1.20,
        "80bau3b": 1.062,
    },
}

# Board of record after the native equality-row aggregation certification:
# 80bau3b flips to a win and cre_a is reclassified to honest coin-flip parity.
CANONICAL_BOARD = {
    "date": "2026-07-17",
    "label": "Protocol v3 aggregation-era board (2026-07-17)",
    # The aggregation artifact re-certifies the five instances the native
    # equality-row aggregation touched (80bau3b/cre_a/greenbea/d2q06c/ken_07);
    # osa_60/osa_14/stocfor3/pds_10 carry from the census wave, pds_20/pilot87
    # (wins) from the first v3 wave, and woodw (loss) from the v3 knife-edge.
    "artifacts": [
        "modal_bench_70203c413cea_paired_hosts3.json",
        "modal_bench_928399cf5fea_paired_hosts3.json",
        "modal_bench_c34417761bb6_paired_hosts3.json",
        "modal_bench_b656ef3f8915_paired_hosts3.json",
    ],
    "agg_artifact": "modal_bench_70203c413cea_paired_hosts3.json",
    "census_artifact": "modal_bench_928399cf5fea_paired_hosts3.json",
    "prior_core_artifact": "modal_bench_c34417761bb6_paired_hosts3.json",
    "prior_woodw_artifact": "modal_bench_b656ef3f8915_paired_hosts3.json",
    "summary": "20W-1P-3L",
    "wins": [
        "qap12",
        "ken_18",
        "d2q06c",
        "fit2p",
        "truss",
        "ken_07",
        "ken_11",
        "ken_13",
        "cre_b",
        "maros_r7",
        "cre_d",
        "degen3",
        "pds_20",
        "osa_30",
        "pilot87",
        "osa_60",
        "osa_14",
        "stocfor3",
        "80bau3b",
    ],
    "coverage_wins": ["qap15"],
    # The native-aggregation flip off the 20W-0P-4L census-wave board.
    "flips": {
        "80bau3b": "1.062 -> 0.881 (20/21 wins) via native equality-row aggregation",
    },
    "deepened": {
        "d2q06c": "0.371 (21/21 wins)",
        "ken_07": "0.410 (21/21 wins)",
    },
    # cre_a reclassified from the census-wave win to an honest coin flip.
    "parity": {
        "cre_a": "coin flip (0.939 and 1.021 across waves); ~2% reject scan is real on a +-3% cell",
    },
    "losses": {
        "greenbea": 1.741,
        "pds_10": "1.26-1.57 (host-dependent PDHG swing)",
        "woodw": 1.20,
    },
}

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
    "422af49",  # plain-Dantzig leaving on DS auto-rescue routes
    "5f89032",  # presolve V2
    "26a9359",  # chol_setup fast path
    "82cd31d",  # native presolve V2 hot path
]

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def table_exists(name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


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
    "generated": "2026-07-17",
    "canonical_board": CANONICAL_BOARD,
    "prior_boards": [CENSUS_BOARD, V3_BOARD, PIN4_BOARD],
}

if table_exists("bench_artifacts"):
    data["certification_artifacts"] = [
        dict(r)
        for r in conn.execute(
            """
            SELECT artifact, short_ref, mode, certification_date, label, cloud, region,
                   cpu_count, load_start, load_end
            FROM bench_artifacts
            ORDER BY certification_date, artifact
            """
        )
    ]
else:
    data["certification_artifacts"] = []

if table_exists("modal_pairs"):
    data["modal_pairs"] = [
        dict(r)
        for r in conn.execute(
            """
            SELECT artifact, substr(ref,1,12) AS ref, certification_date, label,
                   cloud, region, instance, pairs, lx_median, highs_median, lx_wins,
                   ratio_median, ratio_min, verdict
            FROM modal_pairs
            ORDER BY certification_date, artifact, instance
            """
        )
    ]
    data["pin4_pairs"] = [
        dict(r)
        for r in conn.execute(
            """
            SELECT artifact, instance, pairs, lx_median, highs_median, lx_wins,
                   ratio_median, ratio_min, verdict
            FROM modal_pairs
            WHERE artifact IN ('pin4_chunk1.json', 'pin4_chunk2.json')
            ORDER BY instance
            """
        )
    ]
else:
    data["modal_pairs"] = []
    data["pin4_pairs"] = []

if table_exists("modal_v3_pairs"):
    data["v3_pairs"] = [
        dict(r)
        for r in conn.execute(
            """
            SELECT artifact, substr(ref,1,12) AS ref, certification_date, label,
                   cloud, region, instance, hosts_observed, hosts_with_ratio,
                   pairs_total, lx_wins_total, ratio_median_of_hosts,
                   ratio_min_host, ratio_max_host, verdict
            FROM modal_v3_pairs
            ORDER BY certification_date, artifact, instance
            """
        )
    ]
    data["canonical_pairs"] = [
        dict(r)
        for r in conn.execute(
            """
            SELECT artifact, instance, hosts_observed, pairs_total, lx_wins_total,
                   ratio_median_of_hosts, ratio_min_host, ratio_max_host, verdict
            FROM modal_v3_pairs
            WHERE artifact = :agg
               OR (artifact = :census
                   AND instance IN ('lp_osa_60', 'lp_osa_14', 'lp_stocfor3', 'lp_pds_10'))
               OR (artifact = :prior_core AND instance IN ('lp_pds_20', 'lp_pilot87'))
               OR (artifact = :prior_woodw AND instance = 'lp_woodw')
            ORDER BY ratio_median_of_hosts
            """,
            {
                "agg": CANONICAL_BOARD["agg_artifact"],
                "census": CANONICAL_BOARD["census_artifact"],
                "prior_core": CANONICAL_BOARD["prior_core_artifact"],
                "prior_woodw": CANONICAL_BOARD["prior_woodw_artifact"],
            },
        )
    ]
else:
    data["v3_pairs"] = []
    data["canonical_pairs"] = []

REPORT_DATA.write_text(json.dumps(data, indent=1))
print(f"wrote {REPORT_DATA} ({REPORT_DATA.stat().st_size} bytes)")

html = REPORT_HTML.read_text()
start = html.index('<script id="data" type="application/json">')
start = html.index("\n", start) + 1
end = html.index("\n</script>", start)
embedded = json.dumps(data, separators=(",", ":"))
REPORT_HTML.write_text(html[:start] + embedded + html[end:])
print(f"updated {REPORT_HTML} ({REPORT_HTML.stat().st_size} bytes)")

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

if table_exists("bench_artifacts"):
    print("\n=== CERTIFICATION ARTIFACTS ===")
    for r in conn.execute(
        """
        SELECT certification_date, artifact, short_ref, mode, label, cloud, region
        FROM bench_artifacts
        ORDER BY certification_date DESC, artifact
        """
    ):
        print(
            f"{r['certification_date']} {r['artifact']:<38} {r['short_ref']} "
            f"{r['mode']:<6} {r['cloud'] or 'n/a'} {r['region'] or 'n/a'}  {r['label']}"
        )

if table_exists("modal_v3_pairs"):
    print(f"\n=== CANONICAL BOARD PAIRS ({CANONICAL_BOARD['summary']} aggregation-era) ===")
    for r in conn.execute(
        """
        SELECT instance, pairs_total, lx_wins_total, ratio_median_of_hosts,
               verdict, artifact
        FROM modal_v3_pairs
        WHERE artifact = :agg
           OR (artifact = :census
               AND instance IN ('lp_osa_60', 'lp_osa_14', 'lp_stocfor3', 'lp_pds_10'))
           OR (artifact = :prior_core AND instance IN ('lp_pds_20', 'lp_pilot87'))
           OR (artifact = :prior_woodw AND instance = 'lp_woodw')
        ORDER BY ratio_median_of_hosts
        """,
        {
            "agg": CANONICAL_BOARD["agg_artifact"],
            "census": CANONICAL_BOARD["census_artifact"],
            "prior_core": CANONICAL_BOARD["prior_core_artifact"],
            "prior_woodw": CANONICAL_BOARD["prior_woodw_artifact"],
        },
    ):
        name = r["instance"].replace("lp_", "")
        print(
            f"{name:>10} {r['pairs_total']:2d} pairs {r['lx_wins_total']:2d} lx-wins "
            f"ratio={r['ratio_median_of_hosts']:.3f} {r['verdict']:<12} {r['artifact']}"
        )
