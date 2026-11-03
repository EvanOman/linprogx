#!/usr/bin/env python3
"""Build the reproducible LPnetlib comparison table and README charts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

SOLVERS = ("linprogx", "highs", "clarabel")
LABELS = {"linprogx": "linprogx", "highs": "HiGHS", "clarabel": "Clarabel"}
COLORS = {"linprogx": "#087f5b", "highs": "#e8590c", "clarabel": "#7048e8"}
ROUTE_COLORS = {"simplex": "#087f5b", "ipm": "#1971c2", "pdhg": "#9c36b5"}


def _suite_instances(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        instances = payload["v3"]["instances"]
    except KeyError as exc:
        raise ValueError("expected a protocol-v3 suite artifact") from exc
    if not isinstance(instances, dict) or not instances:
        raise ValueError("suite artifact contains no instances")
    return instances


def _paired_instances(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    try:
        paired = payload["v3"]["paired"]
    except KeyError as exc:
        raise ValueError("expected a protocol-v3 paired artifact") from exc
    if not isinstance(paired, dict):
        raise ValueError("paired artifact has an invalid result map")
    return paired


def _seconds(entry: dict[str, Any], solver: str) -> float | None:
    value = entry["solvers"][solver].get("seconds_median_of_hosts")
    return float(value) if value is not None else None


def summarize(
    suite_payload: dict[str, Any], paired_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return compact, chart-ready metrics from the raw Modal artifacts."""
    instances = _suite_instances(suite_payload)
    paired = _paired_instances(paired_payload)
    rows: list[dict[str, Any]] = []
    coverage: Counter[str] = Counter()
    fastest: Counter[str] = Counter()
    suite_wins: Counter[str] = Counter()
    relative_objective_deltas: dict[str, list[float]] = {"highs": [], "clarabel": []}
    residuals: list[float] = []

    for instance, entry in sorted(instances.items()):
        times = {solver: _seconds(entry, solver) for solver in SOLVERS}
        statuses = {
            solver: str(entry["solvers"][solver].get("status", "unknown")) for solver in SOLVERS
        }
        for solver in SOLVERS:
            if statuses[solver] == "optimal":
                coverage[solver] += 1

        available = {
            solver: seconds
            for solver, seconds in times.items()
            if seconds is not None and statuses[solver] == "optimal"
        }
        if available:
            fastest[min(available, key=available.__getitem__)] += 1

        lx = times["linprogx"]
        ratios: dict[str, float | None] = {"highs": None, "clarabel": None}
        if lx is not None:
            for competitor in ("highs", "clarabel"):
                other = times[competitor]
                if other is not None:
                    ratios[competitor] = other / lx
                    if statuses["linprogx"] == statuses[competitor] == "optimal" and lx < other:
                        suite_wins[competitor] += 1

        paired_entry = paired.get(instance)
        paired_ratio = (
            float(paired_entry["ratio_median_of_hosts"])
            if paired_entry and paired_entry.get("ratio_median_of_hosts") is not None
            else None
        )
        objectives = {solver: entry["solvers"][solver].get("objective") for solver in SOLVERS}
        objective_deltas: dict[str, float | None] = {"highs": None, "clarabel": None}
        if objectives["linprogx"] is not None:
            lx_objective = float(objectives["linprogx"])
            for competitor in ("highs", "clarabel"):
                if objectives[competitor] is not None:
                    competitor_objective = float(objectives[competitor])
                    delta = abs(lx_objective - competitor_objective) / max(
                        1.0, abs(competitor_objective)
                    )
                    objective_deltas[competitor] = delta
                    relative_objective_deltas[competitor].append(delta)
        residual = entry["solvers"]["linprogx"].get("max_residual")
        if residual is not None:
            residuals.append(float(residual))
        rows.append(
            {
                "instance": instance,
                "times": times,
                "statuses": statuses,
                "speedup_vs_linprogx": ratios,
                "route": entry["solvers"]["linprogx"].get("backend"),
                "iterations": entry["solvers"]["linprogx"].get("iterations"),
                "linprogx_residual": residual,
                "relative_objective_delta": objective_deltas,
                "paired_linprogx_over_highs": paired_ratio,
                "paired_verdict": paired_entry.get("verdict") if paired_entry else None,
                "paired_wins": paired_entry.get("lx_wins_total") if paired_entry else None,
                "paired_trials": (
                    sum(paired_entry.get("pairs_by_host", [])) if paired_entry else None
                ),
            }
        )

    paired_wins = sum(row["paired_verdict"] == "lx_faster" for row in rows)
    paired_losses = sum(row["paired_verdict"] == "highs_faster" for row in rows)
    paired_count = paired_wins + paired_losses
    route_counts = Counter(str(row["route"]) for row in rows if row["route"])
    return {
        "protocol": {
            "suite": "median of one solve on each of three independent Modal hosts",
            "paired": (
                "median of three host medians, seven interleaved pairs per host" if paired else None
            ),
        },
        "cases": len(rows),
        "coverage": {solver: coverage[solver] for solver in SOLVERS},
        "suite_wins": {competitor: suite_wins[competitor] for competitor in ("highs", "clarabel")},
        "fastest": {solver: fastest[solver] for solver in SOLVERS},
        "routes": dict(sorted(route_counts.items())),
        "accuracy": {
            "max_linprogx_residual": max(residuals, default=None),
            "max_relative_objective_delta": {
                competitor: max(relative_objective_deltas[competitor], default=None)
                for competitor in ("highs", "clarabel")
            },
        },
        "paired_new_cases": {
            "cases": paired_count,
            "wins": paired_wins,
            "losses": paired_losses,
        },
        "rows": rows,
    }


def _set_common_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelcolor": "#343a40",
            "axes.edgecolor": "#ced4da",
            "xtick.color": "#495057",
            "ytick.color": "#495057",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_speedups(summary: dict[str, Any], path: Path) -> None:
    """Plot competitor runtime divided by linprogx runtime for every case."""
    _set_common_style()
    rows = list(summary["rows"])

    def score(row: dict[str, Any]) -> float:
        values = [v for v in row["speedup_vs_linprogx"].values() if v is not None]
        return math.prod(values) ** (1 / len(values)) if values else 0.0

    rows.sort(key=score)
    fig, ax = plt.subplots(figsize=(12.5, 15.5))
    ax.axvspan(1e-2, 1.0, color="#fff4e6", alpha=0.75, zorder=0)
    ax.axvspan(1.0, 1e3, color="#ebfbee", alpha=0.8, zorder=0)
    ax.axvline(1.0, color="#343a40", linewidth=1.3, zorder=1)

    for y, row in enumerate(rows):
        values = row["speedup_vs_linprogx"]
        present = [value for value in values.values() if value is not None]
        if len(present) == 2:
            ax.plot(present, [y, y], color="#ced4da", linewidth=1.1, zorder=1)
        for solver in ("highs", "clarabel"):
            value = values[solver]
            if value is None:
                continue
            complete = row["statuses"][solver] == "optimal"
            ax.scatter(
                value,
                y,
                s=48,
                color=COLORS[solver] if complete else "white",
                edgecolor=COLORS[solver],
                linewidth=1.4,
                marker="o" if complete else "D",
                zorder=3,
            )

    ax.set_xscale("log")
    ax.set_xlim(0.08, 100)
    ax.set_ylim(-1, len(rows))
    ax.set_yticks(range(len(rows)), [row["instance"].removeprefix("lp_") for row in rows])
    ax.grid(axis="x", which="both", color="#dee2e6", linewidth=0.7, alpha=0.8)
    ax.set_xlabel("Competitor time ÷ linprogx time  (higher is better for linprogx)")
    ax.set_title("linprogx across 39 LPnetlib problems", loc="left", pad=22)
    ax.text(
        0,
        1.012,
        "Three-host median · missing points are timeouts/non-optimal results; diamonds mark partial coverage",
        transform=ax.transAxes,
        color="#6c757d",
        fontsize=10.5,
    )
    ax.text(0.16, len(rows) - 0.2, "competitor faster", color="#c2410c", fontsize=9)
    ax.text(1.18, len(rows) - 0.2, "linprogx faster", color="#087f5b", fontsize=9)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["highs"],
                markeredgecolor=COLORS["highs"],
                label="HiGHS",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["clarabel"],
                markeredgecolor=COLORS["clarabel"],
                label="Clarabel",
            ),
        ],
        loc="lower right",
        frameon=False,
        ncol=2,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, path)


def _scatter_panel(ax: Axes, summary: dict[str, Any], competitor: str) -> None:
    rows = [
        row
        for row in summary["rows"]
        if row["times"]["linprogx"] is not None and row["times"][competitor] is not None
    ]
    all_times = [
        value for row in rows for value in (row["times"]["linprogx"], row["times"][competitor])
    ]
    lower = min(all_times) / 1.7
    upper = max(all_times) * 1.7
    ax.fill_between([lower, upper], [lower, upper], [lower, lower], color="#ebfbee", alpha=0.8)
    ax.plot([lower, upper], [lower, upper], color="#868e96", linestyle="--", linewidth=1)
    for route, color in ROUTE_COLORS.items():
        selected = [row for row in rows if row["route"] == route]
        ax.scatter(
            [row["times"][competitor] for row in selected],
            [row["times"]["linprogx"] for row in selected],
            s=43,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            label=route.upper(),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.grid(which="both", color="#e9ecef", linewidth=0.6)
    ax.set_xlabel(f"{LABELS[competitor]} seconds")
    ax.set_title(f"linprogx vs {LABELS[competitor]}", loc="left")
    ax.text(
        0.04,
        0.96,
        "linprogx faster",
        transform=ax.transAxes,
        color="#087f5b",
        va="top",
        fontsize=9,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_overview(summary: dict[str, Any], path: Path) -> None:
    """Plot coverage, fastest-case counts, and runtime parity charts."""
    _set_common_style()
    fig = plt.figure(figsize=(13.5, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.75, 2.2), hspace=0.36, wspace=0.23)
    coverage_ax = fig.add_subplot(grid[0, 0])
    fastest_ax = fig.add_subplot(grid[0, 1])
    highs_ax = fig.add_subplot(grid[1, 0])
    clarabel_ax = fig.add_subplot(grid[1, 1], sharey=highs_ax)

    solvers = list(SOLVERS)
    coverage = [summary["coverage"][solver] for solver in solvers]
    coverage_ax.barh(
        [LABELS[solver] for solver in solvers],
        coverage,
        color=[COLORS[solver] for solver in solvers],
        height=0.55,
    )
    coverage_ax.set_xlim(0, summary["cases"] + 2)
    coverage_ax.set_title("Optimal solves", loc="left")
    coverage_ax.set_xlabel(f"Cases out of {summary['cases']}")
    for index, value in enumerate(coverage):
        coverage_ax.text(value + 0.4, index, str(value), va="center", fontweight="bold")

    fastest = [summary["fastest"][solver] for solver in solvers]
    fastest_ax.barh(
        [LABELS[solver] for solver in solvers],
        fastest,
        color=[COLORS[solver] for solver in solvers],
        height=0.55,
    )
    fastest_ax.set_xlim(0, max(fastest) + 4)
    fastest_ax.set_title("Fastest solver by case", loc="left")
    fastest_ax.set_xlabel("Cases using three-host median")
    for index, value in enumerate(fastest):
        fastest_ax.text(value + 0.3, index, str(value), va="center", fontweight="bold")

    for ax in (coverage_ax, fastest_ax):
        ax.grid(axis="x", color="#e9ecef", linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

    _scatter_panel(highs_ax, summary, "highs")
    _scatter_panel(clarabel_ax, summary, "clarabel")
    highs_ax.set_ylabel("linprogx seconds")
    clarabel_ax.tick_params(axis="y", labelleft=False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=color,
            label=route.upper(),
        )
        for route, color in ROUTE_COLORS.items()
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.565), ncol=3, frameon=False
    )
    fig.suptitle("Coverage, routes, and scale", x=0.07, ha="left", fontsize=19, fontweight="bold")
    fig.text(
        0.07,
        0.935,
        "39 LPnetlib problems · same locked solver stack · three independent AWS hosts",
        color="#6c757d",
        fontsize=11,
    )
    _save(fig, path)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LPnetlib 39-case benchmark",
        "",
        (
            "Times are medians of one solve on each of three independent Modal hosts. "
            "The paired column is the stricter three-host, seven-pairs-per-host result "
            "for the 15 newly added cases."
        ),
        "",
        "| Instance | linprogx | HiGHS | Clarabel | vs HiGHS | vs Clarabel | Route | Paired lx/HiGHS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary["rows"]:
        times = row["times"]
        ratios = row["speedup_vs_linprogx"]

        def seconds(
            solver: str,
            row_times: dict[str, float | None] = times,
            row_statuses: dict[str, str] = row["statuses"],
        ) -> str:
            value = row_times[solver]
            status = row_statuses[solver]
            if value is None:
                return status
            suffix = "†" if status != "optimal" else ""
            return f"{value:.3f}s{suffix}"

        paired = row["paired_linprogx_over_highs"]
        lines.append(
            "| {instance} | {linprogx} | {highs} | {clarabel} | {highs_ratio} | "
            "{clarabel_ratio} | {route} | {paired} |".format(
                instance=row["instance"],
                linprogx=seconds("linprogx"),
                highs=seconds("highs"),
                clarabel=seconds("clarabel"),
                highs_ratio=(f"{ratios['highs']:.2f}x" if ratios["highs"] is not None else "n/a"),
                clarabel_ratio=(
                    f"{ratios['clarabel']:.2f}x" if ratios["clarabel"] is not None else "n/a"
                ),
                route=row["route"] or "n/a",
                paired=f"{paired:.3f}x" if paired is not None else "—",
            )
        )
    lines.extend(
        [
            "",
            "Ratios in the `vs` columns are competitor time divided by linprogx time, so values "
            "above 1.0 favor linprogx. `†` marks a solver with incomplete three-host coverage; "
            "its displayed time is the median of successful hosts only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="protocol-v3 suite JSON")
    parser.add_argument("--paired", type=Path, help="protocol-v3 paired JSON for new cases")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()

    suite_payload = json.loads(args.suite.read_text())
    paired_payload = json.loads(args.paired.read_text()) if args.paired else None
    summary = summarize(suite_payload, paired_payload)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lpnetlib_39_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "lpnetlib_39_results.md").write_text(render_markdown(summary))
    plot_speedups(summary, output_dir / "lpnetlib_39_speedups.png")
    plot_overview(summary, output_dir / "lpnetlib_39_overview.png")


if __name__ == "__main__":
    main()
