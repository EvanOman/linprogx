from __future__ import annotations

import argparse
import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linprogx.compare import Comparison, compare_with_clarabel, compare_with_scipy
from linprogx.samples import SAMPLES, STANDARD_BENCHMARKS, SampleProblem


@dataclass(frozen=True)
class BenchRow:
    sample: str
    solver: str
    status: str
    objective_delta: float | None
    linprogx_ms: float
    solver_ms: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benchmark plots for README assets.")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("assets"))
    args = parser.parse_args()

    rows = collect_rows((*SAMPLES, *STANDARD_BENCHMARKS), repeats=args.repeats)
    args.out.mkdir(parents=True, exist_ok=True)
    write_plots(rows, args.out)
    write_markdown_summary(rows, args.out / "perf-summary.md")
    return 0


def collect_rows(samples: tuple[SampleProblem, ...], *, repeats: int) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for sample in samples:
        for comparison in (
            compare_with_scipy(sample.problem, repeats=repeats),
            compare_with_clarabel(sample.problem, repeats=repeats),
        ):
            rows.append(_row(comparison))
    return rows


def write_plots(rows: list[BenchRow], out: Path) -> None:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    pyplot: Any = importlib.import_module("matplotlib.pyplot")

    _plot_runtime_bars(pyplot, rows, out / "perf_runtime_samples.png", standardized=False)
    _plot_runtime_bars(pyplot, rows, out / "perf_runtime_klee_minty.png", standardized=True)
    _plot_speed_ratios(pyplot, rows, out / "perf_speed_ratios.png")
    _plot_objective_delta(pyplot, rows, out / "perf_objective_delta.png")


def write_markdown_summary(rows: list[BenchRow], path: Path) -> None:
    optimal = [row for row in rows if row.status == "optimal"]
    scipy = [row for row in rows if row.solver == "scipy-highs"]
    clarabel = [row for row in rows if row.solver == "clarabel"]
    max_scipy_delta = max((row.objective_delta or 0.0 for row in scipy), default=0.0)
    max_clarabel_delta = max((row.objective_delta or 0.0 for row in clarabel), default=0.0)
    median_scipy_ratio = _median([row.solver_ms / row.linprogx_ms for row in scipy])
    median_clarabel_ratio = _median([row.solver_ms / row.linprogx_ms for row in clarabel])
    fastest = min(optimal, key=lambda row: min(row.linprogx_ms, row.solver_ms))
    slowest = max(optimal, key=lambda row: max(row.linprogx_ms, row.solver_ms))
    path.write_text(
        "\n".join(
            [
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Cases compared | {len(rows) // 2} |",
                f"| SciPy/HiGHS max objective delta | {max_scipy_delta:.2e} |",
                f"| Clarabel max objective delta | {max_clarabel_delta:.2e} |",
                f"| Median SciPy/HiGHS runtime ratio vs linprogx | {median_scipy_ratio:.2f}x |",
                f"| Median Clarabel runtime ratio vs linprogx | {median_clarabel_ratio:.2f}x |",
                f"| Fastest measured row | {fastest.sample} / {fastest.solver} |",
                f"| Slowest measured row | {slowest.sample} / {slowest.solver} |",
                "",
            ]
        )
    )


def _row(comparison: Comparison) -> BenchRow:
    return BenchRow(
        sample=comparison.name,
        solver=comparison.solver_name,
        status=comparison.linprogx_status,
        objective_delta=comparison.objective_delta,
        linprogx_ms=comparison.linprogx_seconds * 1000,
        solver_ms=comparison.solver_seconds * 1000,
    )


def _plot_runtime_bars(
    pyplot: Any, rows: list[BenchRow], path: Path, *, standardized: bool
) -> None:
    selected = [row for row in rows if row.sample.startswith("klee_minty_") == standardized]
    samples = _ordered_samples(selected)
    height = max(5.0, len(samples) * 0.42)
    fig, ax = pyplot.subplots(figsize=(12, height))
    y = list(range(len(samples)))
    linprogx = [
        _mean(row.linprogx_ms for row in selected if row.sample == sample) for sample in samples
    ]
    scipy = [_value(selected, sample, "scipy-highs", "solver_ms") for sample in samples]
    clarabel = [_value(selected, sample, "clarabel", "solver_ms") for sample in samples]

    ax.barh([value + 0.22 for value in y], linprogx, height=0.2, label="linprogx", color="#28536b")
    ax.barh(y, scipy, height=0.2, label="SciPy/HiGHS", color="#c44536")
    ax.barh([value - 0.22 for value in y], clarabel, height=0.2, label="Clarabel", color="#f3a712")
    ax.set_yticks(y, samples)
    ax.set_xlabel("milliseconds per solve")
    ax.set_title(
        "Runtime on standardized Klee-Minty LPs" if standardized else "Runtime on sample LPs"
    )
    ax.grid(axis="x", color="#d8dee4", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    pyplot.close(fig)


def _plot_speed_ratios(pyplot: Any, rows: list[BenchRow], path: Path) -> None:
    samples = _ordered_samples(rows)
    fig, ax = pyplot.subplots(figsize=(12, 7))
    x = list(range(len(samples)))
    scipy = [
        _value(rows, sample, "scipy-highs", "solver_ms") / _linprogx(rows, sample)
        for sample in samples
    ]
    clarabel = [
        _value(rows, sample, "clarabel", "solver_ms") / _linprogx(rows, sample)
        for sample in samples
    ]
    ax.plot(x, scipy, marker="o", linewidth=2, label="SciPy/HiGHS / linprogx", color="#c44536")
    ax.plot(x, clarabel, marker="o", linewidth=2, label="Clarabel / linprogx", color="#f3a712")
    ax.axhline(1.0, color="#28536b", linestyle="--", linewidth=1.2, label="parity")
    ax.set_xticks(x, samples, rotation=60, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("runtime ratio, log scale")
    ax.set_title("Runtime Ratio Against linprogx")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    pyplot.close(fig)


def _plot_objective_delta(pyplot: Any, rows: list[BenchRow], path: Path) -> None:
    optimal = [row for row in rows if row.objective_delta is not None]
    samples = _ordered_samples(optimal)
    fig, ax = pyplot.subplots(figsize=(12, 7))
    x = list(range(len(samples)))
    scipy = [
        max(_value(optimal, sample, "scipy-highs", "objective_delta"), 1e-16) for sample in samples
    ]
    clarabel = [
        max(_value(optimal, sample, "clarabel", "objective_delta"), 1e-16) for sample in samples
    ]
    ax.scatter(x, scipy, s=55, label="SciPy/HiGHS", color="#c44536")
    ax.scatter(x, clarabel, s=55, label="Clarabel", color="#f3a712")
    ax.set_xticks(x, samples, rotation=60, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("absolute objective delta vs linprogx")
    ax.set_title("Correctness Check: Objective Deltas")
    ax.grid(axis="y", color="#d8dee4", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    pyplot.close(fig)


def _ordered_samples(rows: list[BenchRow]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row.sample not in seen:
            seen.append(row.sample)
    return seen


def _linprogx(rows: list[BenchRow], sample: str) -> float:
    return _mean(row.linprogx_ms for row in rows if row.sample == sample)


def _value(rows: list[BenchRow], sample: str, solver: str, field: str) -> float:
    for row in rows:
        if row.sample == sample and row.solver == solver:
            return float(getattr(row, field) or 0.0)
    msg = f"missing {solver} row for {sample}"
    raise KeyError(msg)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


if __name__ == "__main__":
    raise SystemExit(main())
