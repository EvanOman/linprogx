from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_report_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "tools" / "build_lpnetlib_report.py"
    spec = importlib.util.spec_from_file_location("build_lpnetlib_report_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _solver(seconds: float | None, *, status: str = "optimal", backend: str | None = None) -> Any:
    return {
        "status": status,
        "seconds_median_of_hosts": seconds,
        "objective": 10.0,
        "backend": backend,
        "iterations": 12,
        "max_residual": 1e-10,
    }


def _suite() -> dict[str, Any]:
    return {
        "v3": {
            "instances": {
                "lp_alpha": {
                    "solvers": {
                        "linprogx": _solver(1.0, backend="ipm"),
                        "highs": _solver(2.0),
                        "clarabel": _solver(4.0),
                    }
                },
                "lp_beta": {
                    "solvers": {
                        "linprogx": _solver(3.0, backend="simplex"),
                        "highs": _solver(1.0),
                        "clarabel": _solver(2.0, status="incomplete"),
                    }
                },
            }
        }
    }


def _paired() -> dict[str, Any]:
    return {
        "v3": {
            "paired": {
                "lp_alpha": {
                    "ratio_median_of_hosts": 0.5,
                    "verdict": "lx_faster",
                    "lx_wins_total": 20,
                    "pairs_by_host": [7, 7, 7],
                },
                "lp_beta": {
                    "ratio_median_of_hosts": 3.0,
                    "verdict": "highs_faster",
                    "lx_wins_total": 0,
                    "pairs_by_host": [7, 7, 7],
                },
            }
        }
    }


def test_summarize_counts_coverage_wins_routes_and_paired_verdicts() -> None:
    report = _load_report_module()

    summary = report.summarize(_suite(), _paired())

    assert summary["cases"] == 2
    assert summary["coverage"] == {"linprogx": 2, "highs": 2, "clarabel": 1}
    assert summary["suite_wins"] == {"highs": 1, "clarabel": 1}
    assert summary["fastest"] == {"linprogx": 1, "highs": 1, "clarabel": 0}
    assert summary["routes"] == {"ipm": 1, "simplex": 1}
    assert summary["accuracy"] == {
        "max_linprogx_residual": pytest.approx(1e-10),
        "max_relative_objective_delta": {
            "highs": pytest.approx(0.0),
            "clarabel": pytest.approx(0.0),
        },
    }
    assert summary["paired_new_cases"] == {"cases": 2, "wins": 1, "losses": 1}
    assert summary["rows"][0]["speedup_vs_linprogx"] == {
        "highs": pytest.approx(2.0),
        "clarabel": pytest.approx(4.0),
    }
    assert summary["rows"][0]["paired_trials"] == 21


def test_render_markdown_explains_ratios_and_incomplete_coverage() -> None:
    report = _load_report_module()
    summary = report.summarize(_suite(), _paired())

    markdown = report.render_markdown(summary)

    assert "| lp_alpha | 1.000s | 2.000s | 4.000s | 2.00x | 4.00x | ipm | 0.500x |" in markdown
    assert "2.000s†" in markdown
    assert "competitor time divided by linprogx time" in markdown


def test_summarize_rejects_non_v3_payloads() -> None:
    report = _load_report_module()

    with pytest.raises(ValueError, match="protocol-v3 suite"):
        report.summarize({})


def test_charts_render_from_summary(tmp_path: Path) -> None:
    report = _load_report_module()
    summary = report.summarize(_suite(), _paired())
    speedups = tmp_path / "speedups.png"
    overview = tmp_path / "overview.png"

    report.plot_speedups(summary, speedups)
    report.plot_overview(summary, overview)

    assert speedups.stat().st_size > 10_000
    assert overview.stat().st_size > 10_000
