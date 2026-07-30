from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest


class _FakeImage:
    @classmethod
    def debian_slim(cls, python_version: str) -> _FakeImage:
        return cls()

    def apt_install(self, *packages: str) -> _FakeImage:
        return self

    def pip_install(self, *packages: str) -> _FakeImage:
        return self


class _FakeVolume:
    @classmethod
    def from_name(cls, name: str, create_if_missing: bool) -> _FakeVolume:
        return cls()


class _FakeApp:
    def __init__(self, name: str) -> None:
        self.name = name

    def function(self, **kwargs: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def local_entrypoint(self) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator


def _load_modal_bench(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake_modal = types.SimpleNamespace(App=_FakeApp, Image=_FakeImage, Volume=_FakeVolume)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    path = Path(__file__).resolve().parents[1] / "tools" / "modal_bench.py"
    spec = importlib.util.spec_from_file_location("modal_bench_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _host(host_index: int, ratio: float, wins: int) -> dict[str, Any]:
    verdict = "lx_faster" if ratio < 1.0 else "highs_faster"
    return {
        "host_index": host_index,
        "machine_info": {"cpu_model": f"host-{host_index}"},
        "load_checks": {"loadavg_at_start": "0.00 0.00 0.00 1/1 1"},
        "paired": {
            "lp_osa_14": {
                "pairs": 7,
                "lx": {"median": ratio, "min": ratio * 0.9, "n": 7, "status": "optimal"},
                "hx": {"median": 1.0, "min": 1.0, "n": 7, "status": "optimal"},
                "lx_wins": wins,
                "ratio_median": ratio,
                "ratio_min": ratio * 0.9,
                "verdict": verdict,
            }
        },
    }


def test_protocol_v3_odd_hosts_preserves_host_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)

    aggregate = modal_bench.aggregate_protocol_v3_hosts(
        [_host(0, 0.8, 7), _host(1, 1.3, 1), _host(2, 0.9, 5)]
    )

    entry = aggregate["paired"]["lp_osa_14"]
    assert aggregate["hosts"] == 3
    assert entry["ratio_median_of_hosts"] == pytest.approx(0.9)
    assert entry["verdict"] == "lx_faster"
    assert entry["ratio_min_host"] == pytest.approx(0.8)
    assert entry["ratio_max_host"] == pytest.approx(1.3)
    assert entry["lx_wins_by_host"] == [7, 1, 5]
    assert [host["ratio_median"] for host in entry["per_host"]] == [0.8, 1.3, 0.9]


def test_protocol_v3_even_hosts_uses_statistics_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)

    aggregate = modal_bench.aggregate_protocol_v3_hosts(
        [_host(0, 0.8, 7), _host(1, 1.2, 2), _host(2, 1.4, 1), _host(3, 1.6, 0)]
    )

    entry = aggregate["paired"]["lp_osa_14"]
    assert aggregate["hosts"] == 4
    assert entry["ratio_median_of_hosts"] == pytest.approx(1.3)
    assert entry["verdict"] == "highs_faster"
    assert entry["hosts_observed"] == 4
    assert entry["hosts_with_ratio"] == 4
    assert entry["lx_wins_total"] == 10


def _suite_host(
    host_index: int,
    *,
    linprogx: float,
    highs: float,
    clarabel: float,
) -> dict[str, Any]:
    return {
        "host_index": host_index,
        "rows": [
            {
                "instance": "lp_agg2",
                "solver": "linprogx",
                "status": "optimal",
                "seconds": linprogx,
                "objective": -10.0,
                "residual": 1e-10 * (host_index + 1),
                "backend": "simplex",
                "iterations": 274,
            },
            {
                "instance": "lp_agg2",
                "solver": "highs",
                "status": "optimal",
                "seconds": highs,
                "objective": -10.0,
            },
            {
                "instance": "lp_agg2",
                "solver": "clarabel",
                "status": "optimal",
                "seconds": clarabel,
                "objective": -10.0,
                "residual": 2e-10,
            },
        ],
    }


def test_suite_v3_uses_host_medians_for_all_three_solvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)

    aggregate = modal_bench.aggregate_suite_v3_hosts(
        [
            _suite_host(0, linprogx=1.0, highs=2.0, clarabel=4.0),
            _suite_host(1, linprogx=3.0, highs=2.5, clarabel=5.0),
            _suite_host(2, linprogx=2.0, highs=4.0, clarabel=8.0),
        ]
    )

    entry = aggregate["instances"]["lp_agg2"]
    assert aggregate["protocol"] == "suite-v3"
    assert aggregate["hosts"] == 3
    assert entry["solvers"]["linprogx"]["seconds_median_of_hosts"] == pytest.approx(2.0)
    assert entry["solvers"]["linprogx"]["max_residual"] == pytest.approx(3e-10)
    assert entry["linprogx_over_highs"] == pytest.approx(0.8)
    assert entry["linprogx_over_clarabel"] == pytest.approx(0.4)


def test_suite_v3_marks_a_solver_incomplete_when_a_host_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)
    hosts = [
        _suite_host(0, linprogx=1.0, highs=2.0, clarabel=4.0),
        _suite_host(1, linprogx=1.5, highs=2.5, clarabel=5.0),
    ]
    hosts[1]["rows"][2] = {
        "instance": "lp_agg2",
        "solver": "clarabel",
        "status": "timeout",
        "seconds": 200.0,
    }

    aggregate = modal_bench.aggregate_suite_v3_hosts(hosts)

    clarabel = aggregate["instances"]["lp_agg2"]["solvers"]["clarabel"]
    assert clarabel["status"] == "incomplete"
    assert clarabel["hosts_observed"] == 2
    assert clarabel["hosts_optimal"] == 1
    assert clarabel["seconds_median_of_hosts"] == pytest.approx(4.0)
