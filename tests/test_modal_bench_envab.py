from __future__ import annotations

import importlib.util
import json
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
    spec = importlib.util.spec_from_file_location("modal_bench_envab_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    modal_bench = _load_modal_bench(monkeypatch)

    assert modal_bench._parse_env_overrides("") == {}
    assert modal_bench._parse_env_overrides("K=V, K2=V=2,EMPTY=") == {
        "K": "V",
        "K2": "V=2",
        "EMPTY": "",
    }


@pytest.mark.parametrize("spec", ["MISSING_EQUALS", "=value", "K=V,", "K=1,K=2"])
def test_parse_env_overrides_rejects_invalid_specs(
    monkeypatch: pytest.MonkeyPatch, spec: str
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)

    with pytest.raises(ValueError):
        modal_bench._parse_env_overrides(spec)


def test_run_cell_applies_overrides_only_to_linprogx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)
    captured_envs: list[dict[str, str]] = []

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        captured_envs.append(kwargs["env"])
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"status": "optimal", "seconds": 1.0}),
            stderr="",
        )

    monkeypatch.setattr(modal_bench.subprocess, "run", fake_run)
    override = {"LINPROGX_ENVAB_TEST": "arm-value"}

    modal_bench._run_cell(tmp_path, tmp_path / "fixture.mat", "linprogx", override)
    modal_bench._run_cell(tmp_path, tmp_path / "fixture.mat", "highs", override)

    assert captured_envs[0]["LINPROGX_ENVAB_TEST"] == "arm-value"
    assert "LINPROGX_ENVAB_TEST" not in captured_envs[1]
    assert "LINPROGX_ENVAB_TEST" not in modal_bench.os.environ


def test_envab_interleaves_arms_and_reports_b_over_a(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "lp_greenbea.mat").touch()
    monkeypatch.setattr(modal_bench, "FIXTURES_DIR", str(fixture_dir))
    monkeypatch.setattr(modal_bench, "_prepare_source", lambda git_ref, use_snapshot: tmp_path)
    monkeypatch.setattr(
        modal_bench,
        "_machine_info",
        lambda: {"loadavg": "0.00 0.00 0.00 1/1 1", "cpu_model": "test"},
    )
    calls: list[dict[str, str]] = []
    arm_seconds = {
        "A": iter([10.0, 12.0, 11.0]),
        "B": iter([9.0, 13.0, 8.0]),
    }

    def fake_run_cell(
        workdir: Path,
        fixture: Path,
        solver: str,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        assert solver == "linprogx"
        assert env_overrides is not None
        calls.append(env_overrides)
        arm = env_overrides["ARM"]
        return {
            "solver": solver,
            "status": "optimal",
            "seconds": next(arm_seconds[arm]),
            "backend": "sparse-dual-simplex",
        }

    monkeypatch.setattr(modal_bench, "_run_cell", fake_run_cell)

    result = modal_bench.bench(
        "ref",
        instances=["lp_greenbea"],
        pairs=3,
        mode="envab",
        include_raw_pairs=True,
        env_a={"ARM": "A"},
        env_b={"ARM": "B"},
    )

    entry = result["envab"]["lp_greenbea"]
    assert calls == [
        {"ARM": "A"},
        {"ARM": "B"},
        {"ARM": "A"},
        {"ARM": "B"},
        {"ARM": "A"},
        {"ARM": "B"},
    ]
    assert entry["lxA"]["median"] == pytest.approx(11.0)
    assert entry["lxB"]["median"] == pytest.approx(9.0)
    assert entry["ratio_median"] == pytest.approx(9.0 / 11.0)
    assert entry["lxB_wins"] == 2
    assert [pair["lxB_won"] for pair in entry["pair_results"]] == [True, False, True]


def _envab_host(host_index: int, a_median: float, b_median: float, wins: int) -> dict[str, Any]:
    ratio = b_median / a_median
    verdict = "lxB_faster" if ratio < 1.0 else "lxA_faster"
    return {
        "host_index": host_index,
        "machine_info": {"cpu_model": f"host-{host_index}"},
        "load_checks": {"loadavg_at_start": "0.00 0.00 0.00 1/1 1"},
        "envab": {
            "lp_greenbea": {
                "pairs": 7,
                "lxA": {"median": a_median, "min": a_median * 0.9, "n": 7},
                "lxB": {"median": b_median, "min": b_median * 0.9, "n": 7},
                "lxB_wins": wins,
                "ratio_median": ratio,
                "ratio_min": ratio,
                "verdict": verdict,
            }
        },
    }


def test_envab_v3_aggregates_host_medians_and_preserves_per_host_walls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modal_bench = _load_modal_bench(monkeypatch)
    hosts = [
        _envab_host(0, 10.0, 8.0, 6),
        _envab_host(1, 10.0, 13.0, 2),
        _envab_host(2, 20.0, 18.0, 5),
    ]

    aggregate = modal_bench.aggregate_envab_v3_hosts(hosts)

    entry = aggregate["envab"]["lp_greenbea"]
    assert aggregate["hosts"] == 3
    assert entry["ratio_median_of_hosts"] == pytest.approx(0.9)
    assert entry["ratio_min_host"] == pytest.approx(0.8)
    assert entry["ratio_max_host"] == pytest.approx(1.3)
    assert entry["lxB_wins_by_host"] == [6, 2, 5]
    assert entry["lxB_wins_total"] == 13
    assert [host["lxA"]["median"] for host in entry["per_host"]] == [10.0, 10.0, 20.0]
    assert [host["lxB"]["median"] for host in entry["per_host"]] == [8.0, 13.0, 18.0]
    assert entry["verdict"] == "lxB_faster"
