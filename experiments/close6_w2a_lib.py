"""Shared loading / route-replication helpers for the W2-A arm matrix.

Read-only with respect to shipped behaviour: this module never imports or
mutates production defaults, it only *replicates* the route decisions made by
``linprogx.sparse._solve_eq_box`` so that individual simplex kernels can be
driven directly and timed in isolation.

Campaign: close-six wave 2A (2026-07-31), HEAD fc2f86e.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

EPS = 2e-5
FIXTURE_DIR = Path("/tmp/lpsuite")

#: the eleven simplex-routed LPnetlib cases in scope for W2-A
CASES = (
    "lp_25fv47",
    "lp_degen2",
    "lp_greenbeb",
    "lp_sierra",
    "lp_greenbea",
    "lp_agg2",
    "lp_agg3",
    "lp_cycle",
    "lp_fffff800",
    "lp_israel",
    "lp_tuff",
)

#: the four losses under attack and their funding gates (candidate/shipped CPU)
LOSSES = {
    "lp_25fv47": 0.275,
    "lp_degen2": 0.268,
    "lp_greenbeb": 0.705,
    "lp_sierra": 0.354,
}


def load_instance(path: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A_scipy": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def bounds_of(data: dict[str, Any]) -> list[tuple[float | None, float | None]]:
    return [
        (
            None if low == float("-inf") else float(low),
            None if up == float("inf") else float(up),
        )
        for low, up in zip(data["lo"], data["hi"], strict=True)
    ]


def problem_of(data: dict[str, Any]) -> Any:
    from linprogx.sparse import SparseLPProblem, from_scipy_sparse

    return SparseLPProblem(
        c=data["c"].tolist(),
        A_eq=from_scipy_sparse(data["A_scipy"]),
        b_eq=data["b"].tolist(),
        objective="min",
        bounds=bounds_of(data),
    )


def _materialize(reduction: Any) -> Any:
    from linprogx.sparse import csr_matrix

    if reduction._matrix is not None:
        return reduction._matrix
    return csr_matrix(
        reduction.rows,
        reduction.cols,
        reduction.indptr,
        reduction.indices,
        reduction.data,
    )


class Prepared:
    """Route replication of ``_solve_eq_box`` up to the kernel call.

    Exposes both the presolve-only reduction (what the shipped Dantzig DS sees
    when the aggregation gate declines) and, where available, the aggregated
    reduction (what the DS2 composition sees when the gate accepts).
    """

    def __init__(self, data: dict[str, Any]) -> None:
        from linprogx.presolve import (  # noqa: PLC2701
            _maybe_aggregate,
            aggressive_aggregate_for_ds2,
            presolve_matrix,
        )

        self.problem = problem_of(data)
        self.raw_matrix = self.problem.A_eq
        self.c = [float(v) for v in self.problem.c]
        self.b = [float(v) for v in self.problem.b_eq]
        bounds = self.problem.bounds
        self.lo = [float("-inf") if lo is None else float(lo) for lo, _ in bounds]
        self.hi = [float("inf") if hi is None else float(hi) for _, hi in bounds]

        t0 = time.process_time()
        # NOTE: presolve_matrix takes b BEFORE c.
        self.reduction = presolve_matrix(
            self.raw_matrix, self.b, self.c, self.lo, self.hi, algorithm="auto"
        )
        self.presolve_cpu = time.process_time() - t0

        self.base_matrix = _materialize(self.reduction)
        self.base = (
            self.base_matrix,
            self.reduction,
            self.reduction.c,
            self.reduction.b,
            self.reduction.lo,
            self.reduction.hi,
        )

        # The shipped 20%-rows / 5%-nnz exchange gate.
        agg = aggressive_aggregate_for_ds2(self.reduction)
        self.gate_accepts = agg is not None
        self.agg = None
        if agg is not None:
            self.agg = (_materialize(agg), agg, agg.c, agg.b, agg.lo, agg.hi)

        # The SAME aggregation with the global gate bypassed.  Not shippable as
        # such -- recorded only to measure what the gate is declining.
        forced = _maybe_aggregate(
            self.reduction, 5, agg_max_fill=20, fill_budget=-1, fill_gate=False
        )
        self.forced_agg = None
        if forced is not self.reduction:
            self.forced_agg = (
                _materialize(forced),
                forced,
                forced.c,
                forced.b,
                forced.lo,
                forced.hi,
            )

    def shape(self, which: str = "base") -> tuple[int, int, int]:
        bundle = {"base": self.base, "agg": self.agg, "forced_agg": self.forced_agg}[which]
        if bundle is None:
            return (0, 0, 0)
        m = bundle[0]
        return (*m.shape, m.nnz)


def max_residual(matrix: Any, x: list[float], b: list[float]) -> float:
    ax = matrix.matvec(x)
    return max((abs(float(lhs) - rhs) for lhs, rhs in zip(ax, b, strict=True)), default=0.0)


def certify(prep: Prepared, reduction: Any, raw: dict[str, Any]) -> dict[str, Any]:
    """Postsolve a kernel result and check the eps=2e-5 original-units gate."""
    from linprogx.presolve import postsolve_x

    status = str(raw["status"])
    if status != "optimal":
        return {"certified": False, "status": status, "objective": None, "residual": None}
    x = [float(v) for v in raw["x"]]
    if reduction is not None:
        x = postsolve_x(x, reduction)
    objective = sum(v * coef for v, coef in zip(x, prep.problem.c, strict=True))
    residual = max_residual(prep.raw_matrix, x, prep.b)
    return {
        "certified": residual <= EPS,
        "status": status,
        "objective": objective,
        "residual": residual,
    }


class EnvScope:
    """Set/restore environment variables around a single kernel call.

    Every DS entry point re-reads its env flags via ``ds_refresh_*`` at solve
    start, so per-call scoping is sufficient and no subprocess is needed.
    """

    def __init__(self, env: dict[str, str]) -> None:
        self.env = env
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> EnvScope:
        for key, value in self.env.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc: object) -> None:
        for key, old in self.saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
