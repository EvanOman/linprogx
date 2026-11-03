"""Is greenbea an outlier, or does it represent a CATEGORY we lose?

Runs linprogx (public auto route) against HiGHS and Clarabel on a broader
LPnetlib sample, and reports each instance's STRUCTURAL SIGNATURE alongside the
ratio -- so a loss can be correlated with structure rather than just recorded.

greenbea's signature, from the campaign's own clean-room measurements:
  * 93.4% of presolved columns are one-sided or free (3,611 / 3,868)
  * average column nnz in [5, 8) -- network-ish sparsity
  * the only one of the original 24 cells whose public route is the DUAL SIMPLEX
    (every other cell resolves to IPM or PDHG)

If the cells we lose share that signature, greenbea is a category. If they do
not, it is idiosyncratic.

Wall times here are LOCAL and single-shot. They are a screen, not a board
measurement: the campaign's board is protocol v3 on Modal, and this session
established that the linprogx/HiGHS ratio is strongly host-dependent (HiGHS's own
greenbea wall varied 54% across three same-class Modal hosts; the ratio varied
1.16-1.47). Local ratios identify CANDIDATES; only Modal decides cells.

Usage:
    PYTHONPATH=. uv run python experiments/category_bench.py [--instances a,b,c]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
EPS = 2e-5
# HiGHS times out at 300s on this cell; the campaign forbids it in paired mode.
SKIP = {"lp_qap15"}


def load_instance(path: Path) -> dict[str, Any]:
    import numpy as np
    from scipy.io import loadmat

    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def signature(d: dict[str, Any]) -> dict[str, Any]:
    """Structural features that distinguish greenbea's family."""
    lo, hi = d["lo"], d["hi"]
    m, n = d["A"].shape
    lo_inf = lo == -INF
    hi_inf = hi == INF
    free = lo_inf & hi_inf
    one_sided = lo_inf ^ hi_inf
    import numpy as np

    A = d["A"]
    col_nnz = np.diff(A.indptr)  # csc: nonzeros per column
    return {
        "rows": int(m),
        "cols": int(n),
        "nnz": int(d["A"].nnz),
        "one_sided_frac": float((one_sided | free).sum()) / max(1, n),
        "avg_col_nnz": float(col_nnz.mean()) if len(col_nnz) else 0.0,
    }


def solve_linprogx(d: dict[str, Any]) -> dict[str, Any]:
    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    bounds = [
        (None if lo == -INF else float(lo), None if hi == INF else float(hi))
        for lo, hi in zip(d["lo"], d["hi"], strict=True)
    ]
    problem = SparseLPProblem(
        c=d["c"].tolist(),
        A_eq=from_scipy_sparse(d["A"]),
        b_eq=d["b"].tolist(),
        objective="min",
        bounds=bounds,
    )
    t0 = time.perf_counter()
    res = SparseSolver(
        algorithm="auto", max_iterations=50_000, eps=EPS, check_interval=50_000
    ).solve(problem)
    wall = time.perf_counter() - t0
    return {
        "ms": wall * 1e3,
        "status": res.solution.status.value,
        "obj": res.solution.objective_value,
        "iters": res.solution.iterations,
        "backend": res.backend.rsplit("-", 1)[-1],
    }


def solve_highs(d: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import linprog

    t0 = time.perf_counter()
    r = linprog(
        d["c"],
        A_eq=d["A"],
        b_eq=d["b"],
        bounds=list(
            zip(
                np.where(np.isfinite(d["lo"]), d["lo"], -np.inf),
                np.where(np.isfinite(d["hi"]), d["hi"], np.inf),
                strict=True,
            )
        ),
        method="highs",
    )
    wall = time.perf_counter() - t0
    return {
        "ms": wall * 1e3,
        "status": "optimal" if r.success else r.message[:18],
        "obj": float(r.fun) if r.success else None,
    }


def solve_clarabel(d: dict[str, Any]) -> dict[str, Any]:
    import clarabel
    import numpy as np
    import scipy.sparse as sp

    clarabel_api = vars(clarabel)
    m, n = d["A"].shape
    lo, hi = d["lo"], d["hi"]
    lo_f = np.isfinite(lo)
    hi_f = np.isfinite(hi)
    blocks = [d["A"]]
    bvec = [d["b"]]
    cones: list[Any] = [clarabel_api["ZeroConeT"](m)]
    n_ineq = int(lo_f.sum() + hi_f.sum())
    if n_ineq:
        rows = []
        rhs = []
        for j in np.where(hi_f)[0]:
            e = sp.csc_matrix(([1.0], ([0], [j])), shape=(1, n))
            rows.append(e)
            rhs.append(hi[j])
        for j in np.where(lo_f)[0]:
            e = sp.csc_matrix(([-1.0], ([0], [j])), shape=(1, n))
            rows.append(e)
            rhs.append(-lo[j])
        blocks.append(sp.vstack(rows, format="csc"))
        bvec.append(np.array(rhs, dtype=float))
        cones.append(clarabel_api["NonnegativeConeT"](n_ineq))
    A = sp.vstack(blocks, format="csc")
    bb = np.concatenate(bvec)
    P = sp.csc_matrix((n, n))
    settings = clarabel_api["DefaultSettings"]()
    settings.verbose = False
    t0 = time.perf_counter()
    sol = clarabel_api["DefaultSolver"](P, d["c"], A, bb, cones, settings).solve()
    wall = time.perf_counter() - t0
    st = str(sol.status)
    return {
        "ms": wall * 1e3,
        "status": "optimal" if "Solved" in st else st[:18],
        "obj": float(sol.obj_val) if "Solved" in st else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="")
    ap.add_argument("--out", default="/tmp/category_bench.jsonl")
    args = ap.parse_args()

    names = (
        [f"lp_{x}" if not x.startswith("lp_") else x for x in args.instances.split(",")]
        if args.instances
        else sorted(p.stem for p in SUITE.glob("lp_*.mat"))
    )
    names = [n for n in names if n not in SKIP]

    print(
        f"{'instance':16s} {'rows':>6} {'cols':>6} {'1sided':>7} {'colnnz':>6} "
        f"{'route':>13} {'lx ms':>9} {'hx ms':>9} {'cl ms':>9} {'lx/hx':>7}  agree"
    )
    rows_out = []
    for name in names:
        p = SUITE / f"{name}.mat"
        if not p.exists():
            continue
        try:
            d = load_instance(p)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:16s} LOAD-ERROR {type(exc).__name__}")
            continue
        sig = signature(d)
        rec: dict[str, Any] = {"instance": name, **sig}
        for label, fn in (("lx", solve_linprogx), ("hx", solve_highs), ("cl", solve_clarabel)):
            try:
                rec[label] = fn(d)
            except Exception as exc:  # noqa: BLE001
                rec[label] = {
                    "ms": float("nan"),
                    "status": f"ERR:{type(exc).__name__}",
                    "obj": None,
                }
        lx, hx, cl = rec["lx"], rec["hx"], rec["cl"]
        ratio = lx["ms"] / hx["ms"] if hx["ms"] and hx["ms"] == hx["ms"] else float("nan")
        rec["ratio_lx_hx"] = ratio
        # objective agreement against HiGHS
        agree = "?"
        if lx.get("obj") is not None and hx.get("obj") is not None:
            rel = abs(lx["obj"] - hx["obj"]) / max(1.0, abs(hx["obj"]))
            agree = "OK" if rel <= 1e-6 else f"DIFF {rel:.1e}"
            rec["obj_reldiff_vs_highs"] = rel
        rows_out.append(rec)
        print(
            f"{name:16s} {sig['rows']:>6} {sig['cols']:>6} "
            f"{100 * sig['one_sided_frac']:>6.1f}% {sig['avg_col_nnz']:>6.2f} "
            f"{lx['backend'][:13]:>13} {lx['ms']:>9.1f} {hx['ms']:>9.1f} "
            f"{cl['ms']:>9.1f} {ratio:>7.3f}  {agree}",
            flush=True,
        )

    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows_out))
    print(f"\nartifact: {args.out}")

    # --- the actual question -------------------------------------------------
    ok = [r for r in rows_out if r.get("ratio_lx_hx") == r.get("ratio_lx_hx")]
    losses = [r for r in ok if r["ratio_lx_hx"] > 1.0]
    wins = [r for r in ok if r["ratio_lx_hx"] <= 1.0]
    print(f"\nlocal wins {len(wins)} / losses {len(losses)} of {len(ok)} measured")
    if losses:
        print("\nLOSSES (local wall), with structure:")
        for r in sorted(losses, key=lambda r: -r["ratio_lx_hx"]):
            print(
                f"  {r['instance']:16s} ratio {r['ratio_lx_hx']:6.3f}  "
                f"one-sided {100 * r['one_sided_frac']:5.1f}%  "
                f"col-nnz {r['avg_col_nnz']:5.2f}  route {r['lx']['backend']}"
            )
    if wins and losses:
        import statistics as st

        print(
            f"\nmean one-sided frac: losses {st.mean(r['one_sided_frac'] for r in losses):.3f}"
            f" vs wins {st.mean(r['one_sided_frac'] for r in wins):.3f}"
        )
        print(
            f"mean avg-col-nnz  : losses {st.mean(r['avg_col_nnz'] for r in losses):.2f}"
            f" vs wins {st.mean(r['avg_col_nnz'] for r in wins):.2f}"
        )
        dsl = sum(1 for r in losses if "simplex" in r["lx"]["backend"])
        dsw = sum(1 for r in wins if "simplex" in r["lx"]["backend"])
        print(f"DS-routed         : losses {dsl}/{len(losses)} vs wins {dsw}/{len(wins)}")


if __name__ == "__main__":
    main()
