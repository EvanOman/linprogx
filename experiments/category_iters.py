"""Is greenbea a category? Compare ITERATION COUNTS, which are load-invariant.

Wall-clock on a shared box cannot answer this: a single-shot local run showed
greenbea at 927 ms when a proper median-of-9 measures 377 ms.  Iteration counts
have no such problem -- they are deterministic properties of the algorithm on
the instance.

The campaign's core deficit on greenbea is a TRAJECTORY deficit: linprogx needs
4,399 pivots to HiGHS's 2,836, a ratio of 1.55.  This asks whether that ratio is
an outlier or a property of a structural class, by measuring it across a broader
LPnetlib sample together with each instance's structural signature.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import loadmat

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
SKIP = {"lp_qap15"}


def load(path: Path) -> dict[str, Any]:
    raw = loadmat(path)["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(float),
        "c": aux["c"].ravel().astype(float),
        "lo": aux["lo"].ravel().astype(float),
        "hi": aux["hi"].ravel().astype(float),
    }


def sig(d):
    lo, hi = d["lo"], d["hi"]
    m, n = d["A"].shape
    li, hj = lo == -INF, hi == INF
    return {
        "rows": int(m),
        "cols": int(n),
        "nnz": int(d["A"].nnz),
        "one_sided_frac": float(((li ^ hj) | (li & hj)).sum()) / max(1, n),
        "avg_col_nnz": float(np.diff(d["A"].indptr).mean()),
    }


def lx_iters(d):
    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    bounds = [
        (None if l == -INF else float(l), None if h == INF else float(h))
        for l, h in zip(d["lo"], d["hi"], strict=True)
    ]
    t0 = time.perf_counter()
    r = SparseSolver(
        algorithm="auto", max_iterations=200_000, eps=2e-5, check_interval=50_000
    ).solve(
        SparseLPProblem(
            c=d["c"].tolist(),
            A_eq=from_scipy_sparse(d["A"]),
            b_eq=d["b"].tolist(),
            objective="min",
            bounds=bounds,
        )
    )
    return {
        "iters": int(r.solution.iterations),
        "status": r.solution.status.value,
        "obj": r.solution.objective_value,
        "route": r.backend.rsplit("-", 1)[-1],
        "ms": (time.perf_counter() - t0) * 1e3,
    }


def hx_iters(d):
    import highspy

    m, n = d["A"].shape
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = d["c"]
    lp.col_lower_ = np.where(np.isfinite(d["lo"]), d["lo"], -inf)
    lp.col_upper_ = np.where(np.isfinite(d["hi"]), d["hi"], inf)
    lp.row_lower_ = d["b"]
    lp.row_upper_ = d["b"]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = d["A"].indptr.astype(np.int32)
    lp.a_matrix_.index_ = d["A"].indices.astype(np.int32)
    lp.a_matrix_.value_ = d["A"].data.astype(float)
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    ms = (time.perf_counter() - t0) * 1e3
    info = h.getInfo()
    return {
        "iters": int(info.simplex_iteration_count),
        "status": h.modelStatusToString(h.getModelStatus()),
        "obj": h.getObjectiveValue(),
        "ms": ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="")
    ap.add_argument("--out", default="/tmp/category_iters.jsonl")
    a = ap.parse_args()
    names = (
        [f"lp_{x}" if not x.startswith("lp_") else x for x in a.instances.split(",")]
        if a.instances
        else sorted(p.stem for p in SUITE.glob("lp_*.mat"))
    )
    names = [n for n in names if n not in SKIP]
    print(
        f"{'instance':15s} {'rows':>5} {'cols':>5} {'1sided':>7} {'cnnz':>5} "
        f"{'route':>8} {'lx it':>7} {'hx it':>7} {'it ratio':>8}  agree"
    )
    out = []
    for nm in names:
        p = SUITE / f"{nm}.mat"
        if not p.exists():
            continue
        try:
            d = load(p)
            s = sig(d)
        except Exception as e:
            print(f"{nm:15s} SKIP (unreadable fixture: {type(e).__name__})")
            continue
        try:
            L = lx_iters(d)
        except Exception as e:
            L = {
                "iters": -1,
                "status": f"ERR:{type(e).__name__}",
                "obj": None,
                "route": "-",
                "ms": float("nan"),
            }
        try:
            H = hx_iters(d)
        except Exception as e:
            H = {"iters": -1, "status": f"ERR:{type(e).__name__}", "obj": None, "ms": float("nan")}
        ratio = (L["iters"] / H["iters"]) if (L["iters"] > 0 and H["iters"] > 0) else float("nan")
        ag = "?"
        if L["obj"] is not None and H["obj"] is not None:
            rel = abs(L["obj"] - H["obj"]) / max(1.0, abs(H["obj"]))
            ag = "OK" if rel <= 1e-6 else f"D{rel:.0e}"
        rec = {"instance": nm, **s, "lx": L, "hx": H, "iter_ratio": ratio}
        out.append(rec)
        print(
            f"{nm:15s} {s['rows']:>5} {s['cols']:>5} {100 * s['one_sided_frac']:>6.1f}% "
            f"{s['avg_col_nnz']:>5.2f} {L['route'][:8]:>8} {L['iters']:>7} {H['iters']:>7} "
            f"{ratio:>8.2f}  {ag}",
            flush=True,
        )
    Path(a.out).write_text("\n".join(json.dumps(r) for r in out))
    ok = [r for r in out if r["iter_ratio"] == r["iter_ratio"]]
    if ok:
        import statistics as st

        print(
            f"\nmedian iteration ratio lx/hx: {st.median(r['iter_ratio'] for r in ok):.2f}"
            f"   (greenbea = 1.55)"
        )
        worst = sorted(ok, key=lambda r: -r["iter_ratio"])[:8]
        print("\nWORST trajectory ratios:")
        for r in worst:
            print(
                f"  {r['instance']:15s} {r['iter_ratio']:5.2f}x  one-sided "
                f"{100 * r['one_sided_frac']:5.1f}%  cnnz {r['avg_col_nnz']:5.2f}  route {r['lx']['route']}"
            )
    print(f"\nartifact: {a.out}")


if __name__ == "__main__":
    main()
