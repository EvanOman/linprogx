"""Selection-quality of the CHUZR rule, per instance, in pivots.

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

The DS2 CHUZR brief says exact DSE was measured to be WORSE than Dantzig --
4,675 pivots against 4,399.  That measurement was taken ON GREENBEA.  The
fresh evidence that motivated the rewrite lists five losses, and greenbea is
only the fourth-worst of them: 25fv47 2.74x, degen2 2.69x, greenbeb 1.82x,
greenbea 1.55x, tuff 1.27x.  Nothing in the ledger records what the leaving
rule does on the other four.

This sweeps the shipped solver's own `leaving_rule` knob across the
simplex-routed instances.  Nothing is modified: rule 1 (Dantzig) is what the
auto route ships, rule 0 is Devex, rule 5 is exact Forrest-Goldfarb dual
steepest edge -- the same merit the DS2 component implements.  Pivot counts
are load-invariant, so this is a clean comparison on a loaded box.

Presolve is applied first, so the trajectory is over the same reduced
problem the shipped route solves.

Usage:
  PYTHONPATH=src uv run python experiments/ds2_chuzr_rule_sweep.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

SUITE = Path("/tmp/lpsuite")

# the simplex-routed cells from the fresh census, with their measured ratio
SIMPLEX_ROUTED = {
    "lp_25fv47": 2.74,
    "lp_degen2": 2.69,
    "lp_greenbeb": 1.82,
    "lp_greenbea": 1.55,
    "lp_tuff": 1.27,
    "lp_israel": 0.97,
    "lp_fffff800": 0.81,
    "lp_agg2": 0.51,
    "lp_agg3": 0.48,
}

RULES = {0: "devex", 1: "dantzig", 5: "dse"}


def load(name: str):
    raw = loadmat(SUITE / f"{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return (
        sp.csr_matrix(raw["A"]).astype(float),
        raw["b"].ravel().astype(float),
        aux["c"].ravel().astype(float),
        aux["lo"].ravel().astype(float),
        aux["hi"].ravel().astype(float),
    )


def reduced(A, b, c, lo, hi):
    """linprogx's own presolve, so the DS sees the shipped route's problem."""
    from linprogx.presolve import presolve_matrix
    from linprogx.sparse import from_scipy_sparse

    matrix = from_scipy_sparse(A)
    red = presolve_matrix(matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist())
    if red is None:
        return matrix, b.tolist(), c.tolist(), lo.tolist(), hi.tolist()
    return red._matrix, list(red.b), list(red.c), list(red.lo), list(red.hi)


def highs_iters(A, b, c, lo, hi) -> tuple[int, float]:
    import highspy

    m, n = A.shape
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = c
    lp.col_lower_ = np.where(np.isfinite(lo), lo, -inf)
    lp.col_upper_ = np.where(np.isfinite(hi), hi, inf)
    lp.row_lower_, lp.row_upper_ = b, b
    Ac = A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Ac.indptr.astype(np.int32)
    lp.a_matrix_.index_ = Ac.indices.astype(np.int32)
    lp.a_matrix_.value_ = Ac.data.astype(float)
    h.passModel(lp)
    h.run()
    return int(h.getInfo().simplex_iteration_count), float(h.getObjectiveValue())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=",".join(SIMPLEX_ROUTED))
    ap.add_argument("--rules", default="1,0,5")
    ap.add_argument("--max-iter", type=int, default=200_000)
    ap.add_argument("--out", default="/tmp/ds2_chuzr_rule_sweep.json")
    a = ap.parse_args()

    rules = [int(x) for x in a.rules.split(",")]
    print(
        f"{'instance':13s} {'rows':>5} {'cols':>5} {'HiGHS':>7} "
        + " ".join(f"{RULES[r][:7]:>18s}" for r in rules)
    )
    out = []
    for name in a.instances.split(","):
        p = SUITE / f"{name}.mat"
        if not p.exists():
            print(f"{name:13s} MISSING")
            continue
        A, b, c, lo, hi = load(name)
        try:
            hx, hobj = highs_iters(A, b, c, lo, hi)
        except Exception as exc:  # noqa: BLE001 - probe
            hx, hobj = -1, float("nan")
            print(f"  ({name}: HiGHS failed {type(exc).__name__})")
        M, rb, rc, rlo, rhi = reduced(A, b, c, lo, hi)
        rec = {
            "instance": name,
            "rows": int(A.shape[0]),
            "cols": int(A.shape[1]),
            "red_rows": int(M.shape[0]),
            "red_cols": int(M.shape[1]),
            "highs_iters": hx,
            "highs_obj": hobj,
            "prior_ratio": SIMPLEX_ROUTED.get(name),
            "runs": {},
        }
        cells = []
        for rule in rules:
            t0 = time.perf_counter()
            try:
                r = M.solve_eq_box_dual_simplex(
                    rc,
                    rb,
                    rlo,
                    rhi,
                    max_iter=a.max_iter,
                    leaving_rule=rule,
                    expand=1,
                )
                ms = (time.perf_counter() - t0) * 1e3
                it = int(r["iterations"])
                rec["runs"][RULES[rule]] = {
                    "iterations": it,
                    "status": r["status"],
                    "ms": ms,
                    "us_per_pivot": 1e3 * ms / max(it, 1),
                    "ratio_vs_highs": (it / hx) if hx > 0 else None,
                }
                ok = "" if r["status"] == "optimal" else "!"
                cells.append(
                    f"{it:8d}{ok:1s} {it / hx if hx > 0 else float('nan'):5.2f}x {ms:7.0f}ms"
                )
            except Exception as exc:  # noqa: BLE001 - probe
                rec["runs"][RULES[rule]] = {"status": f"ERR:{type(exc).__name__}"}
                cells.append(f"{'ERR':>18s}")
        out.append(rec)
        print(
            f"{name:13s} {rec['rows']:5d} {rec['cols']:5d} {hx:7d} "
            + " ".join(f"{x:>18s}" for x in cells),
            flush=True,
        )

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nartifact: {a.out}")

    # what would the board look like on the best rule per instance?
    print("\nbest rule per instance (pivots, then wall):")
    for rec in out:
        ok = {k: v for k, v in rec["runs"].items() if v.get("status") == "optimal"}
        if not ok:
            print(f"  {rec['instance']:13s} no rule reached optimal")
            continue
        by_p = min(ok.items(), key=lambda kv: kv[1]["iterations"])
        by_w = min(ok.items(), key=lambda kv: kv[1]["ms"])
        base = rec["runs"].get("dantzig", {})
        bp = base.get("iterations")
        bw = base.get("ms")
        print(
            f"  {rec['instance']:13s} pivots: {by_p[0]:8s} {by_p[1]['iterations']:7d}"
            + (f" ({bp / by_p[1]['iterations']:5.2f}x vs dantzig)" if bp else "")
            + f"   wall: {by_w[0]:8s} {by_w[1]['ms']:7.0f}ms"
            + (f" ({bw / by_w[1]['ms']:5.2f}x vs dantzig)" if bw else "")
        )


if __name__ == "__main__":
    main()
