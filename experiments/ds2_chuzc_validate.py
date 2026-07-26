"""DS2 component A (CHUZC) -- validation and cost harness.

PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.

What this does
--------------
1. Runs the SHIPPED dual simplex on real LPnetlib instances with
   ``LINPROGX_DS2_DUMP`` set, which harvests, for a strided sample of pivots,
   the complete input state of the ratio test plus the decision the shipped
   test made.  The harvest hook is observation-only; the trace-hash oracle
   (``LINPROGX_DS_TRACE_HASH=1``) shows the solve path is bit-identical with
   it compiled in.
2. Replays every harvested pivot through three implementations, all in
   ``linprogx._ds2_chuzc``:

   ``harris_dense``    reimplementation of the shipped Harris two-pass test,
                       scanning columns 0..n_total-1 as the shipped one does.
                       Agreement with the recorded decision is the FIDELITY
                       CHECK: it says the control is really the incumbent.
   ``harris_pattern``  the same test restricted to the pivot row's support.
                       Isolates scan shape from algorithm.
   ``ds2``             the bound-flipping test under test.

3. Reports selection agreement and per-call cycle counts (rdtsc, minimum of
   ``--repeat`` runs -- load-invariant, unlike wall).

Usage
-----
    uv run python experiments/ds2_chuzc_validate.py --instances greenbea degen2
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

SUITE = Path("/tmp/lpsuite")
INF = float("inf")
MAGIC = 0x44533243  # "DS2C"


# --------------------------------------------------------------------------
# harvest
# --------------------------------------------------------------------------
def load_instance(name: str) -> dict[str, Any]:
    from scipy.io import loadmat

    raw = loadmat(SUITE / f"lp_{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(float),
        "c": aux["c"].ravel().astype(float),
        "lo": aux["lo"].ravel().astype(float),
        "hi": aux["hi"].ravel().astype(float),
    }


def harvest(name: str, dump_path: Path, stride: int, max_cases: int) -> dict[str, Any]:
    """Run the shipped solver via the public auto route, harvesting pivots."""
    from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

    d = load_instance(name)
    bounds = [
        (None if low == -INF else float(low), None if up == INF else float(up))
        for low, up in zip(d["lo"], d["hi"], strict=True)
    ]
    os.environ["LINPROGX_DS2_DUMP"] = str(dump_path)
    os.environ["LINPROGX_DS2_DUMP_STRIDE"] = str(stride)
    os.environ["LINPROGX_DS2_DUMP_MAX"] = str(max_cases)
    try:
        result = SparseSolver(algorithm="dual_simplex", max_iterations=200_000, eps=2e-5).solve(
            SparseLPProblem(
                c=d["c"].tolist(),
                A_eq=from_scipy_sparse(d["A"]),
                b_eq=d["b"].tolist(),
                objective="min",
                bounds=bounds,
            )
        )
    finally:
        for key in (
            "LINPROGX_DS2_DUMP",
            "LINPROGX_DS2_DUMP_STRIDE",
            "LINPROGX_DS2_DUMP_MAX",
        ):
            os.environ.pop(key, None)
    return {
        "status": result.solution.status.value,
        "iterations": int(result.solution.iterations),
        "objective": result.solution.objective_value,
        "backend": result.backend,
    }


def parse_dump(path: Path) -> list[dict[str, Any]]:
    blob = path.read_bytes()
    cases: list[dict[str, Any]] = []
    off = 0
    while off < len(blob):
        magic, it, n, n_total, sigma, upd, nnz = struct.unpack_from("<7i", blob, off)
        if magic != MAGIC:
            raise ValueError(f"bad magic at offset {off}")
        off += 28
        delta, expand_tau = struct.unpack_from("<2d", blob, off)
        off += 16
        pattern = np.frombuffer(blob, np.int32, nnz, off).copy()
        off += 4 * nnz
        avals = np.frombuffer(blob, np.float64, nnz, off).copy()
        off += 8 * nnz
        r_ext = np.frombuffer(blob, np.float64, n_total, off).copy()
        off += 8 * n_total
        bstat = np.frombuffer(blob, np.int8, n_total, off).copy()
        off += n_total
        lo = np.frombuffer(blob, np.float64, n_total, off).copy()
        off += 8 * n_total
        hi = np.frombuffer(blob, np.float64, n_total, off).copy()
        off += 8 * n_total
        art = np.frombuffer(blob, np.int8, n, off).copy()
        off += n
        entering, n_flips = struct.unpack_from("<2i", blob, off)
        off += 8
        alpha_pivot, theta_d = struct.unpack_from("<2d", blob, off)
        off += 16
        flips = np.frombuffer(blob, np.int32, n_flips, off).copy()
        off += 4 * n_flips

        alpha = np.zeros(n_total, np.float64)
        alpha[pattern] = avals
        no_flip = np.zeros(n_total, np.uint8)
        no_flip[:n] = art.astype(np.uint8)
        cases.append(
            {
                "iter": it,
                "n": n,
                "n_total": n_total,
                "sigma": sigma,
                "update_count": upd,
                "delta": delta,
                "expand_tau": expand_tau,
                "pattern": np.sort(pattern),
                "alpha": alpha,
                "r_ext": r_ext,
                "bound_status": bstat,
                "lo": lo,
                "hi": hi,
                "no_flip": no_flip,
                "ref_entering": entering,
                "ref_alpha": alpha_pivot,
                "ref_theta": theta_d,
                "ref_flips": sorted(int(v) for v in flips),
            }
        )
    return cases


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------
def replay(cases: list[dict[str, Any]], repeat: int) -> dict[str, Any]:
    ds2 = importlib.import_module("linprogx._ds2_chuzc")

    # Four arms.  The first three share the fixed contract; "ds2_noart" is a
    # COUNTERFACTUAL, not a legal solver configuration: it lets DS2 flip the
    # big-M artificial boxes as if they were genuine bounds, which isolates
    # "the mechanism does not fire" from "the formulation gives it nothing to
    # fire on".
    kinds = ("harris_dense", "harris_pattern", "ds2", "ds2_noart")
    fn_of = {
        "harris_dense": "harris_dense",
        "harris_pattern": "harris_pattern",
        "ds2": "ds2",
        "ds2_noart": "ds2",
    }
    n_total = cases[0]["n_total"]
    states = {k: ds2.State(n_total) for k in kinds}

    cycles: dict[str, list[int]] = {k: [] for k in kinds}
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in kinds}
    census: list[dict[str, Any]] = []

    prev_bounds_key: object = None
    for case in cases:
        if case["n_total"] != n_total:
            raise ValueError("n_total changed mid-run")
        bounds_key = (case["lo"].tobytes(), case["hi"].tobytes())
        bounds_changed = bounds_key != prev_bounds_key
        prev_bounds_key = bounds_key
        for k in kinds:
            st = states[k]
            if bounds_changed:
                st.set_no_flip(None if k == "ds2_noart" else case["no_flip"])
                st.build_range(case["lo"], case["hi"])
            res = ds2.chuzc(
                fn_of[k],
                st,
                case["alpha"],
                case["pattern"],
                case["r_ext"],
                case["bound_status"],
                case["lo"],
                case["hi"],
                case["sigma"],
                case["delta"],
                update_count=case["update_count"],
                dual_tol=1e-7,
                expand_tau=case["expand_tau"],
                harris_delta=1e-7,
                repeat=repeat,
            )
            cycles[k].append(res["cycles"])
            out[k].append(res)
        # untimed census pass on the legal ds2 arm
        st = states["ds2"]
        st.set_census(True)
        ds2.chuzc(
            "ds2",
            st,
            case["alpha"],
            case["pattern"],
            case["r_ext"],
            case["bound_status"],
            case["lo"],
            case["hi"],
            case["sigma"],
            case["delta"],
            update_count=case["update_count"],
            dual_tol=1e-7,
            expand_tau=case["expand_tau"],
            harris_delta=1e-7,
            repeat=1,
        )
        st.set_census(False)
        census.append(st.census())

    report: dict[str, Any] = {
        "cases": len(cases),
        "mean_alpha_nnz": float(np.mean([len(c["pattern"]) for c in cases])),
        "mean_n_total": n_total,
        "cycles": {
            k: {
                "median": statistics.median(cycles[k]),
                "mean": statistics.fmean(cycles[k]),
                "p90": sorted(cycles[k])[int(0.9 * (len(cycles[k]) - 1))],
                "total": sum(cycles[k]),
            }
            for k in kinds
        },
        "stats": {k: states[k].stats() for k in kinds},
    }

    # fidelity of the control against the recorded shipped decision
    same_dense = sum(
        1
        for c, r in zip(cases, out["harris_dense"], strict=True)
        if r["entering"] == c["ref_entering"]
    )
    same_pattern = sum(
        1
        for c, r in zip(cases, out["harris_pattern"], strict=True)
        if r["entering"] == c["ref_entering"]
    )
    same_ds2 = sum(
        1 for c, r in zip(cases, out["ds2"], strict=True) if r["entering"] == c["ref_entering"]
    )
    ds2_vs_dense = sum(
        1
        for a, b in zip(out["ds2"], out["harris_dense"], strict=True)
        if a["entering"] == b["entering"]
    )
    report["agreement"] = {
        "harris_dense_vs_shipped": same_dense,
        "harris_pattern_vs_shipped": same_pattern,
        "ds2_vs_shipped": same_ds2,
        "ds2_vs_harris_dense": ds2_vs_dense,
    }

    flips = [len(r["flips"]) for r in out["ds2"]]
    flips_cf = [len(r["flips"]) for r in out["ds2_noart"]]
    ref_flips = [len(c["ref_flips"]) for c in cases]
    report["flips"] = {
        "ds2_total": sum(flips),
        "ds2_calls_with_flips": sum(1 for f in flips if f),
        "ds2_max": max(flips) if flips else 0,
        "ds2_noart_total": sum(flips_cf),
        "ds2_noart_calls_with_flips": sum(1 for f in flips_cf if f),
        "shipped_total": sum(ref_flips),
        "shipped_calls_with_flips": sum(1 for f in ref_flips if f),
    }
    # Can flips POSSIBLY cover the row's infeasibility in this formulation?
    covered = sum(1 for c in census if c["absorb"] >= c["delta"])
    report["census"] = {
        "mean_cand": float(np.mean([c["n_cand"] for c in census])),
        "mean_flippable": float(np.mean([c["n_flippable"] for c in census])),
        "frac_flippable": float(np.mean([c["n_flippable"] / max(1, c["n_cand"]) for c in census])),
        "calls_where_absorb_ge_delta": covered,
        "calls_partition_exhausted": sum(1 for c in census if c["exhausted"]),
        "calls_degenerate_zero_step": sum(1 for c in census if c["degenerate"]),
        "mean_groups": float(np.mean([c["groups"] for c in census])),
        "calls_one_group": sum(1 for c in census if c["groups"] <= 1),
        "median_absorb_over_delta": float(
            np.median([c["absorb"] / c["delta"] if c["delta"] > 0 else np.inf for c in census])
        ),
    }
    report["groups"] = {
        "mean": float(np.mean([r["groups"] for r in out["ds2"]])),
        "max": max(r["groups"] for r in out["ds2"]),
    }

    # theta agreement on the calls where both picked the same column and ds2
    # flipped nothing -- there the two tests must be doing the same thing,
    # EXCEPT that DS2 clamps the step to zero when the entering column is
    # already dual infeasible (r_q on the wrong side of its bound, which
    # EXPAND admits).  Those are counted separately, not as errors.
    theta_rel: list[float] = []
    n_clamped = 0
    for c, a, b in zip(cases, out["ds2"], out["harris_dense"], strict=True):
        if a["entering"] != b["entering"] or a["flips"] or a["entering"] < 0:
            continue
        q = a["entering"]
        move = 1.0 if c["bound_status"][q] == 0 else -1.0
        if c["r_ext"][q] * move <= 0.0:
            n_clamped += 1
            continue
        denom = max(abs(b["theta_dual"]), 1e-30)
        theta_rel.append(abs(a["theta_dual"] - b["theta_dual"]) / denom)
    report["theta_rel_err_max_noflip"] = max(theta_rel) if theta_rel else None
    report["theta_rel_err_n"] = len(theta_rel)
    report["theta_clamped_dual_infeasible"] = n_clamped

    # ---- soundness: the properties a bound-flipping ratio test must have --
    viol: dict[str, int] = {
        "entering_not_admissible": 0,
        "entering_alpha_below_Ta": 0,
        "flip_not_boxed": 0,
        "flip_is_no_flip_column": 0,
        "flip_equals_entering": 0,
        "flip_ratio_above_entering": 0,
        "step_negative": 0,
    }
    longer_step = 0
    for c, a, b in zip(cases, out["ds2"], out["harris_dense"], strict=True):
        q = a["entering"]
        if q < 0:
            continue
        bs = c["bound_status"][q]
        alpha_q = c["alpha"][q]
        sig = c["sigma"]
        ok = (bs == 0 and sig * alpha_q < 0) or (bs == 1 and sig * alpha_q > 0) or bs == 2
        if not ok:
            viol["entering_not_admissible"] += 1
        ta = 1e-9 if c["update_count"] < 10 else 3e-8 if c["update_count"] < 20 else 1e-6
        if abs(alpha_q) <= ta:
            viol["entering_alpha_below_Ta"] += 1
        if a["theta_dual"] < 0:
            viol["step_negative"] += 1
        ratio_q = abs(c["r_ext"][q] / alpha_q)
        if b["entering"] >= 0:
            ratio_h = abs(c["r_ext"][b["entering"]] / b["alpha_pivot"])
            if ratio_q > ratio_h + 1e-12:
                longer_step += 1
        for j in a["flips"]:
            if j == q:
                viol["flip_equals_entering"] += 1
            if not (np.isfinite(c["lo"][j]) and np.isfinite(c["hi"][j])):
                viol["flip_not_boxed"] += 1
            if c["no_flip"][j]:
                viol["flip_is_no_flip_column"] += 1
            aj = c["alpha"][j]
            if aj != 0.0 and abs(c["r_ext"][j] / aj) > ratio_q + 1e-9 * max(1.0, ratio_q):
                viol["flip_ratio_above_entering"] += 1
    report["soundness"] = viol
    report["ds2_longer_step_than_harris"] = longer_step

    # a sample of disagreements, for the report
    diffs = []
    for c, a, b in zip(cases, out["ds2"], out["harris_dense"], strict=True):
        if a["entering"] != b["entering"] and len(diffs) < 12:
            diffs.append(
                {
                    "iter": c["iter"],
                    "shipped": c["ref_entering"],
                    "harris_dense": b["entering"],
                    "ds2": a["entering"],
                    "ds2_flips": len(a["flips"]),
                    "ds2_alpha": a["alpha_pivot"],
                    "harris_alpha": b["alpha_pivot"],
                    "ds2_ratio": (
                        abs(c["r_ext"][a["entering"]] / a["alpha_pivot"])
                        if a["entering"] >= 0
                        else None
                    ),
                    "harris_ratio": (
                        abs(c["r_ext"][b["entering"]] / b["alpha_pivot"])
                        if b["entering"] >= 0
                        else None
                    ),
                }
            )
    report["sample_disagreements"] = diffs
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="+", default=["greenbea", "degen2", "25fv47"])
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--max-cases", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=25)
    ap.add_argument("--tmpdir", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    ds2 = importlib.import_module("linprogx._ds2_chuzc")

    if not ds2.have_tsc():
        print("WARNING: no rdtsc on this platform; 'cycles' are nanoseconds")

    all_reports: dict[str, Any] = {}
    for name in args.instances:
        dump = Path(args.tmpdir) / f"ds2_chuzc_{name}.bin"
        solve = harvest(name, dump, args.stride, args.max_cases)
        cases = parse_dump(dump)
        if not cases:
            print(f"{name}: no pivots harvested (route did not use dual simplex?)")
            continue
        rep = replay(cases, args.repeat)
        rep["solve"] = solve
        all_reports[name] = rep

        c = rep["cycles"]
        a = rep["agreement"]
        print(f"\n=== {name} ===")
        print(f"  solve: {solve['status']} in {solve['iterations']} iters ({solve['backend']})")
        print(
            f"  cases {rep['cases']}  n_total {rep['mean_n_total']}  "
            f"mean alpha nnz {rep['mean_alpha_nnz']:.0f}"
        )
        print("  agreement (entering column):")
        print(f"    harris_dense   vs shipped : {a['harris_dense_vs_shipped']}/{rep['cases']}")
        print(f"    harris_pattern vs shipped : {a['harris_pattern_vs_shipped']}/{rep['cases']}")
        print(f"    ds2            vs shipped : {a['ds2_vs_shipped']}/{rep['cases']}")
        print("  cycles/call (min of repeats; median over cases):")
        for k in ("harris_dense", "harris_pattern", "ds2", "ds2_noart"):
            print(f"    {k:<15} median {c[k]['median']:>9.0f}  p90 {c[k]['p90']:>9.0f}")
        base = c["harris_dense"]["median"]
        print(
            f"    ds2 / harris_dense = {c['ds2']['median'] / base:.3f}x   "
            f"harris_pattern / harris_dense = "
            f"{c['harris_pattern']['median'] / base:.3f}x   "
            f"ds2 / harris_pattern = "
            f"{c['ds2']['median'] / c['harris_pattern']['median']:.3f}x"
        )
        f = rep["flips"]
        print(
            f"  flips: ds2 {f['ds2_total']} over {f['ds2_calls_with_flips']} calls "
            f"(max {f['ds2_max']}); shipped {f['shipped_total']} over "
            f"{f['shipped_calls_with_flips']} calls; counterfactual "
            f"(artificial boxes flippable) {f['ds2_noart_total']} over "
            f"{f['ds2_noart_calls_with_flips']}"
        )
        cen = rep["census"]
        print(
            f"  census: {cen['mean_flippable']:.0f}/{cen['mean_cand']:.0f} "
            f"candidates have a finite range "
            f"({100 * cen['frac_flippable']:.1f}%); their total absorption "
            f"covers the row infeasibility on "
            f"{cen['calls_where_absorb_ge_delta']}/{rep['cases']} calls "
            f"(median absorb/delta {cen['median_absorb_over_delta']:.3g})"
        )
        print(
            f"  partition: mean {cen['mean_groups']:.2f} groups/call; "
            f"{cen['calls_one_group']}/{rep['cases']} calls yield a single "
            f"group (nothing can be stepped over); "
            f"{cen['calls_partition_exhausted']}/{rep['cases']} ended by "
            f"running out of candidates rather than by covering delta; "
            f"{cen['calls_degenerate_zero_step']}/{rep['cases']} took a "
            f"zero dual step"
        )
        bad = {k: v for k, v in rep["soundness"].items() if v}
        print(
            f"  soundness: {'OK -- no violations' if not bad else bad}; "
            f"ds2 took a strictly longer dual step than the incumbent on "
            f"{rep['ds2_longer_step_than_harris']}/{rep['cases']} calls"
        )
        if rep["theta_rel_err_max_noflip"] is not None:
            print(
                f"  theta_dual max rel err on the "
                f"{rep['theta_rel_err_n']} same-column no-flip calls: "
                f"{rep['theta_rel_err_max_noflip']:.3e} "
                f"({rep['theta_clamped_dual_infeasible']} more clamped to 0 "
                f"because the entering column was already dual infeasible)"
            )

    if args.json:
        Path(args.json).write_text(json.dumps(all_reports, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
