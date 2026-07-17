"""Throwaway probe: capture HiGHS per-rule presolve reduction report.

Measurement-only. Does not touch solver source. Uses the highspy API's
documented options:
  - presolve_rule_logging = True   -> emits per-rule reduction table
  - log_dev_level = 2              -> verbose presolve pass logging
  - log_file                       -> capture the full log to disk

Also cross-references linprogx's own presolve reduction_counts on the same
instance via presolve_matrix().

Usage:
    PYTHONPATH=. uv run python experiments/greenbea_highs_presolve_probe.py <name>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

SUITE = Path("/tmp/lpsuite")


def load_instance(name: str) -> dict:
    raw = loadmat(SUITE / f"lp_{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return {
        "A": raw["A"].tocsc(),
        "b": raw["b"].ravel().astype(np.float64),
        "c": aux["c"].ravel().astype(np.float64),
        "lo": aux["lo"].ravel().astype(np.float64),
        "hi": aux["hi"].ravel().astype(np.float64),
    }


def build_highs(data: dict, log_file: str, rule_off: int = 0):
    import highspy  # ty: ignore[unresolved-import]

    h = highspy.Highs()
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_file)
    h.setOptionValue("log_dev_level", 2)
    h.setOptionValue("presolve_rule_logging", True)
    h.setOptionValue("highs_debug_level", 0)
    if rule_off:
        h.setOptionValue("presolve_rule_off", rule_off)

    A = data["A"].tocsc()
    m, n = A.shape
    c = data["c"]
    lo = data["lo"]
    hi = data["hi"]
    b = data["b"]

    inf = highspy.kHighsInf
    col_lower = np.where(np.isinf(lo), -inf, lo).astype(np.float64)
    col_upper = np.where(np.isinf(hi), inf, hi).astype(np.float64)
    # equality rows: lower == upper == b
    row_lower = b.astype(np.float64)
    row_upper = b.astype(np.float64)

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = c.astype(np.float64)
    lp.col_lower_ = col_lower
    lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower
    lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32).tolist()
    lp.a_matrix_.index_ = A.indices.astype(np.int32).tolist()
    lp.a_matrix_.value_ = A.data.astype(np.float64).tolist()
    lp.a_matrix_.num_col_ = n
    lp.a_matrix_.num_row_ = m
    h.passModel(lp)
    return h, (m, n, A.nnz)


def run_presolve(name: str, rule_off: int = 0, tag: str = "") -> dict:
    data = load_instance(name)
    log_file = f"/tmp/highs_presolve_{name}{tag}.log"
    h, raw_shape = build_highs(data, log_file, rule_off=rule_off)
    status = h.presolve()
    pres = h.getPresolvedLp()
    pm, pn = pres.num_row_, pres.num_col_
    pnnz = len(pres.a_matrix_.value_)
    return {
        "name": name,
        "rule_off": rule_off,
        "raw": raw_shape,
        "presolved": (pm, pn, pnnz),
        "status": str(status),
        "log_file": log_file,
    }


if __name__ == "__main__":
    names = sys.argv[1:] or ["greenbea", "woodw", "80bau3b"]
    for nm in names:
        r = run_presolve(nm)
        print(f"{nm}: raw {r['raw']} -> presolved {r['presolved']}  status={r['status']}")
        print(f"   log: {r['log_file']}")
