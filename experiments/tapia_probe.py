"""Tapia-indicator basis identification probe (crossover go/no-go gate).

The simplex probes showed pivot count is governed by warm-basis quality:
a PDHG stall point misidentifies ~1,500 of cre_a's 3,516 basis members.
This probe asks whether the IPM's exit point does better. Indicator
(Tapia-style, from IPM convergence theory): a column is predicted basic
when its relative distance to the nearest bound dominates its relative
reduced-cost magnitude at the IPM's (x, y).

Reuses the committed primal simplex driver from
revised_simplex_prototype via a patched warm_basis so the only change
under test is the partition source. Go/no-go: initial violations in the
low hundreds and pivots ~<1,500 means the production crossover
subsystem is worth building; otherwise document and stop this line.

Usage:
  PYTHONPATH=. uv run python experiments/tapia_probe.py lp_cre_a 23595407.06
"""

import sys
import time

import numpy as np
import scipy.linalg as dla
import scipy.sparse as sp
from scipy.io import loadmat

import experiments.revised_simplex_prototype as rsp
from linprogx.sparse import from_scipy_sparse


def load(name):
    raw = loadmat(f"/tmp/lpsuite/{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    return (
        raw["A"].tocsc().astype(float),
        raw["b"].ravel().astype(float),
        aux["c"].ravel().astype(float),
        aux["lo"].ravel().astype(float),
        aux["hi"].ravel().astype(float),
    )


def ipm_point(A, b, c, lo, hi):
    M = from_scipy_sparse(sp.csr_matrix(A))
    result = M.solve_eq_box_ipm(
        c.tolist(), b.tolist(), lo.tolist(), hi.tolist(), max_iter=200, tol=1e-9
    )
    print(f"  ipm: status={result['status']} iters={result['iterations']} mu={result['mu']:.2e}")
    return np.array(result["x"]), np.array(result["y"])


def tapia_warm_basis(A, c, lo, hi, x, y, m):
    """Partition by Tapia-style indicator at the IPM exit point."""
    r = c - A.T @ y
    prim = np.minimum(
        np.where(lo > -1e308, x - lo, np.inf), np.where(hi < 1e308, hi - x, np.inf)
    ) / (1.0 + np.abs(x))
    dual = np.abs(r) / (1.0 + np.abs(c))
    basic_pred = prim > dual
    print(
        f"  tapia partition: basic_pred={int(basic_pred.sum())} (m={m}); "
        f"median prim/dual ratio split "
        f"{np.median(np.log10(np.maximum(prim, 1e-300) / np.maximum(dual, 1e-300))):.1f}"
    )
    interior = np.nonzero(basic_pred)[0]
    pinned = np.nonzero(~basic_pred)[0]
    # rank pinned columns: most basic-like first (largest prim/dual)
    score = prim[pinned] / np.maximum(dual[pinned], 1e-300)
    pinned = pinned[np.argsort(-score)]
    cand = np.concatenate([interior, pinned[: int(1.2 * m)]])
    dense = A[:, cand].toarray()
    col_norms = np.linalg.norm(dense, axis=0)
    eye_scale = 1e-3 * max(col_norms.min(), 1e-8)
    aug = np.hstack([dense, eye_scale * np.eye(m)])
    R, perm = dla.qr(aug, mode="r", pivoting=True)
    diag = np.abs(np.diag(R))
    rank = int(np.sum(diag > 1e-10 * diag[0]))
    take = perm[: min(rank, m)]
    chosen = cand[take[take < len(cand)]]
    art_rows = np.sort(take[take >= len(cand)] - len(cand))
    at_lo = ~basic_pred & np.where(lo > -1e308, (x - lo) <= (hi - x), np.zeros(len(x), dtype=bool))
    at_lo |= ~basic_pred & (hi >= 1e308)
    at_hi = ~basic_pred & ~at_lo
    print(f"  tapia warm basis: chosen={len(chosen)} artificials={len(art_rows)} (m={m})")
    return chosen, art_rows, at_lo, at_hi


def main():
    name = sys.argv[1]
    published = float(sys.argv[2]) if len(sys.argv) > 2 else None
    A, b, c, lo, hi = load(name)
    t0 = time.perf_counter()
    x, y = ipm_point(A, b, c, lo, hi)
    print(f"{name}: ipm point in {time.perf_counter() - t0:.1f}s")

    # patch only the partition source; the simplex driver is unchanged
    def patched_warm_basis(A, c, lo, hi, x, m):
        return tapia_warm_basis(A, c, lo, hi, x, y, m)

    setattr(rsp, "warm_basis", patched_warm_basis)  # noqa: B010 — experiment monkeypatch
    t0 = time.perf_counter()
    out = rsp.revised_simplex(A, b, c, lo, hi, x)
    print(f"simplex time: {time.perf_counter() - t0:.1f}s")
    if out is not None and published is not None:
        obj = float(c @ out[0])
        print(f"rel err vs published: {abs(obj - published) / (1 + abs(published)):.2e}")


if __name__ == "__main__":
    main()
