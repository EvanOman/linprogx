"""Crossover prototype: vertex purification from a stalled first-order point.

Given a near-optimal primal point x* (from IPM/PDHG stall), predict the
optimal bound partition, then alternate two exact solves until consistent:

  primal: fix nonbasic columns at their bounds, solve the restricted
          least-squares system for the basic columns;
  dual:   solve B^T y ~= c_B (least squares), compute reduced costs
          r = c - A^T y, and move sign-inconsistent columns between the
          basic and nonbasic sets.

Everything here uses scipy only for prototyping speed (lsqr); the
production path would reuse the C Cholesky engine via the normal
equations (B B^T u = rhs). No public solver source was consulted; this
is the textbook basis-purification idea.

Usage: PYTHONPATH=. uv run python experiments/crossover_prototype.py lp_cre_a 23595407.06
"""

import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.io import loadmat

from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

ACTIVE_TOL = 3e-5  # relative distance-to-bound below which a column is pinned
MAX_MOVES = 25  # violators repaired per round (worst-first; bulk swaps oscillate)


def load(name):
    raw = loadmat(f"/tmp/lpsuite/{name}.mat")["Problem"][0, 0]
    aux = raw["aux"][0, 0]
    A = raw["A"].tocsc().astype(float)
    return (
        A,
        raw["b"].ravel().astype(float),
        aux["c"].ravel().astype(float),
        aux["lo"].ravel().astype(float),
        aux["hi"].ravel().astype(float),
    )


def stall_point(A, b, c, lo, hi, name):
    bounds = [
        (lb if lb > -1e308 else None, ub if ub < 1e308 else None)
        for lb, ub in zip(lo, hi, strict=False)
    ]
    result = SparseSolver(algorithm="auto", eps=2e-5).solve(
        SparseLPProblem(
            c=c.tolist(),
            A_eq=from_scipy_sparse(sp.csr_matrix(A)),
            b_eq=b.tolist(),
            objective="min",
            bounds=bounds,
            name=name,
        )
    )
    return np.array(result.solution.x), result.solution.status.value


def crossover(A, b, c, lo, hi, x, max_rounds=40):
    m, n = A.shape
    scale = 1.0 + np.abs(x)
    at_lo = (x - lo) <= ACTIVE_TOL * scale
    at_hi = (hi - x) <= ACTIVE_TOL * scale
    # ties (zero-width boxes) count as at_lo
    at_hi &= ~at_lo
    basic = ~(at_lo | at_hi)
    print(
        f"  initial partition: basic={basic.sum()} at_lo={at_lo.sum()} at_hi={at_hi.sum()} (m={m})"
    )

    # Degenerate vertices keep basic columns AT bounds, so distance alone
    # under-counts the basis. Grow it to exactly m by promoting at-bound
    # columns whose min-norm reduced costs are smallest (most basic-like).
    if basic.sum() < m:
        B0 = A[:, basic]
        y0 = spla.lsqr(B0.T, c[basic], atol=1e-14, btol=1e-14, iter_lim=20000)[0]
        r0 = c - A.T @ y0
        cscale0 = 1.0 + np.abs(c)
        cand = np.nonzero(~basic)[0]
        order = cand[np.argsort(np.abs(r0[cand]) / cscale0[cand])]
        need = m - int(basic.sum())
        promote = order[:need]
        basic[promote] = True
        at_lo[promote] = False
        at_hi[promote] = False
        print(f"  promoted {need} degenerate columns; basic={basic.sum()}")

    for round_idx in range(max_rounds):
        xb_fixed = np.where(at_lo, lo, np.where(at_hi, hi, 0.0))
        xb_fixed[basic] = 0.0
        rhs = b - A @ xb_fixed
        B = A[:, basic]
        # primal: restricted least squares for the basic columns
        xb = spla.lsqr(B, rhs, atol=1e-14, btol=1e-14, iter_lim=20000)[0]
        x_new = xb_fixed.copy()
        x_new[basic] = xb

        # dual: least-squares y from the basic columns, reduced costs
        y = spla.lsqr(B.T, c[basic], atol=1e-14, btol=1e-14, iter_lim=20000)[0]
        r = c - A.T @ y

        cscale = 1.0 + np.abs(c)
        # violations
        finite_lo = lo > -1e308
        finite_hi = hi < 1e308
        nb_lo_bad = at_lo & (r < -1e-9 * cscale) & (hi > lo)  # wants to leave lo
        nb_hi_bad = at_hi & (r > 1e-9 * cscale)
        basic_lo_bad = basic & (x_new < lo - 1e-9 * scale)
        basic_hi_bad = basic & (x_new > hi + 1e-9 * scale)
        pres = float(np.max(np.abs(A @ x_new - b)))
        obj = float(c @ x_new)
        n_bad = (
            int(nb_lo_bad.sum())
            + int(nb_hi_bad.sum())
            + int(basic_lo_bad.sum())
            + int(basic_hi_bad.sum())
        )
        print(
            f"  round {round_idx}: pres={pres:.2e} obj={obj:.10e} "
            f"dual_viol={int(nb_lo_bad.sum() + nb_hi_bad.sum())} "
            f"primal_viol={int(basic_lo_bad.sum() + basic_hi_bad.sum())}"
        )
        if n_bad == 0 and pres <= 1e-7 * (1.0 + np.max(np.abs(b))):
            # exact certificate: dual objective from the reduced-cost split
            dobj = float(b @ y)
            rr = r.copy()
            ok = True
            add = 0.0
            for j in np.nonzero(np.abs(rr) > 0)[0]:
                if rr[j] > 0:
                    if finite_lo[j]:
                        add += rr[j] * lo[j]
                    elif rr[j] > 1e-9 * cscale[j]:
                        ok = False
                        break
                else:
                    if finite_hi[j]:
                        add += rr[j] * hi[j]
                    elif rr[j] < -1e-9 * cscale[j]:
                        ok = False
                        break
            gap = obj - (dobj + add) if ok else float("inf")
            print(
                f"  CONVERGED: obj={obj:.10e} certified_gap="
                f"{gap:.2e} rel={abs(gap) / (1 + abs(obj)):.2e}"
            )
            return x_new, y, obj, gap
        # repair with PAIRED swaps so |basic| stays m: each entering
        # dual violator displaces a leaving basic column (worst primal
        # violator first, else the basic column nearest a bound).
        enter = sorted(
            np.nonzero(nb_lo_bad | nb_hi_bad)[0],
            key=lambda j: -abs(r[j]) / cscale[j],
        )[:MAX_MOVES]
        leave_viol = sorted(
            np.nonzero(basic_lo_bad | basic_hi_bad)[0],
            key=lambda j: -max(lo[j] - x_new[j], x_new[j] - hi[j]) / scale[j],
        )
        # tie-break pool: basic columns closest to a bound (degenerate)
        basic_idx = np.nonzero(basic)[0]
        dist = np.minimum(
            np.where(lo[basic_idx] > -1e308, x_new[basic_idx] - lo[basic_idx], np.inf),
            np.where(hi[basic_idx] < 1e308, hi[basic_idx] - x_new[basic_idx], np.inf),
        )
        leave_pool = list(leave_viol) + list(basic_idx[np.argsort(dist)])
        used = set()
        moved = 0
        for j in enter:
            k = next((q for q in leave_pool if q not in used and q != j and basic[q]), None)
            if k is None:
                break
            used.add(k)
            at_lo[j] = at_hi[j] = False
            basic[j] = True
            basic[k] = False
            near_lo = lo[k] > -1e308 and (hi[k] >= 1e308 or x_new[k] - lo[k] <= hi[k] - x_new[k])
            at_lo[k] = near_lo
            at_hi[k] = not near_lo
            moved += 1
        # primal violators not consumed as leavers: push to violated bound,
        # backfill with nearest-zero-reduced-cost nonbasic column
        for k in leave_viol:
            if k in used or not basic[k]:
                continue
            basic[k] = False
            at_lo[k] = x_new[k] < lo[k]
            at_hi[k] = not at_lo[k]
            nb = np.nonzero(~basic & ~(at_lo & (hi <= lo)))[0]
            jj = nb[np.argmin(np.abs(r[nb]) / cscale[nb])]
            at_lo[jj] = at_hi[jj] = False
            basic[jj] = True
            moved += 1
        if moved == 0:
            print("  stalled: residual too high but no violations to repair")
            return x_new, y, obj, float("inf")
    print("  round limit reached")
    return x_new, y, obj, float("inf")


def main():
    name = sys.argv[1]
    published = float(sys.argv[2]) if len(sys.argv) > 2 else None
    A, b, c, lo, hi = load(name)
    t0 = time.perf_counter()
    x, status = stall_point(A, b, c, lo, hi, name)
    t_solve = time.perf_counter() - t0
    print(f"{name}: stall status={status} in {t_solve:.1f}s")
    t0 = time.perf_counter()
    x2, y, obj, gap = crossover(A, b, c, lo, hi, x)
    t_cross = time.perf_counter() - t0
    print(f"crossover time: {t_cross:.1f}s")
    if published is not None:
        print(f"rel err vs published: {abs(obj - published) / (1 + abs(published)):.2e}")


if __name__ == "__main__":
    main()
