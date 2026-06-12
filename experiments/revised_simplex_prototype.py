"""Warm-started revised simplex prototype (crossover endgame probe).

Question: starting from the stalled first-order point's predicted
partition, how many simplex pivots does it take to reach a true optimal
basis on the cre-family instances? If the answer is small (hundreds to a
few thousand), a dependency-free production version is feasible: the C
Cholesky engine can solve square basis systems via B B^T u = rhs,
x = B^T u, so no LU machinery is required.

Prototype-only shortcuts: scipy splu for the basis solves (refactored
every pivot), dense QR with column pivoting to pick an independent warm
basis, Big-M artificials to cover rank deficiency. No public solver
source consulted; this is the textbook bounded-variable revised simplex.

Usage:
  PYTHONPATH=. uv run python experiments/revised_simplex_prototype.py lp_cre_a 23595407.06
"""

import sys
import time

import numpy as np
import scipy.linalg as dla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.io import loadmat

from linprogx.sparse import SparseLPProblem, SparseSolver, from_scipy_sparse

ACTIVE_TOL = 3e-5
PIVOT_TOL = 1e-9
RATIO_TOL = 1e-9
BIG_M_FACTOR = 1e7
MAX_PIVOTS = 8000


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
    return np.array(result.solution.x)


def warm_basis(A, c, lo, hi, x, m):
    """Rank `m` independent columns, preferring basic-looking ones."""
    scale = 1.0 + np.abs(x)
    at_lo = (x - lo) <= ACTIVE_TOL * scale
    at_hi = ((hi - x) <= ACTIVE_TOL * scale) & ~at_lo
    interior = np.nonzero(~(at_lo | at_hi))[0]
    pinned = np.nonzero(at_lo | at_hi)[0]
    # rank the pinned columns by min-norm reduced cost (basic-like first)
    B0 = A[:, interior]
    y0 = spla.lsqr(B0.T, c[interior], atol=1e-14, btol=1e-14, iter_lim=20000)[0]
    r0 = np.abs(c - A.T @ y0) / (1.0 + np.abs(c))
    pinned = pinned[np.argsort(r0[pinned])]
    # take interior + enough pinned candidates; one pivoted QR over
    # [candidates | down-scaled identity] picks an independent set and
    # falls back to identity (artificial) columns only where rank needs
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
    print(
        f"  warm basis: interior={len(interior)} chosen={len(chosen)} "
        f"artificials={len(art_rows)} (m={m})"
    )
    return chosen, art_rows, at_lo, at_hi


def revised_simplex(A, b, c, lo, hi, x_start):
    m, n = A.shape
    chosen, art_rows, at_lo0, at_hi0 = warm_basis(A, c, lo, hi, x_start, m)
    m - len(chosen)
    big_m = BIG_M_FACTOR * (1.0 + np.abs(c).max())
    # augment with artificial identity columns to reach a square basis
    A_aug = sp.hstack([A, sp.identity(m, format="csc")], format="csc")
    c_aug = np.concatenate([c, np.full(m, big_m)])
    lo_aug = np.concatenate([lo, np.full(m, -np.inf)])
    hi_aug = np.concatenate([hi, np.full(m, np.inf)])

    # complete with the identity columns the pivoted QR itself selected
    basis = np.concatenate([chosen, n + art_rows]).astype(np.int64)
    assert len(basis) == m, (len(chosen), len(art_rows), m)
    status = np.zeros(n + m, dtype=np.int8)  # 0 basic-or-free marker below
    AT_LO, AT_HI, BASIC = 1, 2, 3
    status[:n][at_lo0] = AT_LO
    status[:n][at_hi0] = AT_HI
    # non-selected interior columns: send to nearest finite bound
    for j in range(n):
        if status[j] == 0:
            status[j] = AT_LO if lo[j] > -np.inf else AT_HI
    status[n:] = AT_LO  # artificials nominally at "lo" (we treat as value 0 via lo=-inf? no)
    # artificials: treat as fixed at 0 when nonbasic
    lo_aug[n:] = 0.0
    hi_aug[n:] = 0.0
    status[basis] = BASIC

    t_factor = 0.0
    t_solve = 0.0
    degen_streak = 0
    for pivot in range(MAX_PIVOTS):
        Bmat = A_aug[:, basis]
        t0 = time.perf_counter()
        try:
            lu = spla.splu(Bmat.tocsc(), permc_spec="COLAMD")
        except RuntimeError:
            print(f"  pivot {pivot}: singular basis, aborting")
            return None
        t_factor += time.perf_counter() - t0
        t0 = time.perf_counter()
        nb_val = np.where(status == AT_LO, lo_aug, np.where(status == AT_HI, hi_aug, 0.0))
        nb_val[basis] = 0.0
        nb_val[~np.isfinite(nb_val)] = 0.0
        rhs = b - A_aug @ nb_val
        xb = lu.solve(rhs)
        y = lu.solve(c_aug[basis], trans="T")
        t_solve += time.perf_counter() - t0

        r = c_aug - A_aug.T @ y
        cscale = 1.0 + np.abs(c_aug)
        # certificate check: Lagrangian dual bound from these reduced
        # costs — violators with finite bounds are absorbable, so this
        # can certify before the simplex strictly terminates
        x_full = np.where(status == AT_LO, lo_aug, np.where(status == AT_HI, hi_aug, 0.0))
        x_full[basis] = xb
        obj_now = float(c_aug @ np.where(np.isfinite(x_full), x_full, 0.0))
        rn = r[:n]
        pos = rn > 0
        neg = rn < 0
        certifiable = np.all(
            (lo[:n] > -1e308)[pos] | (rn[pos] <= 1e-9 * cscale[:n][pos])
        ) and np.all((hi[:n] < 1e308)[neg] | (-rn[neg] <= 1e-9 * cscale[:n][neg]))
        if certifiable:
            dobj = float(b @ y)
            dobj += float(np.sum(np.where(pos & (lo > -1e308), rn * lo, 0.0)))
            dobj += float(np.sum(np.where(neg & (hi < 1e308), rn * hi, 0.0)))
            gap = obj_now - dobj
            pres_now = float(np.max(np.abs(A @ x_full[:n] - b)))
            art_ok = np.all(np.abs(x_full[n:]) <= 1e-9)
            if (
                abs(gap) <= 1e-9 * (1.0 + abs(obj_now))
                and pres_now <= 1e-7 * (1.0 + float(np.max(np.abs(b))))
                and art_ok
            ):
                print(
                    f"  CERTIFIED after {pivot} pivots: obj={obj_now:.10e} "
                    f"gap={gap:.2e} pres={pres_now:.2e} "
                    f"(factor {t_factor:.1f}s solve {t_solve:.1f}s)"
                )
                return x_full[:n], y, basis
        nonbasic = status != BASIC
        viol_lo = nonbasic & (status == AT_LO) & (r < -PIVOT_TOL * cscale) & (hi_aug > lo_aug)
        viol_hi = nonbasic & (status == AT_HI) & (r > PIVOT_TOL * cscale) & (hi_aug > lo_aug)
        cand = np.nonzero(viol_lo | viol_hi)[0]
        n_art_active = int(np.sum(np.abs(xb[np.nonzero(basis >= n)[0]]) > 1e-9))
        if len(cand) == 0:
            obj = float(c_aug[basis] @ xb + c_aug @ nb_val)
            x_full = nb_val.copy()
            x_full[basis] = xb
            pres = float(np.max(np.abs(A @ x_full[:n] - b)))
            print(
                f"  OPTIMAL BASIS after {pivot} pivots: obj={float(c[:n] @ x_full[:n]):.10e} "
                f"pres={pres:.2e} active_artificials={n_art_active} "
                f"(factor {t_factor:.1f}s solve {t_solve:.1f}s)"
            )
            return x_full[:n], y, basis
        if degen_streak > 200:
            j_in = int(cand.min())  # Bland's rule under degeneracy stall
        else:
            j_in = int(cand[np.argmax(np.abs(r[cand]) / cscale[cand])])
        going_up = status[j_in] == AT_LO

        d = lu.solve(A_aug[:, [j_in]].toarray().ravel())
        # entering moves t >= 0 (up from lo or down from hi); basic vars
        # move by -d t (up) or +d t (down)
        move = d if going_up else -d
        t_max = hi_aug[j_in] - lo_aug[j_in]  # bound flip distance
        leave = -1
        leave_to = 0
        for i in range(m):
            if move[i] > RATIO_TOL:
                lim = (xb[i] - lo_aug[basis[i]]) / move[i]
                if lim < t_max - 1e-15:
                    t_max = lim
                    leave = i
                    leave_to = AT_LO
            elif move[i] < -RATIO_TOL:
                lim = (xb[i] - hi_aug[basis[i]]) / move[i]
                if lim < t_max - 1e-15:
                    t_max = lim
                    leave = i
                    leave_to = AT_HI
        if t_max == np.inf:
            print(f"  pivot {pivot}: unbounded direction (j={j_in})")
            return None
        degen_streak = degen_streak + 1 if t_max <= 1e-12 else 0
        if pivot % 100 == 0:
            obj = float(c_aug[basis] @ xb + c_aug @ nb_val)
            print(
                f"  pivot {pivot}: viol={len(cand)} obj={obj:.6e} "
                f"art_active={n_art_active} t={t_max:.2e}"
            )
        if leave < 0:
            # bound flip: entering moves to its opposite bound
            status[j_in] = AT_HI if going_up else AT_LO
        else:
            status[basis[leave]] = leave_to
            status[j_in] = BASIC
            basis[leave] = j_in
    print("  pivot limit reached")
    return None


def main():
    name = sys.argv[1]
    published = float(sys.argv[2]) if len(sys.argv) > 2 else None
    A, b, c, lo, hi = load(name)
    t0 = time.perf_counter()
    x = stall_point(A, b, c, lo, hi, name)
    print(f"{name}: stall in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    out = revised_simplex(A, b, c, lo, hi, x)
    print(f"simplex time: {time.perf_counter() - t0:.1f}s")
    if out is not None and published is not None:
        obj = float(c @ out[0])
        print(f"rel err vs published: {abs(obj - published) / (1 + abs(published)):.2e}")


if __name__ == "__main__":
    main()
