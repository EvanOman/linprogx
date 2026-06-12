"""Warm-started bounded-variable dual simplex prototype (crossover probe).

The primal-simplex probe (revised_simplex_prototype.py) reached the
published cre_a optimum but burned ~20k pivots recovering from a
primal-infeasible warm start. This probe flips the warm start around:

  - same pivoted-QR warm basis B from the stall point's partition;
  - y solves B^T y = c_B, so r_B = 0 — then every NONBASIC column is
    placed on the bound matching sign(r_j), making the start dual
    feasible BY CONSTRUCTION;
  - columns whose sign-matching bound is infinite get a temporary
    expanded bound (textbook dual phase-1 device); if any ends active
    at termination the bound is enlarged and the run continues;
  - dual pivots then repair the (hopefully few) primal violations.

scipy splu refactored per pivot — prototype only; production would use
the C Cholesky square-basis solve. No public solver source consulted.

Usage:
  PYTHONPATH=. uv run python experiments/dual_simplex_prototype.py lp_cre_a 23595407.06
"""

import sys
import time

import numpy as np
import scipy.linalg as dla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.io import loadmat

from linprogx.sparse import from_scipy_sparse

ACTIVE_TOL = 3e-5
DUAL_TOL = 1e-9
RATIO_TOL = 1e-9
PRIMAL_TOL = 1e-9
MAX_PIVOTS = 20000
AT_LO, AT_HI, BASIC = 1, 2, 3


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
    # call the C PDHG backend directly: the high-level API drops the
    # dual iterate, and the stall y (only ~tens of violating reduced
    # costs) is the whole value of the warm start
    M = from_scipy_sparse(sp.csr_matrix(A))
    result = M.solve_eq_box_pdhg(
        c.tolist(),
        b.tolist(),
        lo.tolist(),
        hi.tolist(),
        max_iter=50_000,
        tol=2e-5,
        check_interval=2048,
    )
    print(f"  stall backend status={result['status']} iters={result['iterations']}")
    return np.array(result["x"]), np.array(result["y"])


def warm_basis(A, c, lo, hi, x, y_stall, m):
    scale = 1.0 + np.abs(x)
    at_lo = (x - lo) <= ACTIVE_TOL * scale
    at_hi = ((hi - x) <= ACTIVE_TOL * scale) & ~at_lo
    interior = np.nonzero(~(at_lo | at_hi))[0]
    pinned = np.nonzero(at_lo | at_hi)[0]
    # rank pinned columns by the SOLVER's stall dual, not a min-norm
    # least-squares dual (measured: the latter shows ~1,700 spurious
    # violations, the former only ~tens)
    r0 = np.abs(c - A.T @ y_stall) / (1.0 + np.abs(c))
    pinned = pinned[np.argsort(r0[pinned])]
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
    return chosen, art_rows


def dual_simplex(A, b, c, lo, hi, x_start, y_stall):
    m, n = A.shape
    chosen, art_rows = warm_basis(A, c, lo, hi, x_start, y_stall, m)

    # artificial identity columns (cost 0, bounds [0,0]) cover rank gaps;
    # fixed at 0 they never re-enter and never affect dual feasibility
    A_aug = sp.hstack([A, sp.identity(m, format="csc")[:, art_rows]], format="csc")
    n_aug = n + len(art_rows)
    c_aug = np.concatenate([c, np.zeros(len(art_rows))])
    lo_aug = np.concatenate([lo, np.zeros(len(art_rows))])
    hi_aug = np.concatenate([hi, np.zeros(len(art_rows))])

    basis = np.concatenate([chosen, np.arange(n, n_aug)]).astype(np.int64)
    assert len(basis) == m

    # expanded bounds for dual phase-1: rule-based margin off the stall x
    expand_hi = np.concatenate([1e4 * (1.0 + np.abs(x_start)), np.zeros(len(art_rows))])
    expand_lo = -expand_hi
    art_hi = np.zeros(n_aug, dtype=bool)
    art_lo = np.zeros(n_aug, dtype=bool)

    status = np.full(n_aug, AT_LO, dtype=np.int8)
    status[basis] = BASIC

    t_total = time.perf_counter()
    expansions = 0
    pivot = 0
    degen_streak = 0
    while pivot < MAX_PIVOTS:
        Bmat = A_aug[:, basis].tocsc()
        try:
            lu = spla.splu(Bmat, permc_spec="COLAMD")
        except RuntimeError:
            print(f"  pivot {pivot}: singular basis, aborting")
            return None
        y = lu.solve(c_aug[basis], trans="T")
        r = c_aug - A_aug.T @ y
        cscale = 1.0 + np.abs(c_aug)

        # place nonbasic columns on the sign-matching bound (dual
        # feasibility); infinite side -> temporary expanded bound
        for j in np.nonzero(status != BASIC)[0]:
            if r[j] >= -DUAL_TOL * cscale[j]:
                if lo_aug[j] > -np.inf:
                    status[j] = AT_LO
                    art_lo[j] = False
                else:
                    status[j] = AT_LO
                    art_lo[j] = True
            else:
                if hi_aug[j] < np.inf:
                    status[j] = AT_HI
                    art_hi[j] = False
                else:
                    status[j] = AT_HI
                    art_hi[j] = True

        nb_val = np.where(
            status == AT_LO,
            np.where(art_lo, expand_lo, lo_aug),
            np.where(art_hi, expand_hi, hi_aug),
        )
        nb_val[basis] = 0.0
        rhs = b - A_aug @ nb_val
        xb = lu.solve(rhs)

        bscale = 1.0 + np.abs(xb)
        viol_lo_b = xb < lo_aug[basis] - PRIMAL_TOL * bscale
        viol_hi_b = xb > hi_aug[basis] + PRIMAL_TOL * bscale
        viol = np.nonzero(viol_lo_b | viol_hi_b)[0]

        if pivot % 100 == 0:
            x_full = nb_val.copy()
            x_full[basis] = xb
            obj = float(c_aug @ x_full)
            print(
                f"  pivot {pivot}: primal_viol={len(viol)} obj={obj:.6e} "
                f"art_bounds_active={int(np.sum((art_lo | art_hi) & (status != BASIC)))}"
            )
        if len(viol) == 0:
            # primal + dual feasible; check no expanded bound is load-bearing
            x_full = nb_val.copy()
            x_full[basis] = xb
            on_art = (art_lo | art_hi) & (status != BASIC)
            if np.any(on_art):
                expansions += 1
                if expansions > 5:
                    print("  expanded bounds still active after 5 enlargements")
                    return None
                expand_hi[on_art] *= 100.0
                expand_lo = -expand_hi
                print(f"  enlarging {int(on_art.sum())} expanded bounds, continuing")
                pivot += 1
                continue
            obj = float(c[:n] @ x_full[:n])
            pres = float(np.max(np.abs(A @ x_full[:n] - b)))
            dobj = float(b @ y)
            rn = r[:n]
            dobj += float(np.sum(np.where((rn > 0) & (lo > -1e308), rn * lo, 0.0)))
            dobj += float(np.sum(np.where((rn < 0) & (hi < 1e308), rn * hi, 0.0)))
            bad_pos = (rn > 1e-9 * cscale[:n]) & (lo <= -1e308)
            bad_neg = (rn < -1e-9 * cscale[:n]) & (hi >= 1e308)
            certifiable = not (np.any(bad_pos) or np.any(bad_neg))
            gap = obj - dobj if certifiable else float("inf")
            print(
                f"  OPTIMAL after {pivot} pivots ({time.perf_counter() - t_total:.1f}s): "
                f"obj={obj:.10e} pres={pres:.2e} certifiable={certifiable} "
                f"gap={gap:.3e} rel={abs(gap) / (1 + abs(obj)):.2e}"
            )
            return x_full[:n], y, basis

        # leaving: worst primal violation
        worst = viol[
            np.argmax(
                np.maximum(lo_aug[basis][viol] - xb[viol], xb[viol] - hi_aug[basis][viol])
                / bscale[viol]
            )
        ]
        leaving_low = bool(viol_lo_b[worst])  # leaves to its LOWER bound
        # dual direction: w = B^{-T} e_worst ; alpha_j = a_j . w
        e = np.zeros(m)
        e[worst] = 1.0
        w = lu.solve(e, trans="T")
        alpha = A_aug.T @ w  # length n_aug

        # ratio test over nonbasic candidates that restore feasibility:
        # leaving to LOWER bound means x_B[worst] must increase, so the
        # entering column j must have alpha_j with the right sign given
        # which bound j sits on (moving j off its bound raises x_B[worst])
        nonbasic = status != BASIC
        if leaving_low:
            cand_mask = nonbasic & (
                ((status == AT_LO) & (alpha < -RATIO_TOL))
                | ((status == AT_HI) & (alpha > RATIO_TOL))
            )
        else:
            cand_mask = nonbasic & (
                ((status == AT_LO) & (alpha > RATIO_TOL))
                | ((status == AT_HI) & (alpha < -RATIO_TOL))
            )
        cand = np.nonzero(cand_mask)[0]
        if len(cand) == 0:
            print(f"  pivot {pivot}: dual unbounded (primal infeasible row {worst})")
            return None
        ratios = np.abs(r[cand]) / np.abs(alpha[cand])
        if degen_streak > 200:
            j_in = int(cand[np.argmin(cand)])  # Bland fallback
        else:
            j_in = int(cand[np.argmin(ratios)])
        if np.abs(r[j_in]) <= 1e-13 * cscale[j_in]:
            degen_streak += 1
        else:
            degen_streak = 0

        status[basis[worst]] = AT_LO if leaving_low else AT_HI
        status[j_in] = BASIC
        basis[worst] = j_in
        pivot += 1

    print("  pivot limit reached")
    return None


def main():
    name = sys.argv[1]
    published = float(sys.argv[2]) if len(sys.argv) > 2 else None
    A, b, c, lo, hi = load(name)
    t0 = time.perf_counter()
    x, y_stall = stall_point(A, b, c, lo, hi, name)
    print(f"{name}: stall in {time.perf_counter() - t0:.1f}s")
    out = dual_simplex(A, b, c, lo, hi, x, y_stall)
    if out is not None and published is not None:
        obj = float(c @ out[0])
        print(f"rel err vs published: {abs(obj - published) / (1 + abs(published)):.2e}")


if __name__ == "__main__":
    main()
