"""Mehrotra predictor-corrector IPM prototype (scipy-based).

Validates the algorithm choices before a dependency-free C implementation:
normal equations A D A' solved by direct sparse LU, primal-dual
regularization for rank-deficient rows, native box-bound handling
(z_l for finite lower bounds, z_u for finite upper bounds, free variables
regularized).

Run: uv run python experiments/ipm_prototype.py
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from bench_cycle import DATA_PATH as CYCLE_PATH
from bench_cycle import EXPECTED_CYCLE_OBJECTIVE, load_cycle
from bench_cycle import _bounds as cycle_bounds
from bench_large import DATA_PATH as DFL_PATH
from bench_large import EXPECTED_DFL001_OBJECTIVE, load_dfl001
from bench_large import _bounds as dfl_bounds
from linprogx.presolve import postsolve_x, presolve_eq_box


def ruiz_scale(A, iters=10):
    A = sp.csr_matrix(A, copy=True)
    m, n = A.shape
    r = np.ones(m)
    s = np.ones(n)
    for _ in range(iters):
        Aabs = sp.csr_matrix((np.abs(A.data), A.indices, A.indptr), shape=A.shape)
        row_max = np.asarray(Aabs.max(axis=1).todense()).ravel()
        col_max = np.asarray(Aabs.max(axis=0).todense()).ravel()
        row_max[row_max == 0] = 1.0
        col_max[col_max == 0] = 1.0
        dr = 1.0 / np.sqrt(row_max)
        dc = 1.0 / np.sqrt(col_max)
        A = sp.diags(dr) @ A @ sp.diags(dc)
        r *= dr
        s *= dc
    return sp.csr_matrix(A), r, s


def ipm_solve(A, b, c, lo, hi, *, max_iter=60, tol=1e-9, verbose=True, start="simple"):
    # Equilibrate: A_s = R A S, x = S x_s, y = R y_s.
    A, row_scale, col_scale = ruiz_scale(A)
    b = b * row_scale
    c = c * col_scale
    lo = np.where(np.isfinite(lo), lo / col_scale, lo)
    hi = np.where(np.isfinite(hi), hi / col_scale, hi)
    c_scale = max(1.0, np.max(np.abs(c)))
    c = c / c_scale
    saved_scales = (row_scale, col_scale, c_scale)

    m, n = A.shape
    A = sp.csr_matrix(A)

    # Variables whose box has (near-)zero width have no interior: pin them
    # and move their contribution into b.
    pinned = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 1e-10)
    if pinned.any():
        xp = np.where(pinned, 0.5 * (lo + hi), 0.0)
        b = b - A @ xp
        keep = ~pinned
        A = sp.csr_matrix(A[:, keep])
        lo_full, hi_full = lo, hi
        c = c[keep]
        lo = lo[keep]
        hi = hi[keep]
    At = sp.csr_matrix(A.T)
    has_lo = np.isfinite(lo)
    has_hi = np.isfinite(hi)

    if start == "mehrotra":
        # Least-squares starting point (Mehrotra's heuristic, box-adapted):
        # min-norm primal consistent with Ax=b, dual from projecting c, and
        # positive shifts sized by how negative the raw slacks/duals are.
        AAt = (A @ A.T).tocsc() + 1e-8 * sp.eye(m, format="csc")
        lu0 = splu(AAt, permc_spec="MMD_AT_PLUS_A")
        x = A.T @ lu0.solve(b)
        y = lu0.solve(A @ c)
        r = c - (A.T @ y)
        zl = np.where(has_lo, np.maximum(r, 0.0), 0.0)
        zu = np.where(has_hi, np.maximum(-r, 0.0), 0.0)

        sl_raw = np.where(has_lo, x - lo, 1.0)
        su_raw = np.where(has_hi, hi - x, 1.0)

        def positive_shift(values: np.ndarray, mask: np.ndarray) -> float:
            if not mask.any():
                return 0.0
            return max(0.0, -1.5 * float(values[mask].min())) + 0.1

        shift_l = positive_shift(sl_raw, has_lo)
        shift_u = positive_shift(su_raw, has_hi)
        width = np.where(has_lo & has_hi, hi - lo, np.inf)
        lo_target = np.where(has_lo, lo + np.minimum(shift_l, 0.4 * width), -np.inf)
        hi_target = np.where(has_hi, hi - np.minimum(shift_u, 0.4 * width), np.inf)
        x = np.clip(x, np.minimum(lo_target, hi_target), np.maximum(lo_target, hi_target))
        # guarantee a strict interior margin even for narrow boxes
        margin = np.where(np.isfinite(width), np.minimum(1e-4, 0.25 * width), 1e-4)
        x = np.where(has_lo, np.maximum(x, lo + margin), x)
        x = np.where(has_hi, np.minimum(x, hi - margin), x)
        z_shift_l = positive_shift(zl, has_lo)
        z_shift_u = positive_shift(zu, has_hi)
        zl = np.where(has_lo, zl + z_shift_l, 0.0)
        zu = np.where(has_hi, zu + z_shift_u, 0.0)
    else:
        # Simple starting point: box interior, duals sized to c.
        x = np.where(
            has_lo & has_hi,
            0.5 * (lo + hi),
            np.where(has_lo, lo + 1.0, np.where(has_hi, hi - 1.0, 0.0)),
        )
        z_mag = np.maximum(1.0, np.abs(c))
        zl = np.where(has_lo, z_mag, 0.0)
        zu = np.where(has_hi, z_mag, 0.0)
        y = np.zeros(m)
    delta_reg = 1e-8 * max(1.0, np.max(np.abs(A.data)))

    bnorm = 1.0 + np.linalg.norm(b)
    cnorm = 1.0 + np.linalg.norm(c)

    n_lo = int(has_lo.sum())
    n_hi = int(has_hi.sum())
    n_comp = max(1, n_lo + n_hi)

    slack_floor = 1e-13
    for it in range(max_iter):
        sl = np.where(has_lo, np.maximum(x - lo, slack_floor), 1.0)
        su = np.where(has_hi, np.maximum(hi - x, slack_floor), 1.0)
        zl = np.where(has_lo, np.maximum(zl, slack_floor), 0.0)
        zu = np.where(has_hi, np.maximum(zu, slack_floor), 0.0)
        # residuals
        rp = b - A @ x
        rd = c - (At @ y) - np.where(has_lo, zl, 0.0) + np.where(has_hi, zu, 0.0)
        mu = (
            np.sum(np.where(has_lo, sl * zl, 0.0)) + np.sum(np.where(has_hi, su * zu, 0.0))
        ) / n_comp
        pres = np.linalg.norm(rp) / bnorm
        dres = np.linalg.norm(rd) / cnorm
        if verbose:
            print(f"  it={it:2d} pres={pres:.2e} dres={dres:.2e} mu={mu:.2e}", flush=True)
        if pres < tol and dres < tol and mu < tol * 10:
            break

        # scaling matrix H = Zl/Sl + Zu/Su (+ regularization for free vars).
        # The regularization shrinks with mu so it stops limiting the final
        # dual accuracy (a fixed delta leaves an O(delta) dual residual).
        delta_it = max(1e-12, min(delta_reg, 1e-2 * mu))
        H = np.where(has_lo, zl / sl, 0.0) + np.where(has_hi, zu / su, 0.0)
        H = H + delta_it  # primal regularization keeps D finite for free vars
        D = 1.0 / H

        ADA = (A @ sp.diags(D) @ At).tocsc() + delta_it * sp.eye(m, format="csc")
        try:
            lu = splu(ADA, permc_spec="MMD_AT_PLUS_A")
        except RuntimeError:
            delta_reg *= 100
            continue

        def solve_newton(rp_, rd_, rcl, rcu, *, sl=sl, su=su, D=D, lu=lu, zl=zl, zu=zu):
            # eliminate dz: dz_l = (rcl - zl*dx)/sl, dz_u = (rcu + zu*dx)/su
            rhs_x = rd_ - np.where(has_lo, rcl / sl, 0.0) + np.where(has_hi, rcu / su, 0.0)
            rhs = rp_ + A @ (D * rhs_x)
            dy = lu.solve(rhs)
            dx = D * ((At @ dy) - rhs_x)
            dzl = np.where(has_lo, (rcl - zl * dx) / sl, 0.0)
            dzu = np.where(has_hi, (rcu + zu * dx) / su, 0.0)
            return dx, dy, dzl, dzu

        # affine (predictor) direction
        rcl_aff = np.where(has_lo, -sl * zl, 0.0)
        rcu_aff = np.where(has_hi, -su * zu, 0.0)
        dx_a, dy_a, dzl_a, dzu_a = solve_newton(rp, rd, rcl_aff, rcu_aff)

        def max_step(v, dv, mask):
            neg = mask & (dv < 0)
            if not neg.any():
                return 1.0
            return min(1.0, np.min(-v[neg] / dv[neg]))

        ap_aff = min(max_step(sl, dx_a, has_lo), max_step(su, -dx_a, has_hi))
        ad_aff = min(max_step(zl, dzl_a, has_lo), max_step(zu, dzu_a, has_hi))

        mu_aff = (
            np.sum(np.where(has_lo, (sl + ap_aff * dx_a) * (zl + ad_aff * dzl_a), 0.0))
            + np.sum(np.where(has_hi, (su - ap_aff * dx_a) * (zu + ad_aff * dzu_a), 0.0))
        ) / n_comp
        sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.1

        # corrector
        rcl = np.where(has_lo, sigma * mu - sl * zl - dx_a * dzl_a * ap_aff * ad_aff, 0.0)
        rcu = np.where(has_hi, sigma * mu - su * zu + dx_a * dzu_a * ap_aff * ad_aff, 0.0)
        dx, dy, dzl, dzu = solve_newton(rp, rd, rcl, rcu)

        ap = 0.995 * min(max_step(sl, dx, has_lo), max_step(su, -dx, has_hi))
        ad = 0.995 * min(max_step(zl, dzl, has_lo), max_step(zu, dzu, has_hi))

        x = x + ap * dx
        y = y + ad * dy
        zl = np.where(has_lo, zl + ad * dzl, 0.0)
        zu = np.where(has_hi, zu + ad * dzu, 0.0)

    if pinned.any():
        x_full = np.where(pinned, 0.5 * (lo_full + hi_full), 0.0)
        x_full[np.where(~pinned)[0]] = x
        x = x_full
    row_scale, col_scale, c_scale = saved_scales
    x = x * col_scale
    y = y * row_scale * c_scale
    return x, y, {"iters": it, "pres": pres, "dres": dres, "mu": mu}


def run(name, load, path, bounds_fn, expected):
    data = load(path)
    bounds = bounds_fn(data)
    lo_l = [float("-inf") if low is None else float(low) for low, _ in bounds]
    hi_l = [float("inf") if up is None else float(up) for _, up in bounds]
    rows, cols = data["A"].shape
    indptr, indices, dat = data["A"].to_components()
    red = presolve_eq_box(
        rows, cols, indptr, indices, dat, data["b"].tolist(), data["c"].tolist(), lo_l, hi_l
    )
    if red is None:
        msg = f"{name}: presolve found nothing to reduce; prototype expects a reduction"
        raise RuntimeError(msg)
    A = sp.csr_matrix((red.data, red.indices, red.indptr), shape=(red.rows, red.cols))
    print(f"== {name}: reduced {A.shape}, nnz {A.nnz} ==", flush=True)
    t0 = time.perf_counter()
    x, y, info = ipm_solve(A, np.array(red.b), np.array(red.c), np.array(red.lo), np.array(red.hi))
    secs = time.perf_counter() - t0
    xf = postsolve_x(np.clip(x, red.lo, red.hi).tolist(), red)
    obj = sum(v * cc for v, cc in zip(xf, data["c"].tolist(), strict=True))
    res = float(np.max(np.abs(data["A_scipy"] @ np.array(xf) - data["b"])))
    print(
        f"{name}: {secs:.2f}s iters={info['iters']} delta={abs(obj - expected):.3e} "
        f"true_res={res:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    run("cycle", load_cycle, CYCLE_PATH, cycle_bounds, EXPECTED_CYCLE_OBJECTIVE)
    run("dfl001", load_dfl001, DFL_PATH, dfl_bounds, EXPECTED_DFL001_OBJECTIVE)
