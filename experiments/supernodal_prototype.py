"""Left-looking supernodal Cholesky prototype (validation before C port).

The dense-tail factor pays t^3/3 flops because it treats the sparse
upper region of the tail as dense. A supernodal factor does dense BLAS
panels over only the true fill, recovering both BLAS speed and the
sparse flop count (maros_r7: 5.7e8 true vs 6.5e9 dense-block).

This prototype validates the supernodal arithmetic and confirms the flop
count against the true fill, using numpy dense panels. No public solver
source consulted; this is the textbook left-looking supernodal algorithm
(Davis, "Direct Methods for Sparse Linear Systems";
this is implemented from the textbook description).

Usage: PYTHONPATH=. uv run python experiments/supernodal_prototype.py lp_maros_r7
"""

import sys
import time
from importlib import import_module
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

_csparse = cast(Any, import_module("linprogx._csparse"))


def load_Gp(name):
    raw = loadmat(f"/tmp/lpsuite/{name}.mat")["Problem"][0, 0]
    A = raw["A"].tocsr().astype(float)
    m = A.shape[0]
    G = (A @ A.T).tocsc()
    G = G + sp.identity(m) * 1e-6
    perm = np.array(_csparse.min_degree(G.indptr.tolist(), G.indices.tolist()))
    return G[perm][:, perm].tocsc(), m


def etree(Gp):
    m = Gp.shape[0]
    parent = np.full(m, -1)
    ancestor = np.full(m, -1)
    indptr, indices = Gp.indptr, Gp.indices
    for k in range(m):
        for p in range(indptr[k], indptr[k + 1]):
            i = indices[p]
            while i != -1 and i < k:
                nxt = ancestor[i]
                ancestor[i] = k
                if nxt == -1:
                    parent[i] = k
                i = nxt
    return parent


def column_row_sets(Gp, parent):
    """Row index set of each L column (rows >= j with a nonzero), via the
    reach of Gp's column patterns up the elimination tree."""
    m = Gp.shape[0]
    indptr, indices = Gp.indptr, Gp.indices
    below = [set() for _ in range(m)]
    for k in range(m):
        reach = set()
        for p in range(indptr[k], indptr[k + 1]):
            i = indices[p]
            while i < k and i not in reach:
                reach.add(i)
                i = parent[i]
        for i in reach:
            below[i].add(k)  # column i has a nonzero in row k (k > i)
        below[k].add(k)
    return [np.array(sorted(s), dtype=np.int64) for s in below]


def supernode_partition(parent, colcount):
    m = len(parent)
    nchild = np.zeros(m, dtype=int)
    for j in range(m):
        if parent[j] >= 0:
            nchild[parent[j]] += 1
    starts = []
    for j in range(m):
        if j > 0 and parent[j - 1] == j and nchild[j] == 1 and colcount[j - 1] == colcount[j] + 1:
            continue
        starts.append(j)
    starts.append(m)
    return np.array(starts, dtype=np.int64)


def supernodal_chol(Gp, snode_start, colrows):
    """Left-looking supernodal numeric factorization. Returns L as a dense
    m x m lower-triangular array (prototype storage) and the flop count."""
    m = Gp.shape[0]
    Gd = Gp.toarray()
    Gd = np.tril(Gd) + np.tril(Gd, -1).T  # symmetrize (Gp stored lower)
    L = np.zeros((m, m))
    ns = len(snode_start) - 1
    col_snode = np.zeros(m, dtype=int)
    for s in range(ns):
        col_snode[snode_start[s] : snode_start[s + 1]] = s
    # each supernode's full row list (its own columns + the off-diagonal
    # rows), taken from the first column's row set
    snode_rows = []
    for s in range(ns):
        j0 = snode_start[s]
        snode_rows.append(colrows[j0])  # rows >= j0 (incl. the snode cols)
    flops = 0.0
    for s in range(ns):
        j0, j1 = snode_start[s], snode_start[s + 1]
        w = j1 - j0
        rows = snode_rows[s]  # sorted, rows[:w] == [j0..j1)
        nr = len(rows)
        # dense frontal panel F: nr x w, initialized from G
        F = Gd[np.ix_(rows, np.arange(j0, j1))].copy()
        # subtract contributions from descendant supernodes K < s whose
        # row set intersects [j0, j1)
        for k in range(s):
            kr = snode_rows[k]
            k0, k1 = snode_start[k], snode_start[k + 1]
            wk = k1 - k0
            off = kr[wk:]  # off-diagonal rows of supernode k
            if len(off) == 0:
                continue
            in_J = (off >= j0) & (off < j1)
            if not in_J.any():
                continue
            # columns of the update = off rows landing in J (relative to j0)
            upd_cols = off[in_J] - j0
            # rows of the update = off rows landing in J ∪ R_J
            rowset = set(int(r) for r in rows)
            in_target = np.array([int(r) in rowset for r in off])
            if not in_target.any():
                continue
            Loff = L[np.ix_(off, np.arange(k0, k1))]  # |off| x wk
            A_blk = Loff[in_J, :]  # (pivot rows) x wk
            B_blk = Loff[in_target, :]  # (target rows) x wk
            update = B_blk @ A_blk.T  # (target) x (pivot)
            flops += B_blk.shape[0] * A_blk.shape[0] * wk
            # scatter-subtract into F
            tgt_rows = off[in_target]
            row_pos = np.searchsorted(rows, tgt_rows)
            F[np.ix_(row_pos, upd_cols)] -= update
        # factor the w x w diagonal block, then the panel
        D = F[:w, :]
        Ldiag = np.linalg.cholesky(D)
        flops += w**3 / 3.0
        F[:w, :] = Ldiag
        if nr > w:
            # solve L_panel = F_off @ Ldiag^{-T}
            F[w:, :] = np.linalg.solve(Ldiag, F[w:, :].T).T
            flops += (nr - w) * w * w
        # scatter F into L
        L[np.ix_(rows, np.arange(j0, j1))] = F
    return L, flops


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "lp_maros_r7"
    Gp, m = load_Gp(name)
    par = etree(Gp)
    colrows = column_row_sets(Gp, par)
    colcount = np.array([len(r) for r in colrows])
    snode_start = supernode_partition(par, colcount)
    ns = len(snode_start) - 1
    print(
        f"{name}: m={m} supernodes={ns} true_fill_flops={np.sum(colcount.astype(float) ** 2):.2e}"
    )
    t0 = time.perf_counter()
    L, flops = supernodal_chol(Gp, snode_start, colrows)
    t_factor = time.perf_counter() - t0
    # validate: L L^T == Gp (symmetrized)
    Gd = Gp.toarray()
    Gd = np.tril(Gd) + np.tril(Gd, -1).T
    resid = np.max(np.abs(L @ L.T - Gd))
    print(
        f"  supernodal flops={flops:.2e} factor_residual={resid:.2e} "
        f"(prototype wall {t_factor:.1f}s)"
    )
    print(f"  vs dense-block flops would be (m^3/3)={m**3 / 3.0:.2e}")


if __name__ == "__main__":
    main()
