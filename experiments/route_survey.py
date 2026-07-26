"""Classify all fetched instances by route WITHOUT solving, by replicating the
public route predicates exactly (presolve first, since the predicate runs on the
PRESOLVED model)."""
import sys; sys.path.insert(0,'.')
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from linprogx.presolve import presolve_matrix
from linprogx.sparse import from_scipy_sparse, _ipm_stall_risk
SUITE=Path("/tmp/lpsuite"); INF=float('inf')
rows=[]
for p in sorted(SUITE.glob("lp_*.mat")):
    try:
        raw=loadmat(p)["Problem"][0,0]; aux=raw["aux"][0,0]
        A=raw["A"].tocsc(); b=raw["b"].ravel().astype(float)
        c=aux["c"].ravel().astype(float)
        lo=aux["lo"].ravel().astype(float); hi=aux["hi"].ravel().astype(float)
    except Exception: continue
    try:
        red=presolve_matrix(from_scipy_sparse(A), b.tolist(), c.tolist(),
                            lo.tolist(), hi.tolist(), algorithm="auto")
        if red is None: continue
        M=red._matrix
        pr,pc = (M.shape if M is not None else (red.rows, red.cols))
        pnnz = M.nnz if M is not None else len(red.data)
    except Exception: continue
    # auto route: pdhg if rows > AUTO_IPM_MAX_ROWS else ipm ; then stall-predictor
    from linprogx.sparse import SparseSolver
    thr = SparseSolver.AUTO_IPM_MAX_ROWS
    base = "pdhg" if pr > thr else "ipm"
    route = base
    if base=="ipm" and pr<=4000 and pc<=30000 and _ipm_stall_risk(list(red.c), list(red.lo), list(red.hi), pnnz, pc):
        route = "SIMPLEX"
    rows.append((p.stem, pr, pc, pnnz, pnnz/max(1,pc), route))
rows.sort(key=lambda r:(r[5]!="SIMPLEX", r[0]))
print(f"{'instance':16s} {'prows':>6} {'pcols':>6} {'pnnz':>7} {'cnnz':>6}  route")
for r in rows: print(f"{r[0]:16s} {r[1]:>6} {r[2]:>6} {r[3]:>7} {r[4]:>6.2f}  {r[5]}")
ns=sum(1 for r in rows if r[5]=="SIMPLEX")
print(f"\nSIMPLEX-routed: {ns} / {len(rows)}")
