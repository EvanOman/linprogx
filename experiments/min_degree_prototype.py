"""Exact minimum-degree ordering prototype.

Quotient-graph formulation with element absorption (no supervariables, no
approximate degrees — exact degree recomputation with set unions). Validates
fill quality against SuperLU's MMD_AT_PLUS_A on the benchmark normal-equation
matrices before the C port.

Run: PYTHONPATH=. uv run python experiments/min_degree_prototype.py
"""

from __future__ import annotations

import heapq
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from bench_cycle import DATA_PATH as CYCLE_PATH
from bench_cycle import _bounds as cycle_bounds
from bench_cycle import load_cycle
from bench_large import DATA_PATH as DFL_PATH
from bench_large import _bounds as dfl_bounds
from bench_large import load_dfl001
from linprogx.presolve import presolve_eq_box


def min_degree_order(ADA: sp.csc_matrix) -> list[int]:
    m = ADA.shape[0]
    indptr, indices = ADA.indptr, ADA.indices
    adj: list[set[int]] = [set() for _ in range(m)]
    for j in range(m):
        for idx in range(indptr[j], indptr[j + 1]):
            i = int(indices[idx])
            if i != j:
                adj[i].add(j)
                adj[j].add(i)

    elements: list[set[int]] = []
    var_elements: list[set[int]] = [set() for _ in range(m)]
    alive = np.ones(m, dtype=bool)
    degree = [len(adj[v]) for v in range(m)]
    heap = [(degree[v], v) for v in range(m)]
    heapq.heapify(heap)
    order: list[int] = []

    while len(order) < m:
        d, v = heapq.heappop(heap)
        if not alive[v] or d != degree[v]:
            continue
        nbhd: set[int] = {u for u in adj[v] if alive[u]}
        for e in var_elements[v]:
            nbhd |= elements[e]
        nbhd.discard(v)

        order.append(v)
        alive[v] = False

        eid = len(elements)
        elements.append(nbhd)
        absorbed = var_elements[v]
        for e in absorbed:
            elements[e] = set()
        for u in nbhd:
            adj[u].discard(v)
            # adjacency covered by the new element is redundant
            adj[u] -= nbhd
            var_elements[u] -= absorbed
            var_elements[u].add(eid)
        for e_set in elements[:eid]:
            e_set.discard(v)
        for u in nbhd:
            s = {w for w in adj[u] if alive[w]}
            for e in var_elements[u]:
                s |= elements[e]
            s.discard(u)
            degree[u] = len(s)
            heapq.heappush(heap, (degree[u], u))
    return order


def fill_of(ADA: sp.csc_matrix, perm: list[int] | np.ndarray) -> int:
    P = ADA[perm][:, perm].tocsc()
    lu = splu(P, permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    return int(lu.L.nnz)


def normal_matrix(name, load, path, bounds_fn) -> sp.csc_matrix:
    data = load(path)
    bounds = bounds_fn(data)
    lo = [float("-inf") if low is None else float(low) for low, _ in bounds]
    hi = [float("inf") if up is None else float(up) for _, up in bounds]
    rows, cols = data["A"].shape
    indptr, indices, dat = data["A"].to_components()
    red = presolve_eq_box(
        rows, cols, indptr, indices, dat, data["b"].tolist(), data["c"].tolist(), lo, hi
    )
    assert red is not None
    A = sp.csr_matrix((red.data, red.indices, red.indptr), shape=(red.rows, red.cols))
    return (A @ A.T).tocsc() + 1e-8 * sp.eye(red.rows, format="csc")


if __name__ == "__main__":
    for name, load, path, bounds_fn in [
        ("cycle", load_cycle, CYCLE_PATH, cycle_bounds),
        ("dfl001", load_dfl001, DFL_PATH, dfl_bounds),
    ]:
        ADA = normal_matrix(name, load, path, bounds_fn)
        lu = splu(
            ADA, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)
        )
        print(f"{name}: m={ADA.shape[0]} nnz={ADA.nnz} | MMD L nnz = {lu.L.nnz}", flush=True)
        t0 = time.perf_counter()
        order = min_degree_order(ADA)
        secs = time.perf_counter() - t0
        assert sorted(order) == list(range(ADA.shape[0]))
        print(f"  exact MD: order time {secs:.2f}s, L nnz = {fill_of(ADA, order)}", flush=True)
