# Kernel campaign dossier — breaking the conservation law (2026-07-19)

TARGET: greenbea, the board's last loss (~1.69 on-host; local 0.42s vs HiGHS
0.24s). THE LAW TO BREAK (47th settled): pivots x us/pivot ~ 0.38-0.40s for
our kernels across every start: cold 4,399 x 90.5us; foreign 3,529 x 113.1;
native B* 3,334 x 113.4. Flip arithmetic: cold-path needs ~54 us/pivot
(-40%); B*-path needs ~65-72 us/pivot on the DENSE trajectory PLUS an
auxiliary solve under ~0.05s (throwaway construction cost 0.145s).

WALL SPLIT (cold trajectory, local census): pivot-row/PRICE scan 24.8%,
BTRAN 18.9%, FTRAN 17.9%, ratio test 14.9%, reduced-cost update 9.7%,
LU update 6.1%, refactorization 5.5%. B*-trajectory deltas: pivot-row
+7.4us, BTRAN +3.7, FTRAN +2.8 (density-driven).

REGIMES: cold trajectory = solve vectors moderately dense (rho p50 897 of
1,525 rows; alpha p50 3,625 of 3,868 cols). B*-trajectory = dense from
pivot 1 (59-94% densities). Working set is CACHE-RESIDENT (23k nnz problem).

CLOSED (do not re-probe as-is; distinctions required):
- fp32 rounding of solve/pricing VECTORS: trajectory diverges at pivot 117;
  kernel container gains only 0.98-1.18x (cache-resident). [41st]
- dense-U sweep for the U-SOLVE only: ~0-1% in all regimes. [25th/47th]
- p=4 pivot panels/batching: survival 1.28 pivots -> 0.59x. [42nd]
- symmetric permutation for contiguity: accesses already 96% cache-line
  reusing; oracle ceiling 1.26%. [45th]
- Schur/bordered factorization: rho density is factorization-INVARIANT;
  bases are one giant component. [43rd]
- Native dual Phase-1 via the derived auxiliary: B* verified dual-feasible,
  3,334 pivots, but trajectory densifies to 113.4us and the pipeline loses
  to cold. [46th/47th]

THE UNTESTED SEAM: none of the floor proofs measured INSTRUCTION THROUGHPUT.
The hot loops are sparse-style C (index load -> dependent load -> branch):
plausibly low IPC, defeated auto-vectorization, per-element branching. On
dense-regime vectors, branchless contiguous SIMD may win on instructions
while touching the same cache-resident data. Also untested: threading
crossovers for the scan, fp32 COMPARISONS (not values), scan+update fusion,
compiler-flag/annotation headroom, auxiliary-solve cost engineering, basis
QUALITY refinement, and density-shaped hybrid starts.

CODE MAP: src/linprogx/_csparse.c — CSRMatrix_solve_eq_box_dual_simplex
(DS loop), lu_ftran_sparse/lu_btran_sparse/lu_ft_ftran/lu_ft_btran (solves),
pivot-row/PRICE and ratio-test inline in the loop, LINPROGX_DS_SOLVE_SLICE
(slice timers), leaving_rule=5 (exact DSE), the basis-injection warm-start
hook (see experiments/greenbea_basis_transfer_probe.py), the derived
auxiliary + B* construction (experiments/phase1_predictions_2026_07_18.md,
worktree scripts referenced there). Fixtures /tmp/lpsuite.

RULES (every angle): no network whatsoever (logs audited; violations =
discard). Never read/fetch any solver's source. No per-problem tuning
(global mechanisms only). eps=2e-5 fixed; certificate-backed optimality;
byte-identical or trajectory-identical off-paths for any knob; honest
falsifier-first reporting with the kill criterion stated. Build:
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation, then
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e .
--no-build-isolation after every C change. No git ops. Foreground.

DELIVERABLE (every angle): experiments/k<N>_<slug>_2026_07_19.md with
measurements, the projection arithmetic against the flip targets above, and
verdict LIVE/KILLED. Final message: verdict + key numbers + path.
