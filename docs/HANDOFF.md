# linprogx Handoff — Performance Work

**Branch:** `perf-supernodal-simplex` (based on `main`)
**Last commit:** `cad2eb7` Validate left-looking supernodal factorization (Python prototype)
**Full history/rationale:** `docs/sparse-pdhg-performance-handoff.md` (Updates 1–22)

## Goal

Make linprogx — an independently built LP solver — **exceed both Clarabel
and HiGHS** on coverage and runtime across the LPnetlib suite.

Hard constraints (do not violate):
- **Never read public solver source code.** Papers / textbooks / literature OK.
- **No per-problem tuning.** Global thresholds calibrated to measured machine
  throughput are fine.
- **Never loosen `eps=2e-5`.** Every reported optimum must be certificate-backed.
- **Tests + end-to-end checks alongside every change.** Keep `just ci` green.
- BLAS/LAPACK deps are allowed (links OpenBLAS already).

## Current standing (measured)

- Coverage tied 23/24 (HiGHS, Clarabel, linprogx each miss one: qap15 /
  ken_18 / greenbea respectively).
- **Exceeds Clarabel: beats it on 19/23 solved instances.** Done.
- **Does NOT yet exceed HiGHS.** Remaining gaps and their true causes:
  1. **maros_r7** ~3.9s vs HiGHS 0.95s — factor-bound. The dense-tail
     factor pays t³/3 flops (one big dense block); the true sparse fill is
     ~13–21× smaller. **→ supernodal numeric factor.**
  2. **pds_20** 44s (PDHG) vs HiGHS 12.5s — normal equations are
     fill-explosive (direct factor infeasible; Clarabel also loses here at
     115s). **→ needs a sparse simplex.**
  3. **~8 sub-second instances** (woodw, 80bau3b, …) — KERNEL-bound
     (IPM factors every iteration vs HiGHS simplex). NOT driver-bound:
     measured Python overhead is only 3–20% (~6–21ms), so a native driver
     is the wrong lever and is shelved.

## Engine map (where things live)

All numeric kernels: `src/linprogx/_csparse.c` (~4600 lines, C extension).
Python orchestration/routing: `src/linprogx/sparse.py` (`SparseSolver`).
- IPM: `CSRMatrix_solve_eq_box_ipm` (Mehrotra; kwargs `max_iter, tol, debug,
  threads, blas`). Exit chain: relaxed acceptance → dual polish →
  `ipm_dual_cleanup` (min-norm dual repair) → certificate gate.
- Sparse Cholesky: `chol_setup` (AMD order, etree, symbolic, supernode
  partition, dense-tail cost-model split) + `chol_refactor` (numeric).
- Dense tail: OpenBLAS `dpotrf` for tail ≥ 400, with a 1e-11 diagonal
  ridge emulating the per-pivot floor (lets degenerate `cre` certify);
  hand kernel `tail_dense_chol` is the fallback.
- PDHG: `CSRMatrix_solve_eq_box_pdhg` (restarted adaptive; `threads` kwarg,
  bit-identical across thread counts).
- Routing (`sparse.py` `_solve_eq_box`): presolve → rows ≤ 50k IPM else
  PDHG; on IPM non-certification retries floored (`blas=False`) on the
  presolved then unpresolved problem.

Bench harness: `experiments/suite_bench.py --worker <FILE> <solver>`.
Suite data: `/tmp/lpsuite/*.mat` (download: `experiments/download_lpsuite.sh`).
Published optima: `/tmp/netlib_optima.csv`. Scoreboard:
`assets/lpnetlib_suite.md` + `assets/lpnetlib_suite_results.jsonl`.

## NEXT STEP — port the supernodal factor to C (highest priority)

The Python prototype is **validated** (`experiments/supernodal_prototype.py`):
left-looking supernodal Cholesky, residual 3.5e-15 on maros_r7, flops
4.87e8 vs dense-block 1.03e10. Supernode DETECTION is already in C
(`chol_setup` builds `ctx->snode_start`/`ctx->n_snodes`; `supernode_sizes()`
test hook; matches the reference exactly on maros_r7).

Port the prototype's `supernodal_chol` into `chol_refactor`, replacing the
single dense-tail block with per-supernode dense panels:
1. Precompute (in `chol_setup`) each supernode's row structure and the
   per-supernode descendant-update lists + relative-index maps (the
   intricate part — do it once, symbolically, like `pair_offset`).
2. In `chol_refactor`, for each supernode: assemble its dense panel from
   `Cx`, subtract descendant updates (`dgemm`/outer products), `dpotrf`
   the diagonal block, `dtrsm` the off-diagonal panel; scatter into `Lx`.
3. Keep results in the existing CSC `Lx` so `chol_solve`/SMW/matvec are
   untouched. Validate by residual vs the current factor on the
   `normal_equations_solve` test hook, then run the IPM end-to-end behind
   the certificate gate.
Expected: maros_r7 refactor 2.69s → ~0.2–0.5s, total → ~1.5–2s. NOTE: this
still does not beat HiGHS 0.95s alone — the simplex (below) is also needed
to fully close maros_r7 and the small instances.

## SECOND LEVER — sparse revised simplex (larger, multi-session)

Attacks pds_20 (fill-explosive, IPM can't factor) and the small
simplex-favorable instances. Prior probes in `experiments/`:
`revised_simplex_prototype.py`, `dual_simplex_prototype.py`,
`tapia_probe.py` — key finding: warm-starting from an IPM/PDHG point fails
(degenerate basis identification is the bottleneck). A competitive build is
likely a cold bounded-variable revised simplex with proper basis-factor
updates (product-form / LU), reusing the C Cholesky machinery only where
applicable. Big effort; sequence after the supernodal factor lands.

## Workflow notes

- Build after C edits: `uv pip install -e . --force-reinstall --no-cache`
  then check the `.so` mtime; assert build had 0 `error:` lines (stale-.so
  has bitten this project repeatedly — verify the binary actually rebuilt).
- Gate every commit on `just ci` (ruff lint + format + ty + pytest). 139
  tests currently pass.
- Measure on a QUIET machine (`uptime`); the shared box hits loadavg ~12
  and inflates all timings ~1.5×. Use back-to-back paired measurements for
  before/after; only record official scoreboard numbers from a quiet run.
- When a degenerate instance stops certifying after a factor change, it is
  trajectory sensitivity — the floored retry (`blas=False`) is the safety
  net; check `best_gap` via `debug=True` before assuming a real failure.
- This repo runs an autonomous `/loop` with a Stop-hook enforcing the goal.
  Re-arm the loop after each iteration; record official suite runs into
  `assets/` + the handoff doc; rebase past the coverage-badge bot when
  pushing to `main`.
