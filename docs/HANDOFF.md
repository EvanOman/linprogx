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
  1. **maros_r7** ~3.0s vs HiGHS 0.94s — factor-bound. The dense-tail
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

### Update 2026-06-20 — C supernodal path landed, auto-gated

Implemented the first C port of the left-looking supernodal numeric factor in
`src/linprogx/_csparse.c`:

- `chol_setup` now computes the supernode partition; supernode row lists,
  panel `Lx`/`Cx` offsets, descendant update lists, and relative pivot/target
  maps are built lazily on the first supernodal refactor or symbolic test hook.
- `chol_refactor_supernodal` factors dense panels into the existing CSC `Lx`,
  using OpenBLAS `dpotrf`/`dgemm`/`dtrsm` for large panel work and scalar
  loops for small updates.
- `solve_eq_box_ipm(..., supernodal=...)` accepts an experimental override.
  The default is auto-gated by global structure (`tail_len >= 512` and mean
  supernode width >= 4), which selects supernodal on `maros_r7` but avoids the
  fragmented-supernode regressions seen on `pilot87` and `cre_b`.

Validation added:

- `supernode_symbolic_structure()` test hook matches an independent Python
  reference for row/update maps.
- `normal_equations_solve(..., True)` residual test covers the new numeric
  factor directly.
- `solve_eq_box_ipm(..., supernodal=True)` smoke test covers the IPM entry.
- `just ci` passes after the raw-feasibility and lazy-symbolic updates:
  143 tests, coverage 86.85%.

Measured on a loaded machine, so treat timings as directional:

- `experiments/suite_bench.py --worker /tmp/lpsuite/lp_maros_r7.mat linprogx`
  reports optimal, 18 iterations, residual 2.9e-11, 3.54s.
- Direct `maros_r7` IPM default/old/super comparisons showed the auto route
  selecting supernodal and generally beating forced old row-wise factorization.
- Forced supernodal is still bad for fragmented cases: `pilot87` and `cre_b`
  were ~2-3x slower, so do not remove the structural gate without new evidence.

Follow-up in the same session:

- Added `LINPROGX_SUPERNODAL_PROFILE=1` instrumentation for the supernodal
  refactor. On `maros_r7`, one forced-supernodal normal-equations solve spent
  roughly 0.02s assembly, 0.02s BLAS update, 0.02s scalar update, and smaller
  slices in panel gather/scatter/chol/trsm. The single refactor is no longer
  the only lever; iteration count matters.
- Moved the existing certificate-producing dual polish into the IPM loop after
  residuals are small, with a size gate so tiny high-accuracy problems still
  run to strict tolerance. Min-norm dual cleanup is only attempted in-loop when
  residuals are small and the last step length is effectively stalled, avoiding
  cleanup on clean problems.
- Follow-up 2026-06-30: the first in-loop polish accepted `maros_r7` with a
  certificate gap but raw equality residual above the public `eps=2e-5`
  contract. The IPM now threads an original-unit `feas_tol` through strict,
  relaxed, polish, and cleanup exits, and the Python public wrapper recomputes
  the postsolved residual before reporting `optimal`.
- Current loaded-machine public worker after the raw-feasibility guard:
  `experiments/suite_bench.py --worker /tmp/lpsuite/lp_maros_r7.mat linprogx`
  reports optimal, 17 iterations, residual `3.19e-9`, and 3.91s. A paired
  HiGHS run on the same machine was 0.90s, so this is sound again but still
  not competitive with HiGHS.
- The full supernodal symbolic maps are now deferred for row-wise cases. With
  `LINPROGX_CHOL_DEBUG=1`, the old setup-only map phase was about 0.10s on
  `pilot87` and 0.39s on `cre_b`; after deferral the setup phase is effectively
  zero unless the supernodal numeric path is actually selected.
- Follow-up 2026-06-30: measured and rejected precomputing direct `Lx` offset
  arrays for every update row. It added memory traffic and did not reduce
  gather time on `maros_r7`. Kept a narrower fast path: when a BLAS update's
  pivot columns are contiguous, scatter subtracts a contiguous row slice
  instead of looking up each pivot column. On the same forced-supernodal
  normal-equations probe, `update_scatter` moved from roughly 0.02s to
  0.012-0.014s per refactor. Added a width-one scalar-update specialization
  after measuring that ~78% of scalar-update work on `maros_r7` has
  source-supernode width 1; `scalar_update` moved from roughly 0.023-0.025s to
  0.013-0.016s per refactor. Overall public-worker timings remain too noisy
  on the loaded shared box for a headline number.
- Follow-up 2026-06-30: added a guarded primal feasibility polish for
  certificate-ready IPM iterates whose scaled residual and dual gap are already
  good but whose original-unit equality residual is just above `eps`. It solves
  one slack-weighted normal-equation correction, then accepts only if original
  residual, bound violation, and Lagrangian gap all pass. This makes
  `cre_a` with public `eps=1e-9` stay on IPM instead of falling through to
  PDHG, and lets `maros_r7` certify at 15 iterations with residual
  `2.9e-11` instead of waiting until iteration 17.
- Follow-up 2026-07-01: exposed a public `SparseSolver(..., threads=...)`
  knob and changed the default PDHG route to use the existing deterministic
  4-thread kernel. On the loaded shared box, public `pds_20` improved from
  roughly 53.6s in the sequential worker probe to 31.4-32.6s, with the same
  21,696 iterations and residual `1.8e-5`. Explicit PDHG thread probing on
  the presolved `pds_20` system showed 4 threads faster than 1/2 threads;
  `threads=0` was noisy and slower in that run.
- Same follow-up: tested raising the public PDHG default to 8 threads and
  rejected it. Direct presolved C probes looked promising on `pds_20`
  (roughly 31.1s at 4 threads vs 27.9-29.4s at 8) and slightly positive on
  `qap15`, but high-level paired `SparseSolver(algorithm="auto")` runs on
  `pds_20` were 33.17s/30.15s at 4 threads vs 33.08s/33.76s at 8. The
  explicit `threads=` knob remains; the default stays 4 until the public route
  shows a stable win.
- Follow-up 2026-07-01: fixed the persistent PDHG thread pool so the explicit
  `threads=` knob can grow after an earlier smaller solve and so the profile
  reports both active `threads` and created `pool_threads`. Before the fix, a
  process that first solved with `threads=2` would silently cap a later
  `threads=4` solve at two workers. Re-ran public `pds_20` in one process with
  the repaired pool and `LINPROGX_PDHG_PROFILE=1`: 4 threads reported
  `threads=4 pool_threads=4` and solved in `34.5s`; 8 threads grew to
  `threads=8 pool_threads=8` but slowed to `49.6s`, with the same 21,696
  iterations, objective, and residual. Keep the default at 4.
- Same follow-up: added a PDHG early CGLS cleanup exit only for systems with
  at most 5,000 rows, and only when primal residual is within `10 * tol` and
  the normal gap/dual tests already pass. A looser large-system cleanup probe
  was measured and rejected: it reduced `pds_20` iterations but made wall time
  worse by turning cleanup into a second long solve.
- Same follow-up: a small `pds_20` tuning grid did not find a better global
  PDHG setting. Rejected probes: `objective_scale=4`, `objective_scale=8`,
  `adaptive_weight=2`, `restart_artificial=0.25`,
  `eval_interval_override=128`, and `eval_interval_override=32`; all were
  slower or failed the solve budget. The current best pds lever in this tree
  is still the 4-thread public route, not restart tuning.
- Same follow-up: added `LINPROGX_PDHG_PROFILE=1` timing for the native PDHG
  loop after external `perf` was blocked by `perf_event_paranoid=4`. On
  loaded-machine `pds_20`, the profile was dominated by loop work:
  `step ~23.4s`, transpose matvec `~5.2s`, accepted-iterate accumulation
  `~3.1s`, eval `~0.9s`, and no cleanup time. Fusing the accepted
  `x_sum += x` pass into the transpose matvec removed a full column sweep per
  iteration without changing the algorithm. The immediate back-to-back
  profile moved `accumulate` from `3.07s` to `0.72s` and total C time from
  `32.64s` to `29.55s`, with the same 21,696 iterations, objective, and
  residual. This is a real local throughput win, but still not enough to close
  the `pds_20` gap to HiGHS.
- Same follow-up: measured and rejected fusing the remaining `y_sum += y`
  average accumulation into the accepted trial's row-reduction pass. It
  removed the standalone `accumulate` slice, but the canonical row-reduction
  pass became more memory-bound (`trial_row_reduce` roughly `0.68s` before vs
  `1.22-1.34s` after on `pds_20`), so the total run only moved within load
  noise (`25.27s` before vs `25.08s` and `23.68s` after). Keep the simpler
  standalone pass unless a different design avoids the extra y-sum buffer
  traffic.
- Same follow-up: implemented a targeted BLAS thread split for the C
  supernodal path. The monolithic dense-tail factor keeps its existing
  4-thread OpenBLAS policy, but supernodal panel `dpotrf`/`dgemm`/`dtrsm`
  now force one OpenBLAS thread because that path issues many small BLAS
  calls. A controlled in-process probe that first triggered the extension's
  one-time BLAS init, then alternated OpenBLAS threads on public `maros_r7`,
  measured `1` thread at `2.66s` and `2.64s` vs `4` threads at `3.00s` and
  `2.73s`, all with residual `2.9e-11`. Public worker with
  `LINPROGX_SUPERNODAL_PROFILE=1 LINPROGX_CHOL_DEBUG=1` now reports
  `blas_threads=1`, optimal in `3.03s`, residual `2.9e-11`, 15 iterations;
  paired HiGHS on the same loaded box was `0.94s`, so this narrows but does
  not close the gap.
- Same follow-up: checked two tempting shortcuts and rejected both. First,
  changing the process-wide/default OpenBLAS policy was not stable enough for
  a global default change; keep the split policy above instead of applying
  one thread to the dense-tail route. Second, the existing
  dependency-free sparse tableau simplex is not a small-instance routing
  shortcut: `woodw` solved through auto/IPM in `0.27s`, while
  `algorithm="simplex"` failed to finish before a 30s timeout; `80bau3b`
  auto/IPM solved in `0.48s` before the wrapper timed out in the simplex
  leg. Do not route LPnetlib-scale equality/bounds fixtures to the tableau
  simplex.
- Follow-up 2026-07-01: investigated the `greenbea` coverage miss. Direct
  IPM on the presolved system reaches raw feasibility (`~2.3e-10`) in about
  0.6-0.8s but stalls with 112 lower-only columns still carrying negative
  reduced costs, so certification would be false; HiGHS objective is about
  `-7.2555e7` and the stalled IPM primal is about `-7.2462e7`. The public
  auto route now keeps this feasible IPM candidate when fallback PDHG also
  fails instead of returning the PDHG point with residual `~215`. Loaded
  worker timing on `greenbea` moved from ~6.1s PDHG iteration-limit with huge
  residual to ~2.8s IPM iteration-limit with residual `2.3e-10`.
- Same follow-up: rejected two quick `greenbea` crossover probes. The existing
  Tapia/revised-simplex prototype on the raw problem chose 472 artificial
  columns and hit a singular basis after 4 pivots. A throwaway primal-descent
  LSQR direction probe from the presolved IPM point preserved feasibility but
  improved the objective by only ~0.006 over 20 steps, far from the ~93k
  objective gap. A real revised simplex still needs a proper basis factor and
  pivot policy, not a min-norm descent heuristic.
- Same follow-up: fresh `greenbea` debug confirmed the raw problem exits at
  iter 60 with raw residual `~106`, while the presolved path reaches raw
  residual `2.3e-10` at iter 61, then the next Newton step goes NaN. The
  feasible iterate is still not certifiable: dual residual stalls at
  `1.839e-6`, objective is `-7.2462e7` vs HiGHS `-7.2555e7`, and a Python
  reproduction of the current min-norm dual cleanup starts with 112 bad
  reduced costs but expands the union to 2,799 columns and still has 1,463
  bad signs. More dual cleanup is the wrong lever for `greenbea`; it needs a
  genuine primal/basis improvement.

- Follow-up 2026-07-01 (later session): implemented relaxed supernode
  amalgamation (Ashcraft/Grimes-style) in `chol_setup`. Adjacent fundamental
  supernodes are merged greedily when the elimination tree links them
  (`parent[last col of left] == first col of right`) and the merged panel
  carries at most `LINPROGX_SNODE_RELAX` (default 0.15) structural zeros.
  Padding positions are exact zeros of L — every update product into them
  has a structurally-zero factor — so the numeric factor is unchanged; they
  alias a sentinel zero slot appended to `Lx` so the hot gather/scatter
  loops stay branch-free. The union row list of a merged group is its
  columns followed by the below-diagonal structure of its last column
  (valid because `parent[j] == j+1` holds inside every group). Measured on
  `maros_r7` (loadavg ~5, paired A/B via the env knob): fundamental
  partition 316 supernodes (mean 9.9) -> 167 (mean 18.8); per-refactor
  scalar updates 4955 -> 1052, BLAS updates 211 -> 109; profile per-refactor
  cost ~0.11s -> ~0.065s; public worker 2.36-2.59s (relax=0, old partition)
  vs 2.06-2.24s (relax=0.15), same 15 iterations, objective identical to
  13 digits, residual 3.3e-11 vs 2.9e-11. A 0.25 setting tied 0.15, so the
  default stays at the more conservative 0.15. Fragmented cases stay on the
  row-wise path (post-merge mean widths pilot87 2.6, cre_b 1.6, cre_a 1.8,
  all below the auto-gate's 4.0), with statuses/iterations/residuals
  unchanged and timings within load noise. Eight-instance sanity sweep
  (woodw, 80bau3b, d2q06c, stocfor3, degen3, fit2p, ken_11, truss) all
  optimal with objectives matching published optima at the usual
  tolerances. The Python symbolic reference in `tests/test_ipm.py` mirrors
  the same merge rule; new tests cover solve correctness at relax 0/0.15/0.4
  and partition coarsening on `cre_a`. `just ci` green: 152 tests, 86.56%.

Next C-factor work: with amalgamation in, the supernodal refactor cost on
`maros_r7` is roughly halved but total time (~2.1s) is still dominated by
15 IPM iterations x (refactor + triangular solves); HiGHS is ~0.94s on the
same box. Remaining levers: fewer IPM iterations (better centering /
predictor quality), a cheaper triangular-solve path over the supernodal
panels, or the sparse revised simplex lever for `pds_20` and small
simplex-favorable cases.

- Follow-up 2026-07-01 (same later session): fresh QUIET-box official
  3-solver suite run (results in the session scratchpad, not yet promoted
  to `assets/`) at the amalgamation commit showed broad improvement over
  the stale scoreboard: maros_r7 1.98s (HiGHS 0.74s), ken_18 13.2s (8.1s),
  pds_20 15.4s (10.3s), cre_b 6.1s (1.9s), ken_13 0.66s (0.84s, now a
  win), osa_30 3.21s (3.79s, win), d2q06c/fit2p/qap12/qap15/truss still
  wins — but `osa_60` regressed from the old scoreboard's 18.95s to
  TIMEOUT. Diagnosis: the old 18.95s "optimal" carried raw residual 8e-3,
  an eps-contract violation the 2026-06-30 raw-feasibility guard now
  correctly rejects; the honest path then burned the 180s budget in
  retries. The presolved IPM run actually reaches pres=2.4e-8 /
  dres=3.1e-11 but goes NaN one iteration after raw enters the polish
  window, with raw 8.6e-3 stuck above eps due to row scaling.
- Fix (commit after the amalgamation one): (1) primal-polish raw window
  1e-3 -> 1e-1 (it is only a cost pre-filter; the polish self-checks bound
  violation, recomputed raw residual, and Lagrangian gap), (2) min-norm
  dual cleanup attempted in that window on the 16-iteration rate limit
  without requiring a step stall, (3) the same certified-dual + guarded
  primal-polish chain retried once on non-optimal exit from the restored
  best iterate (residual/aty recomputed; the possibly NaN-poisoned last
  factor is bypassed because the polish refactors with its own slack
  weights). Public `osa_60`: optimal, 15.4s, residual 7.5e-9, 51
  iterations — now BEATS HiGHS's 19.1s and is certificate-backed.
- Same commit: Gondzio multiple centrality correctors in the IPM loop —
  up to `LINPROGX_IPM_MCC` (default 2) recentering back-solves per
  iteration with zero primal/dual rhs, targets projected into the
  [0.1, 10] * sigma * mu hypercube from a 0.08-extended trial step,
  accepted only when both step lengths hold and their sum grows by 0.01+,
  skipped once ap, ad >= 0.9. Iterations: maros_r7 15->12, pilot87
  150->128, cre_b 71->62, cre_a 36->31. Loaded-box 12-instance sweep all
  optimal with objectives at published optima within certificate
  tolerance. `test_gondzio_correctors_preserve_solution_and_save_iterations`
  pins equal-objective/fewer-iterations on the cre_a fixture.
- Re-probed supernodal OpenBLAS threads after amalgamation widened the
  panels (largest 705 cols): 1 thread still fastest on maros_r7 (2.33s vs
  2.46s/2.41s at 2/4). Default stays 1;
  `LINPROGX_SUPERNODAL_BLAS_THREADS` exists for future re-probes.
- NEXT: re-run the official 3-solver suite on a QUIET box with the
  corrector+polish build and promote it to `assets/lpnetlib_suite.md` +
  `assets/lpnetlib_suite_results.jsonl` (the current assets scoreboard
  predates the supernodal work entirely). Standing after tonight:
  exceeds Clarabel; vs HiGHS the remaining timing losses are maros_r7
  (2.0s vs 0.74s), cre_b/cre_d, ken_18, pds_10/pds_20, and several
  sub-second instances; coverage 23/24 each (linprogx misses greenbea,
  HiGHS misses qap15) with osa_60 back in the linprogx column.

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
- Gate every commit on `just ci` (ruff lint + format + ty + pytest). 150
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
