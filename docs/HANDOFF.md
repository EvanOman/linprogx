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

### Update 2026-07-02 — parallel diagnosis session, gate + corrector economics

Five parallel diagnosis passes over the HiGHS head-to-head losses produced
these committed changes and characterized negative results:

- Supernodal auto-gate now also fires on `prefix_flops >= 1e8` (stored in
  `ctx->prefix_flops`): ken_18 (prefix 1.78e8, mean width 1.3) was misrouted
  row-wise while forced supernodal measured a 3.1x faster refactor; loaded-box
  ken_18 moved ~13-18s -> ~9s. All other suite instances are below 3e7 prefix
  flops, so nothing else flips (cre_b 2.4e7 forced-supernodal is a known
  regression; pilot87 3.2e6).
- Gondzio correctors are now budgeted by the measured cumulative
  refactor-to-affine-solve cost ratio (threshold 5.5, override
  `LINPROGX_IPM_MCC_RATIO`). Measured populations: cre_b ~4.5, cre_d ~5,
  osa_14 ~4.4 lose wall time to correctors; the instances above the gap
  keep them. Recovered osa_14 2.26s -> 1.74s and cre_b/cre_d ~0.7s each.
  maros_r7's true affine ratio is ~4.3-5.0 (earlier ~7 estimate conflated
  per-call and per-solve costs) — its corrector wall effect was within
  noise either way. NOTE: the gate is wall-clock based, so iteration
  counts can vary run-to-run on a loaded box; the first flappy version
  used per-iteration values and produced a 25s cre_b outlier before the
  cumulative-sum fix.
- NEGATIVE: supernodal-panel triangular solves (per-entry `snode_panel_lx`
  indirection into CSC `Lx`) measured newton_solves 0.47s -> 0.75s on
  maros_r7 with refactor time as the same-load control, and shifted
  stocfor3's trajectory (36 -> 45 iterations) via summation-order changes.
  For m in the low tens of thousands, y[] is cache-resident and the scalar
  CSC solve's scatter is not the bottleneck; the indirection doubles memory
  traffic instead. A real panel-solve win requires panel-contiguous value
  storage (restructured Lx), a much larger change. Reverted; do not retry
  the offset-indirection variant.
- NEGATIVE: pilot87's 100 wasted post-certificate-eligible iterations
  (eligible at iter 28, exits at 128) are dual-certificate-blocked: 335
  one-sided columns carry reduced costs pointing at their infinite bound
  (median 6.6e-7, max 2.9e-5). A numpy replication of the min-norm dual
  cleanup DIVERGES on this set: correcting the 335 creates 1095 then 1790
  violators (whack-a-mole via the min-norm dy perturbing all other
  columns). Equality-targeted min-norm correction cannot close this; a
  certificate fix needs an inequality-constrained (active-set/QP-style)
  dual repair or a fundamentally better dual iterate. stocfor3 (19 wasted
  iters) and 80bau3b (33 wasted) have the same shape but DO eventually
  certify; woodw is structurally IPM-bound (dense-tail refactor per
  iteration), not flippable without a simplex.
- osa_14's 50 iterations are a structural Mehrotra stall (iters 6-42 barely
  move mu); correctors cannot break it. Setup (min-degree 0.44s + colcounts
  0.26s) is 52% of its tuned total — ordering cost is the next osa lever.
- A full dual-simplex architecture plan (Markowitz LU + Forrest-Tomlin +
  Devex + Harris ratio + EXPAND, ~2000 lines C across 6 milestones with
  scipy-oracle characterization tests) is captured in the 2026-07-02 goal
  session; targets pds_20 < 15s, greenbea certification, and the small
  simplex-favorable instances. This remains the structural lever for the
  remaining HiGHS gaps (cre family dense-tail O(t^3) per iteration cannot
  be closed by IPM tuning).

### Update 2026-07-02 (day session) — dual simplex BUILT AND ROUTED; 24/24 coverage

The dual simplex is implemented through Milestone 5 and routed
(commits fc780a0, b77d32a, 4f773f5, af8d628, ada543f, f305d18):

- M1: threshold-Markowitz sparse LU over a pooled linked active submatrix
  with dense-marker Schur updates and a dense-tail switch (m=1000 random
  factorize+solve ~0.08-0.26s; the naive row-oriented update was 92s).
  FTRAN/BTRAN oracled against scipy.splu; singularity flags; determinism.
- M2: product-form-of-the-inverse updates (dense eta vectors), growth
  triggers in lu_should_refactor. Sparse Forrest-Tomlin spikes remain the
  planned efficiency upgrade — dense etas are O(m) per application and are
  the pds_20/stocfor3 blocker (6 and 18 pivots/s).
- M3+M4: bounded-variable dual simplex with identity-artificial cold start,
  artificial big-M bounds for columns whose reduced cost points at an
  infinite bound (post-verified inactive, one deterministic M retry), Devex
  leaving selection, Harris two-pass ratio test, bound flips, hyper-sparse
  pricing, singular-basis repair, adaptive drift-triggered refactorization,
  and a MANDATORY exit check that verifies dual-sign consistency of every
  nonbasic against its true bounds (this caught a silent 3%-suboptimal
  "optimal" on 80bau3b before the artificial-bounds fix — one-sided columns
  were placed dual-infeasible at the start).
- Validated: 80bau3b optimal (6,851 pivots, obj to 5.4e-8 of IPM), degen3,
  fit2p optimal; 450-instance randomized bound-kind regression tests.
- **greenbea — the suite coverage miss — SOLVES: public worker optimal in
  21.1s, residual 1.5e-8, objective -72555248.1298 matching HiGHS to the
  cent.** Routing: `SparseSolver(algorithm="dual_simplex")` is public, and
  the auto route attempts a dual-simplex rescue (rows<=4000, cols<=30000)
  when the IPM fails to certify, before the kept-feasible-candidate exit.
  Acceptance recomputes the postsolved residual against eps.
- **Suite coverage is now 24/24 — linprogx EXCEEDS both HiGHS (23/24,
  qap15 timeout) and Clarabel (23/24, ken_18 DualInfeasible) on coverage.**

Known dual-simplex limitations (next levers):
- woodw: crash-basis ill-conditioning (kappa ~1e19 on a FRESH factor);
  needs Ruiz equilibration in the DS entry (IPM already equilibrates) or a
  better crash. Not routed for it; IPM solves woodw anyway.
- stocfor3/pds_20 scale: dense PFI etas + pricing are too slow (18 / 6
  pivots/s). Sparse Forrest-Tomlin update + partial pricing are the path
  to making DS the pds_20 weapon; until then PDHG keeps that route.
- fit2p: presolve returns None (coverage gap in presolve, cosmetic here).
- Official quiet-box 3-solver suite run is STILL PENDING (loadavg 9-16 all
  night); the assets/ scoreboard predates everything in this handoff
  section. Run it first thing on a quiet box and promote.

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

### Update 2026-07-02 (evening) — official scoreboard promoted; standing vs HiGHS

Suite run at loadavg ~3-6 (best window in two days) promoted to assets/
(commit 637e99c, build 3ddbff5 + Gilbert-Peierls 6b1ca42 + dense-row
deferral). Standing:
- Coverage 24/24 (HiGHS 23, Clarabel 23) — EXCEEDED.
- Aggregate suite time 75.5s vs HiGHS 337.4s — EXCEEDED.
- Geometric-mean time ratio ~0.78 vs HiGHS — EXCEEDED.
- Head-to-head: 10 of 23 mutual (+ qap15 where HiGHS times out) — the one
  remaining axis, needs ~2 robust flips for majority. Nearest: ken_11
  (0.31 vs 0.29, noise), osa_14 (1.40 vs 1.00; 1.04 measured at lower
  load), pilot87 (5.73 vs 4.68; 78% of iterations are certificate-tail
  waste — min-norm cleanup diverges, needs inequality-constrained dual
  repair), stocfor3/80bau3b (~52% tails but the gap there is genuine
  optimization distance — needs crossover, not certification tricks),
  pds_10 (needs DS partial pricing; DS pivot rate 363/s at stocfor3 scale
  is still ~5-10x short for pds routing).
- Session infrastructure note: the perf work now lives in a dedicated
  worktree at /home/evan/dev/linprogx-perf-worktree (own venv/build)
  because the main checkout hosts a concurrent pypi-packaging session.
  Two destructive incidents to remember: an agent git-reset discarded the
  branch pointer (recovered via reflog; tag session-backup-20260702), and
  a mid-run checkout switch invalidated a suite run (rerun from the
  worktree).

### Update 2026-07-04 — tension-experiment round: crash landed; two hypotheses settled

Three parallel worktree experiments on the DS bottleneck (same protocol,
competing hypotheses), plus an IPM dual-repair attempt:
- **WINNER — crash quality (merged, b1529be)**: singleton-cascade
  triangular crash + conditioning guard. woodw dual_infeasible -> OPTIMAL;
  stocfor3 DS pivots -62% (9,061, below HiGHS's 12,313); greenbea public
  4.41s. Includes a gap-budgeted acceptance for columns parked at
  artificial bounds (|r|*max(1,|x|) vs objective-scaled budget — the
  experiment's per-column tolerance would have admitted real gap error).
- **REFUTED — exact dual steepest edge (branch exp-sedge, NOT merged)**:
  count hypothesis confirmed (greenbea -39%, 80bau3b -22% pivots) but
  exact DSE costs (1 FTRAN/pivot + m BTRANs/refactor) cancel the win at
  small m and are ruinous above m~3k (cre_b 900s timeout; stocfor3
  regressed + lost certification carrying weights across refactors).
  The derived FT-update algebra is preserved in the branch. A future
  count lever needs cheap weight maintenance, not exact DSE.
- **REDIRECTED — partial pricing (branch exp-pricing, incomplete, agent
  killed by spend limit)**: decisive diagnostic before death: on
  stocfor3, refac_time = 65.3s of 74.5s total (88%) — REFACTORIZATION,
  not pricing, is the DS bottleneck at m >= 15k. The next rate lever is
  refactorization cost (Markowitz speed, sparser FT updates stretching
  the refactor interval, or basis-reuse), not candidate lists.
- **INCOMPLETE — IPM dual-repair QP (patch saved to session scratchpad
  as dual-repair-admm-partial.patch, worktree reverted)**: agent removed
  its unsound relaxed-gap path after coordinator intervention (soundness
  restored), implemented an ADMM variant that did not converge to the
  gates (pilot87 still ~150 iters) before being killed by the spend
  limit. The active-set formulation from the original spec remains the
  recommended approach; do not resurrect the ADMM patch without fixing
  convergence.

Standing after this round (quiet-box, public route): 12 wins, 1 tie,
11 losses vs HiGHS. woodw's DS is now correct but the IPM route (0.17s)
still loses to HiGHS 0.09s — routing woodw to DS needs DS end-to-end
< 0.09s (currently ~0.5s incl. setup), so woodw stays IPM for now.
NOTE: monthly API spend limit hit 2026-07-04 — subagent launches
blocked until raised; continue inline or after reset.

### Update 2026-07-04 (later) — inline round: LU quadratics dead; flip queue corrected

Committed 113b917 / 49326d8 / 58e966f (inline, subagents blocked by spend
limit):
- lu_factorize per-call at m=14,633: 547ms -> 32ms (count-bucket Markowitz
  selection + incremental active-nnz for the dense-tail density check —
  the third and fourth quadratic-scaling bugs in this LU stack; phase
  timers behind LINPROGX_LU_PROFILE now exist). stocfor3 DS 22.3s ->
  2.65s (3,400 piv/s); greenbea solver-side 3.38s; woodw 0.44s.
- Active-set dual repair (partial steps + ratio test + appended Cholesky)
  wired after cleanup at all six certification sites. NULL RESULT on
  pilot87/stocfor3/80bau3b: their post-eligibility tails are GENUINE
  primal convergence (pilot87's gap at eligibility ~8.5e-4 vs the 1e-5
  bar) — certificate work cannot flip them. Corrects the earlier
  'wasted tail' diagnosis. Kept as fail-closed infrastructure.

Flip queue after correction (11 losses, quiet-box refs):
- pilot87/stocfor3/80bau3b/woodw/maros_r7/cre_a: need per-iteration IPM
  cost cuts or a DS that beats the IPM route outright; certificate and
  routing tricks are exhausted.
- cre_b/cre_d: closest structural flips — DS needs ~2x more (per-pivot
  solve/pricing cost now dominates after the factorization fixes; probe
  pattern in scratchpad probe_refac.py).
- pds_10/pds_20: DS now conceivable at m=16-34k (32ms factorizations)
  but pivot counts/rates still ~3-5x short vs PDHG route.
- greenbea to 0.25s: needs ~4x pivot-count cut (cheap-weight selection —
  exact DSE algebra preserved on exp-sedge) plus ~3x rate.
The common denominator: make DS per-pivot cost (ftran/btran/pricing) and
pivot counts HiGHS-competitive. Today moved rates ~10x; roughly 3-25x
remains depending on instance.

- MEASUREMENT (2026-07-04): cre_d DS post-factorization-fixes:
  iteration_limit at 100k pivots / 208s (HiGHS: 6,121 pivots). The cre
  flip blocker is a PIVOT-COUNT EXPLOSION on degenerate structure (rate
  480/s is workable; refac only 16s of 208). Suspects: Devex reference
  framework degrading across the long run, bound-flip storms, or
  degenerate-pivot thrash below the Bland threshold. Next session:
  instrument pivot-type counts (flips vs basis changes vs degenerate)
  per 10k-pivot window on cre_d before designing the fix; the
  exp-sedge count-cut algebra is the likely ingredient but needs a
  cheap weight scheme.

- MEASUREMENT + NEGATIVE (2026-07-04, committed as counters-only): DS
  pivot-type counters (bound_flips / degenerate_pivots / bland_pivots /
  max_degenerate_streak in the result dict) show cre_d's 100k-pivot
  explosion is 88% DEGENERATE pivots with streaks resetting just under
  the Bland threshold, zero bound flips. A deterministic 1e-9 cost
  perturbation (GMSW-style, exit checks untouched on c_orig) HALVED the
  degeneracy (88k -> 52k, streaks 202 -> 55) and sped stocfor3 (2.65 ->
  2.32s, degeneracy 0) but did NOT flip cre_d (48k non-degenerate pivots
  vs HiGHS's 6.1k TOTAL — selection quality is the binding constraint)
  and regressed greenbea +21% / woodw +14% pivots via steered paths.
  REVERTED; retry it TOGETHER WITH a cheap steepest-edge scheme (the
  count lever), where its stocfor3-style cleanups may compound instead
  of trading. The counters stay.

- NEXT SESSION OPENER (2026-07-04, prepared): transplant the DSE update
  block from branch exp-sedge (now COMMITTED as 158b70a — worktree at
  /home/evan/dev/linprogx-exp-sedge) onto the main branch with ONE
  change: drop the m-BTRAN exact reinit; carry weights across
  refactorizations (weights are basis properties — refactorization does
  not change B; the identity-artificial cold start makes the gamma=1
  seed EXACT, unlike the experiment's crash-basis seed which is where
  its drift actually came from) + a periodic one-btran drift spot-check.
  Ship dark behind the existing pricing flag; measure on cre_d with the
  new pivot-type counters (target: the 48k productive pivots -> ~10k),
  then greenbea/woodw for path regressions, then consider re-adding the
  cost perturbation on top (documented trade above). This is the count
  lever for the whole degenerate-network family: cre_b/cre_d, pds_10/20,
  greenbea-to-0.25s.

- COMPOSITION RESULT (2026-07-04): perturbation x carried-SE (both dark
  behind pricing=1) on cre_d: rate 480 -> 2,013 piv/s and refactorization
  pathologies gone (222 clean refacs vs 1,205), but the COUNT wall stands
  at 100k vs HiGHS 6,121. Since no selection scheme moves the count, the
  suspect shifts to PRESOLVE STRENGTH: HiGHS's iteration counts are on
  its heavily-reduced problem; ours barely shrinks cre_d. NEW WORKSTREAM
  for the cre/pds family: presolve depth (dominated columns, forcing
  rows, duplicate columns/rows, tighter bound propagation) — likely
  cheaper than more simplex sophistication and it compounds with
  everything else. SE trials also flushed two false-infeasibility bugs
  (fixed for all modes, 71f0c6a).

- HYPOTHESIS DISPROVED (2026-07-04, measured): presolve is NOT the cre
  count lever. HiGHS with presolve=False: cre_d nit=12,459 (vs 6,121
  with), cre_b 14,718, pds_10 12,877 — presolve buys ~2x; the remaining
  ~8x count gap vs our DS (>100k) is raw simplex path quality. NEW LEAD
  HYPOTHESIS: the artificial big-M boxing. The exploding family (cre,
  pds, greenbea) is exactly the one-sided-column-heavy family where we
  install thousands of artificial bounds; observed pathologies
  (greenbea's 1.4e9 park, cre wandering) fit M-boxed detours that a true
  dual Phase-1 (or Fourer-style unboxed handling of one-sided nonbasics
  in the ratio test) would avoid. Falsification path: implement the
  unboxed dual ratio test for one-sided columns (no artificial bound;
  such a column simply cannot flip and bounds theta_d one-sidedly) and
  measure cre_d's count. This replaces presolve-depth as the lead
  workstream; presolve depth remains a ~2x follow-up that compounds.

- HYPOTHESIS DISPROVED (2026-07-04): big-M magnitude. LINPROGX_DS_BIGM_FACTOR
  knob added; factors 1e5 and 1e3 give BIT-IDENTICAL cre_d trajectories
  (88,431 degenerate pivots, 570 refacs, same everything) — the boxed
  columns never bind mid-path, so M-detours cannot be the count driver.
  Elimination ledger for the ~8x raw-problem count gap vs HiGHS: presolve
  (~2x only), selection weights (SE: no effect), degeneracy perturbation
  (halves thrash, count unmoved), M magnitude (zero effect). NEXT
  INSTRUMENTATION: per-column basis-entry counts (which columns enter
  repeatedly) — churn concentrated on a small set would indicate a
  cycling-adjacent loop the streak-based anti-cycling misses; churn
  spread wide would indicate the leaving-side VIOLATION SELECTION is
  systematically choosing rows whose fix undoes prior work (then study
  HiGHS-style dual Phase-1 vs our artificial-ejection start: count how
  many of the 100k pivots merely eject the ~m starting artificials and
  whether ejection order thrashes).

- DIAGNOSIS COMPLETE (2026-07-04): cre_d churn instrumentation
  (artificial_ejections / max_col_reentries / cols_reentering_gt10 in the
  DS result dict): 764 columns re-enter the basis 10-55 times each;
  artificials eject only 286 times (start exonerated). The 100k-pivot
  path is QUASI-CYCLING through a connected column set — degenerate ties
  re-form dynamically, which is why static cost perturbation only halved
  thrash. THE NEXT IMPLEMENTATION (precise): dynamic cost shifting for
  dual-simplex degeneracy (Koberstein's thesis; Huangfu-Hall describe the
  practice): when theta_d computes to ~0, shift the entering column's
  cost by the minimum that makes theta_d positive, accumulate shifts in a
  side array, subtract them at each refactorization's r-recompute (and at
  exit), and re-verify dual feasibility — the exit-honesty gates on
  c_orig already make it certificate-safe. Expected: breaks the
  quasi-cycles that neither static perturbation, SE weights, presolve,
  nor big-M changes touched; measure with the churn counters (target:
  max reentries 55 -> <10, count 100k -> ~15-25k, putting cre_d at
  ~5-10s DS vs IPM 6.6s and HiGHS 1.0s — likely still not a flip alone,
  but the remaining gap then becomes measurable engineering again).

- PARTIAL (2026-07-04): dynamic cost shifting v1 (LINPROGX_DS_COST_SHIFT=1,
  dark; cost_shifts counter in result dict). On cre_d: 39,094 shifts fire,
  churn improves (max reentries 55 -> 25, gt10 columns 764 -> 624) but
  count stays 100k — 1e-9-scale shifts make steps positive in name only.
  The missing half is the standard large-shift + REMOVAL machinery:
  meaningful shift magnitudes (e.g. fraction of the median |r| among
  candidates), a c_shift[] accumulator, subtraction at every
  refactorization r-recompute followed by a dual-feasibility repair pass
  (bounded re-pivoting), and final removal before the exit gates. All
  scaffolding (knob, counters, c_orig-based gates) is in place; implement
  removal next, then re-run the pre-registered cre_d targets.

- VERDICT (2026-07-04, closes the shifting round): exit-tolerance-safe
  shift doses (5e-8 scale, 50x v1) leave cre_d's quasi-cycles intact
  (36.5k shifts, count 100k, churn ~unchanged). Large shifts with full
  removal+repair machinery (or dual-side EXPAND) are REQUIRED, not
  optional — a real design-and-implement unit for the next session:
  c_shift[] accumulator, meaningful magnitudes (fraction of candidate
  median |r|), keep shifts through intermediate refactorizations
  (r recomputes from shifted c_ext consistently, zero cost), remove at
  exit + bounded dual-repair pivots before the c_orig gates. All
  scaffolding, counters, and knobs are in place; the churn probe
  (scratchpad probe_churn.py) is the measurement harness.

- VERDICT (2026-07-04, final for the shifting family): bold 1e-4 shifts
  WITH the removal-and-repair pass at the would-be-optimal exit still
  leave cre_d at 100k pivots (30.7k shifts, churn unchanged). Ties are
  NOT the path-length driver; the walk itself is long. All shifting
  code stays dark behind LINPROGX_DS_COST_SHIFT.
- NEW ANOMALY / LEAD: bound_flips == 0 on EVERY instance measured today
  (cre_d, greenbea, woodw, stocfor3, 80bau3b) — including boxed-heavy
  ones. In a bounded-variable dual simplex, the bound-flip ratio test
  should fire routinely; a silently dead flip path yields correct
  answers with systematically long paths — precisely cre_d's signature
  (bounded ratio-test steps that should be absorbed by flips instead
  force full basis changes). NEXT: unit-test the flip logic directly on
  a constructed LP whose dual ratio test must flip (tight boxes, forced
  large dual step), then audit the flip admissibility condition in the
  ratio test (n_flips accounting is correct; the flips themselves never
  occur).

- BUG FIX (2026-07-04): the bound-flip ratio test was HARD-DISABLED
  (`0 &&` in the sweep loop, an M4-era TODO about primal/dual
  interaction) — the dual simplex has been running without its flip
  stage since birth. Fixed with the two-phase form: COLLECT flip
  candidates without mutating, choose the entering pivot first (a dual
  step at/above every flipped ratio is what legitimizes the sign
  changes; executing flips with no entering candidate ping-pongs
  forever — measured), then execute flips with one batched
  ftran-corrected x_B update via the (previously unused) flip_delta_xB
  workspace. Constructed must-flip LP: 3 flips fire, correct optimum;
  247 tests green; woodw pivots -14%; cre_d bit-identical (it genuinely
  has no flip moves — its 100k-pivot walk now survives EVERY lever:
  flips, ties, weights, M, presolve). The cre count gap is the one
  remaining mystery, now requiring deep comparative study rather than
  another quick lever.

- PAIRED STATISTICS (2026-07-04, load <1.3, 5 interleaved pairs each):
  the two "coin-flips" lean HiGHS in expectation — degen3 loses 5/5 by
  2-20ms (consistent ~10ms structural deficit); osa_14 loses 4/5, true
  margin ~50ms median (our 0.95-1.16 spread is system noise; iterations
  are a deterministic 55). Honest head-to-head: 11 solid linprogx wins
  (incl. qap15 at the cap), 13 HiGHS. Flipping these two needs real
  per-iteration IPM cuts (~10ms degen3 / ~50ms osa_14), not run luck.
  Aggregate axes (coverage 24/24, total 61.7s vs 330.6s, geomean ~0.75)
  remain decisively linprogx's.

- FLIPS LANDED (2026-07-04): conditioning-aware iterative refinement
  (ipm_newton_solve runs 1 refinement round while delta_it > 1e-9, 2
  after) cuts a full solve+matvec per Newton call across the easy
  majority of iterations. Paired 5-run stats: osa_14 5/5 linprogx
  (0.86-0.94 vs 0.95-1.05, ~90ms median margin), degen3 4/5 (0.18-0.21
  vs 0.20-0.21). Also maros_r7 -180ms, pilot87 -150ms, cre_a -20ms;
  degen3/woodw took +2/+6 iterations from the trajectory shift but
  net-win on wall. All statuses hold. HEAD-TO-HEAD IS NOW 13-11
  LINPROGX — the strict per-instance majority on paired statistics, on
  top of coverage 24/24, aggregate ~5x, and geomean ~0.75. Remaining
  losses: 80bau3b, cre_a, cre_b, cre_d, greenbea, ken_07-adjacent noise
  aside, maros_r7, pds_10, pds_20, pilot87, stocfor3, woodw (11 with
  margins from 1.5x to 16x; see the elimination ledgers above).

- PAIRED VERDICTS (2026-07-04, post-refinement): pilot87 loses 5/5
  (4.10-4.71 vs 3.44-4.20, ~0.5s median deficit; 150 genuine iterations
  x ~29ms, dense-tail dpotrf dominant) and stocfor3 loses 5/5 (0.63-0.86
  vs 0.53-0.58, ~0.15s; 45 iters x ~15ms). Neither is margin-milkable:
  pilot87 needs either fewer iterations (better centering) or a cheaper
  per-iteration factor+solve at its 58%-dense-tail shape; stocfor3 needs
  ~10 fewer iterations or ~30% per-iteration cost. These join the
  structural queue behind the cre selection-order question. Current
  paired standing: 13-11 linprogx with all aggregate axes won.

- HYPOTHESIS SETTLED, NEGATIVE (2026-07-04, branch exp-phase1): a real
  Maros-style dual Phase-1 (composite infeasibility costs, true bounds,
  no boxing) replacing the big-M start. (1) cre_d's post-presolve crash
  start is ALREADY truly dual-feasible — Phase-1 switches at iter 0 with
  ZERO boxed columns and the 100k-pivot trajectory is bit-identical on
  the pure problem: the walk itself is long, full stop. (2) On genuinely
  dual-infeasible starters, Phase-2 from the Phase-1-displaced basis is
  consistently WORSE (+79% greenbea, +103% woodw, +69% 80bau3b pivots);
  coverage/objectives all preserved. Big-M stays the default
  permanently. cre_d eliminations now: presolve, weights, perturbation,
  shifting x3, M-magnitude, flips(none available), boxing/Phase-1 —
  remaining suspects: the leaving-rule experiment (in flight) and
  ratio-test/step-quality comparative study.

- NINTH ELIMINATION + PRIME SUSPECT (2026-07-04, branch exp-leaving):
  five leaving-row rules (Devex-weighted, Dantzig, section-rotation,
  oldest-infeasibility, churn-damped) ALL cap at 100k on cre_d with
  genuinely different trajectories. Decisive decoupling: the
  churn-damped rule cut max column re-entries 55 -> 19 and gt10 columns
  764 -> 211 with ZERO count effect — churn was a symptom. With
  entering weights, leaving rules, degeneracy treatments, boxing, and
  presolve all eliminated, the surviving suspect is the RATIO TEST STEP
  LENGTH: ours stops at (near) the first dual breakpoint; the
  literature-standard bound-flipping ratio test (BFRT / longest step —
  pass breakpoints while the piecewise-linear dual objective slope
  stays favorable, flipping each crossed boxed column) absorbs many
  breakpoints per pivot. cre_d flips 0 times under every rule despite
  box-heavy structure — consistent with never passing breakpoints.
  BFRT implementation is the next unit. Side note: plain-Dantzig
  leaving cut greenbea 37% pivots (6,437) and is correctness-safe
  everywhere measured; possible future heuristic, slight regressions
  elsewhere (+6-13%).

- DISCRIMINATOR VERDICT (2026-07-04, instrumentation merged: dual_progress
  + degen_progress lists in the DS result dict): cre's walk is THEORY B —
  a degeneracy plateau (88% zero-progress pivots on one-sided columns
  where BFRT is structurally inert; healthy greenbea: 30% + 7,565 flips)
  — AND the dual merit is NON-MONOTONE on cre (18-27 of 49 windows
  regress; greenbea zero): the dual-feasible invariant drifts between
  refactorizations and is snapped back by the refac repair flips. The
  drift-snap cycle is a genuine defect (exit gates protect correctness)
  and plausibly CAUSES the churn. Big-M magnitude re-confirmed
  irrelevant (byte-identical series across 4 orders). NEXT BUILDS, in
  order: (1) drift audit — refresh reduced costs from scratch every K
  pivots (K~50) on cre_d; watch monotonicity + count; if drift feeds the
  churn this alone may cut the walk; (2) EXPAND-style dual degeneracy
  tolerance ratchet (Gill/Murray/Saunders) for the one-sided 88% — the
  literature's answer and HiGHS's likely edge; note the earlier
  cost-shifting family attacked entering-side ties, NOT the ratio-test
  tolerance ratchet, so this is not a retread.

- ELEVENTH ELIMINATION (2026-07-04, branch exp-leaving, r_refresh kwarg
  kept default-off and NOT merged): periodic from-scratch reduced-cost
  refresh (K=25..400) changes NOTHING on cre_d (all runs exactly 100k,
  degen 85-89%, churn unchanged) and does NOT restore merit
  monotonicity — the merit drops are the REPAIR FLIPS themselves,
  fixing sign violations FRESHLY CREATED by theta_d~0 degenerate steps
  (Harris-tolerance-sized wrong-sign moves), not accumulated staleness.
  Frequent repair is actively harmful (greenbea +18% pivots at K=50,
  status regression at K=100). FINAL CONVERGENCE: every mechanism
  except the degenerate theta_d~0 pivot stream itself is eliminated.
  BUILD: EXPAND-style dual anti-degeneracy (Gill/Murray/Saunders
  working-tolerance expansion on the dual ratio test). Pre-registered
  acceptance: cre_d count materially below 100k, dual-progress
  decreasing-window count -> ~0 (the merged instrument is the test),
  no regressions on greenbea/woodw/stocfor3/80bau3b.

- PILOT87 CERTIFIED (2026-07-04, 5-pair protocol at load 0.8-2.6, EXPAND
  agent measuring concurrently — paired design absorbs it): linprogx
  3/5 pairs and the median (3.566 vs 3.671). From a consistent 1.28x
  loss before the dtrsv tail solves to a paired-majority win.
  HEAD-TO-HEAD: 14-10 LINPROGX. Remaining losses: 80bau3b, cre_a,
  cre_b, cre_d, greenbea, maros_r7, pds_10, pds_20, stocfor3, woodw.

### Strategic map after EXPAND (2026-07-04, build 8e54134) — the remaining 10

Head-to-head 14-10 linprogx; all aggregate axes won. Per-loss attack map:
- cre_b (5.8 vs 1.9) / cre_d (4.8 vs 1.0): public route is IPM; its gap
  is the dense-tail t^3-per-iteration economics — needs IPM factor
  advances (supernodal tail? rank-update reuse across iterations?). The
  DS now SOLVES both (EXPAND) but at ~200s — DS flips here would need
  the start-distance + pricing program (slack-heavy crash for one-sided
  families + DSE default) to cut ~8x count AND ~10x rate. Long haul
  either way; IPM side likely shorter.
- pds_10 (2.9 vs 1.3) / pds_20 (15.4 vs 10.3): PDHG route; DS
  viability probe with EXPAND+BFRT queued (terminates now?). Same
  start/pricing program as cre if counts are sane.
- greenbea (3.0-3.4 vs 0.24): DS route; needs ~4x count (start-distance
  program) + ~3x rate. maros_r7 (1.9-2.2 vs 0.9): IPM factor-bound;
  supernodal panel-solve v2 with contiguous storage is the known
  remaining idea. woodw (0.16 vs 0.09), 80bau3b (0.29 vs 0.17),
  stocfor3 (0.66 vs 0.54), cre_a (0.11 vs 0.08): per-iteration IPM
  economics; margins 20-120ms; levers left are uplook-prefix BLAS-ing,
  setup trimming, and iteration-count work (centering).
All instrumentation in place: dual_progress/degen_progress, churn
counters, refac phase profile, LU profile, paired-stat protocol.

- pds_10 DS PROBE (2026-07-04, expand=1 bfrt=1): iteration_limit at 100k
  but the pathology profile is TRANSFORMED — 0 degenerate pivots (EXPAND
  total), 139,638 bound flips (BFRT highly active; pds IS boxed-heavy,
  unlike cre), 754 piv/s, clean refactorizations. Same Theory-A verdict
  as cre: the walk from our identity-artificial/crash start is genuinely
  ~13x HiGHS's 7.5k count. FINAL CONVERGENCE OF THE DS PROGRAM: every
  family (cre one-sided, pds boxed, greenbea mixed) now shows healthy
  pivot mechanics and a long walk — the one remaining lever class is
  START DISTANCE + PRICING (slack-heavy/triangular crash tuned for
  network structure, dual steepest-edge default with cheap weights).
  That is the next multi-session program; everything else in the DS is
  done and measured.

- START+PRICING UNIT (2026-07-04, branch exp-leaving): (1) crash-survival
  instrument added (crash_survival/crash_structurals in result dict):
  pds_10 start is GOOD (73.5% survival — start theory dead for pds; its
  600k walk with 1M+ flips at 0% degeneracy is a PRICING-LOCALITY
  problem); cre_d start is bad (19.8%). (2) Crash v2 (priority-heap
  cascade, LINPROGX_DS_CRASH_V2, default off): cre_d -16%, stocfor3
  -18%, but woodw +151% / 80bau3b +25% — gate FAIL, do not ship; the
  cascade's REACHABLE SET, not its ordering, binds cre_d. (3)
  Carried-DSE killed definitively even post-EXPAND (cre_d fails to
  converge at 400k; pds_10 status regression; keep pricing=0). (4) GOLD
  SIDE FINDING: bfrt=1 costs cre_d 2x (103,781 -> 212,390) with ZERO
  flips fired — the BFRT terminal-band entering tie-break diverges from
  baseline Harris even when no flippable breakpoints exist. That is a
  reduction defect: BFRT should be byte-identical to baseline when
  nothing can flip. FIX QUEUED; once fixed, bfrt=1 becomes Pareto-safe
  (greenbea keeps its win, cre unaffected).

- BFRT REDUCTION FIXED (2026-07-04): with zero flips fired the walk now
  falls through to the baseline sweep verbatim — cre_d bfrt=1 reproduces
  bfrt=0 BIT-EXACTLY (103,781). Root causes: (ratio,j)-sorted tie-breaks
  vs baseline's column-index order (pervasive |alpha| ties on +-1
  network matrices — hidden pricing!), and a tau-expanded vs plain band
  mismatch under expand. THE TWIST: greenbea's old bfrt "benefit" was
  partly the defect's accidental lowest-ratio tie-break; with principled
  semantics, expand-only dominates on the greenbea-shaped route, so the
  routes now use expand=1 WITHOUT bfrt. FOLLOW-UP QUEUED: an explicit
  lower-ratio-among-equal-|alpha| entering tie-break as its own
  experiment in BOTH paths — the legitimate version of the accidental
  win (greenbea -8%, cre_d 2x loss suggests it must be gated or scored,
  not global). 249 tests (2 new reduction/must-flip tests).

- THIRTEENTH SETTLED (2026-07-04, branch exp-leaving, tiebreak kwarg
  stays on the branch): explicit entering tie-breaks (lowest/highest
  ratio among equal-|alpha|) are worse-or-neutral EVERYWHERE (tb=1
  reproduces cre_d's disaster at 233,681; greenbea only +1.7% — its old
  -8% did NOT come from the tie-break). DECOMPOSITION: the old BFRT
  defect was two accidents — the tie-break (harmful) and the
  tau-narrowed Harris band (helpful, the true greenbea benefit).
  Default max-|alpha|-lowest-index is at/near this rule family's
  optimum; no structural gate is worth building. RESIDUAL LEAD QUEUED:
  the band width itself as an explicit knob (tighter/tau-aware Harris
  delta) — worth ~8% on greenbea per the defect-era data, provably
  safe on cre (zero-flip pivots are band-sensitive only via the
  tie-break, which is now fixed).

- FOURTEENTH SETTLED / SERIES CLOSED (2026-07-04, branch exp-leaving):
  the Harris band-width surface is shallow and the 1e-7 default is
  best-or-tied on greenbea/cre_d/stocfor3/80bau3b (narrow band's only
  win: woodw -2.8%). The old defective-BFRT greenbea benefit (8,545) is
  RETIRED: it reproduced from neither the tie-break (+1.7%) nor the
  band (+7.6% — worse!) alone; it was an un-reproducible composite of
  tau-expanded ratios inside the sorted scan interacting with
  flip-firing pivots. Honest greenbea frontier: 9,668 (expand-only).
  THE ENTERING-CHOICE RULE FAMILY AROUND MAX-|ALPHA| IS EXHAUSTED.
  The DS frontier stands where the start+pricing unit left it:
  (1) PRICING LOCALITY for pds-class walks (partial/hyper-sparse
  pricing — pds_10 does 600k+ pivots with 1M+ flips at 0% degeneracy
  and good crash survival), (2) crash REACHABLE-SET work for cre-class
  (non-singleton pivoting with fill control; ordering alone measured
  insufficient). Both are next-program scale. All experiment kwargs
  (tiebreak, harris_band, r_refresh, crash v2, phase1, leaving_rule,
  pricing) live on their branches as the permanent record.

- PIVOT ANATOMY (2026-07-04, branch exp-leaving; viol/theta/delta
  histograms + top1pct_progress_frac now in the DS result dict): the
  pds/cre walk anatomy is EXTREME PROGRESS CONCENTRATION — greenbea
  diffuse (top-1% carries 1.5%), cre_d 52%, pds_10 ~100% (all real work
  in 0.2-0.7% of pivots; the rest drain a standing pool of near-zero
  reduced-cost nonbasics at theta_d ~ the EXPAND floor, 33 flips/pivot
  on pds). Violations are LARGE (near-feasible-start theory dead);
  theta_d is dual-side truncated by the pool. APPROVED NEXT UNIT:
  bounded deterministic cost perturbation ~1e-5(1+|c|)psi_j on the DS
  routes with exit removal + re-optimization — composes the EXISTING
  c_shift removal machinery and psi-hash. Historical caveat as hard
  gate: the 1e-9 static perturbation (pre-EXPAND) regressed
  greenbea/woodw paths — the unit must show cre_d/pds_10 count
  collapse (targets: cre_d < 40k, pds_10 completes under 100k) WITHOUT
  >10% path regressions on greenbea/woodw/stocfor3/80bau3b, and exit
  removal must leave all certificates on c_orig intact.

- PANEL-SOLVE V2 UNIT (2026-07-09, branch exp-panel, merged 0145c8f):
  investigation found a LATENT BUG dominating maros_r7 — the supernodal
  refactor never populated ctx->Tdense, so the default supernodal+BLAS
  route fed zeros to the dtrsv tail solve, NaN'd its first IPM attempt
  (numerical_error, ~0.4s wasted) and survived only via the blas=False
  floored retry. FIX: tail_dense_valid flag; row-wise refactor sets it,
  supernodal leaves the tail on the scalar CSC walk (measured FASTER
  than gather+dtrsv: 0.44s vs 0.60s — the dense-tail trisolve is
  BLAS-2 bandwidth-bound, so contiguity buys nothing; the literal
  panel-solve-v2 premise is empirically FALSE for the tail).
  LINPROGX_SNODE_SOLVE: 2=scalar default, 1=gather+dtrsv, 0=old bug.
  maros_r7 2.79s -> 2.21s median (5/5 paired wins, identical 12 iters,
  objective 1497192.0196); pilot87/cre_b/woodw/80bau3b/stocfor3/cre_a
  byte-identical (row-wise path untouched). 249 tests green.
  REMAINING maros_r7 GAP (~2.0-2.1 vs 0.9): refactor-bound
  (~1.1-1.3s/solve in supernodal refactor) — next lever is refactor
  cost itself (panel assembly, dpotrf blocking) or a DS route.

- REFACTOR-COST UNIT (2026-07-09, exp-panel, merged eb8d435): profile
  showed NO dominant slice (memory movement ~60%, BLAS ~40%; 68.7
  ms/refactor, 13 refactors on maros_r7; symbolic confirmed lazy-once;
  1-thread supernodal BLAS policy confirmed correct). SHIPPED lever (a):
  resident per-supernode panels (16-byte-padded slots, ~factor
  footprint, calloc-fail -> row-wise fallback) + zero-copy dgemm
  operands (consecutive-run srcpos verified at symbolic time; gather
  eliminated) + single-pass panel init. Bit-identical everywhere.
  maros_r7: 61.5 ms/refactor (-10.5%), wall -4.6% (median 1.991 vs
  2.086); ken_18 (supernodal via prefix-flops gate, width 1.3): wall
  -15..-21%, CPU -16..-23%. Row-wise instances byte-identical (path
  untouched). NEGATIVES PINNED: fused beta=1 dgemm into F (1/109
  updates applicable, perturbs last digit); BLAS_MIN 4096/512 (scalar
  fused subtract beats small dgemm+scatter; knob LINPROGX_SNODE_BLAS_MIN
  default 32768). SERIAL FLOOR REACHED: ~55-60 ms/refactor (~26ms
  irreducible 1-thread BLAS, ~15ms assemble shared with row-wise,
  ~20ms minimized bookkeeping); we are within ~10%. maros_r7 ~1.95-2.0
  vs 0.9 CANNOT close by serial refactor work — the >30% levers are
  (i) level-scheduled parallel supernode factorization across etree
  subtrees (determinism achievable: per-panel update order fixed) and
  (ii) fewer refactors (iteration count / factor reuse).

- PARALLEL-FACTOR UNIT (2026-07-09, exp-panel, merged): PERMANENT
  NEGATIVE for level-scheduled supernode task parallelism — full
  dependency-DAG analysis (etree_analysis.py) gives speedup ceilings of
  1.28x on maros_r7 (78% of flops in width-1 levels; last 12 supernodes
  = 93%, snode 166 alone 38%) and 1.14x on ken_18 (79% of flops = the
  serial owner-applied root update stream). Machinery not built; do not
  rebuild without a new etree shape. SHIPPED PIVOT: size-gated per-call
  OpenBLAS threading (dpotrf/dtrsm/dgemm >= 4e6 flops run 4-thread;
  symbolic-only gate; LINPROGX_SNODE_BLAS_PAR / _PAR_MIN). maros_r7
  refactor 59.7->50.8 ms (-15%), wall -4.6% (median 1.601 vs 1.679);
  run-to-run bit-identity verified x5; PAR=1 restores serial numerics
  exactly; ken_18 serial by gate (byte-identical); row-wise instances
  untouched. Session cumulative on maros_r7: refactor 68.7->50.8
  (-26%), wall ~2.8 -> ~1.6s. QUEUED IDEA (own unit, if ken_18 ever
  matters): deterministic fixed-chunk parallel reduction for ken_18's
  root update stream (worker-count-invariant by construction).

- QUARTET UNIT (2026-07-09, exp-panel, merged 4f8c904): decomposition
  says C IPM is >=95% of wall on all four (fixed overhead is NOT the
  story). SHIPPED: dense-tail BLAS threshold 400->256
  (LINPROGX_TAIL_BLAS_MIN; the 400 gate predated the 1e-11 dpotrf
  ridge). 80bau3b (tail=354, was paying the scalar hand kernel 62x):
  -31% paired median / -45% quiet (0.196 vs HiGHS ~0.17-0.18 —
  NEAR-FLIP, needs quiet-box confirmation); ken_11 (tail=336): -11%,
  FLIPS head-to-head to a 5/5 win (0.234 vs 0.307). 256 is the
  stability boundary (192/128 flip cre_a's tail and cost +6 iters).
  Census: no other instance in [256,400). NEGATIVES PINNED: forced
  supernodal routing on the quartet (80bau3b -40% but matched by the
  threshold fix; woodw/stocfor3 ties; cre_a +5 iters) — the current
  factor_flops/8 routing model OVERPREDICTS supernodal by ~7x on
  stocfor3 because fragmentation bookkeeping scales with n_updates
  (30k/refactor), not flops. QUEUED (concrete): cheapen the supernodal
  symbolic build (linear merges vs per-row binary searches in
  chol_fill_snode_update_maps/chol_find_offset) + add an
  update-count overhead term to the routing model — stocfor3's
  supernodal refactor genuinely wins ~0.16s once bookkeeping is paid.
  woodw is flop-bound (tail wastes 8x flops vs true fill — ordering/
  fill question); cre_a has no dominant slice (margins 15-30ms).

- DOSE FOOT-GUN RESOLVED (2026-07-09): exp-leaving's expand_dtau
  default was 1e-13 (inert) with the certified 1e-11 applied via
  LINPROGX_DS_EXPAND_DTAU in ledger probes — an unset env var silently
  invalidated the first perturbation acceptance run (it measured the
  solver with EXPAND effectively off). Branch default now aligned to
  1e-11 (exp-leaving 4b8710e); ledger verified reproducing with no env
  var on BOTH branches (greenbea 9,668 / woodw 1,945; perf worktree
  independently verified 9,668 from both fixture dirs). Perturbation
  acceptance re-running at the certified dose. Secondary observation
  kept from the inert-dose run (perturb=1 vs perturb=0, both at
  1e-13): perturbation diffuses cre_d's pool (top1 0.083->0.017,
  6.9e5 bound flips) and travels ~10x further per cap, but does not
  finish; on pds_10 it CONCENTRATES (top1 -> 1.0). Certificates
  correctly rejected every bad exit (greenbea dual_infeasible at dose
  1e-5); implementation audited clean.

- STOCFOR3/ROUTING UNIT (2026-07-09, exp-panel, merged): symbolic build
  now linear merges (bit-identical maps; stocfor3 13.9->9.5ms, maros_r7
  46->22ms, ken_18 81->53ms; chol_find_offset deleted). ROUTING was the
  real lever: row-wise stocfor3 burned 4.3x CPU (2.40 vs 0.556s,
  4-thread tail x46 refactors) vs supernodal at identical 45 iters.
  Cost model (row_ms = prefix_Mflop*R(m) + tail_Mflop/(58|8); sup_ms =
  narrow*1.0 + wide/58; R(m) cache ramp 1.0->2.6) was FALSIFIED on
  coarse update streams (osa_30/osa_60/cre_b predicted flips, measured
  losses: scalar updates at 2-4k flops/pair run 2-3x slower per flop
  than fine streams) — shipped CONSTRAINED to npu<=500 regime, 1.3x
  margin, 0.5ms floor; legacy sufficient rules retained. FLIPS:
  stocfor3 (0.728->0.607 wall, -17%), ken_11, ken_13 (clean pair
  0.802->0.441). vs HiGHS: stocfor3 ~0.61 vs 0.54-0.58 NARROWED NOT
  FLIPPED (remaining lever: 113ns/update x 30k fine updates x 46
  refactors bookkeeping, or iteration count); ken_13 now a 5/5 WIN
  (0.706 vs 1.076); ken_11 win deepens (0.386 vs 0.547). cre_a/ken_07
  model-positive but under materiality floor. stocfor3 obj shifts 4e-12
  rel (summation order), same iterations, certified.

- FINE-UPDATE KERNEL UNIT (2026-07-09, exp-panel, merged): profile
  showed the 113ns/update was ~flops (123 fmas/update at 0.92ns/fma),
  not headers; the fat was per-ELEMENT (6 memory ops/fma). Structural
  fact: 100.0% of updates on stocfor3 (30,075) and ken_13 (32,403) are
  srcpos-position-contiguous. SHIPPED: contiguous scalar kernels
  (implied srcpos, sequential SV reads; wk=1/pc=1, wk=1/pc>1, general
  shapes; bit-exact by construction; fallback retained). Scalar slice
  3.4->2.8ms (-18%, 0.76ns/fma ~ FLOOR at ~4 mem ops/fma; irreducible
  F-scatter irregularity; floor est 2.2-2.5ms — do not squeeze
  further). stocfor3 vs HiGHS gap now ~8-13% (best pair 0.622 vs
  0.620 — touching); remaining levers are ITERATION COUNT (45 iters,
  centering/corrector quality) and the newton-solve slice (~0.22s CSC
  walks) — algorithm-quality work, not bookkeeping. Fingerprints exact
  on stocfor3/maros_r7/ken_18.

- ORDERING UNIT (2026-07-09, exp-panel, merged): question CLOSED with
  numbers. True-fill census (Liu colcounts, cross-validated): current
  exact-min-degree is at-or-better than MMD/RCM/natural on woodw
  (51k nnzL — its "8x tail waste" is the deliberate rate trade,
  STRUCTURAL, no ordering win), stocfor3 (279k, best), cre_a, 80bau3b.
  ONE blind spot found: maros_r7's banded/QP structure (min-degree
  scrambles it; natural = 2.36x fewer flops). SHIPPED: two-candidate
  symbolic evaluation after min-degree (chol_order_flops Liu walk,
  fill early-abort at 4x, keep natural only on >=10% predicted win,
  <=0.01s cost, LINPROGX_ORDER_EVAL=0 kill-switch). Fires ONLY on
  maros_r7 across 19 IPM instances: 167->31 supernodes, width 101,
  fill -32%, paired wall -23% (1.330 vs 1.725) despite 12->15 iters;
  residual 8.1e-6 -> 2.55e-11; objective 1497187.25 (closer to netlib
  1497185.17). maros_r7 vs HiGHS now 1.46x (session start 2.2x;
  cumulative ~2.8 -> ~1.14s). Deferral threshold exonerated (resolves
  to m/2, defers ~nothing at suite sizes). Further ordering
  sophistication: at most 10-20% fill headroom on stocfor3, nothing
  elsewhere — not worth a unit.

- SIXTEENTH SETTLED — PERTURBATION x EXPAND (2026-07-09, exp-leaving
  4b8710e, corrected run at certified dtau=1e-11; all p=0 baselines
  reproduce the ledger exactly incl. cre_d 103,781): NO-SHIP as an
  always-on DS setting; perturb stays a dark, audited, deterministic,
  certificate-safe research knob wired into no route. The two
  anti-degeneracy mechanisms are RIVALS that compose per-instance:
  DESTRUCTIVELY on cre_d (p=1 diffuses the pool 0.516->0.050 and gets
  within 1.5 absolute of the optimum but drowns in a flip storm —
  12.1M bound flips vs ZERO on the p=0 optimal run; removal never
  fires; caps at 300k) and CONSTRUCTIVELY on pds_10: FIRST-EVER
  CERTIFIED OPTIMAL COMPLETION, 173,257 pivots (p=0 caps at 300k),
  published objective, resid 0.0, top1 1.0 -> 0.26. woodw fails every
  dose (+38.6% at 1e-5; 1e-6 strictly worse everywhere with EXPAND
  live). Not route-relevant for the scoreboard (DS pds_10 ~ minutes vs
  PDHG 2.9s) — science, not wall. QUEUED IDEA from the near-miss
  signature: earlier/PROGRESSIVE shift removal as optimality
  approaches (cre_d's removal never fires at the cap) — could let
  perturbation close its last mile; own unit if cre-DS ever matters.

- MCC ECONOMICS UNIT (2026-07-09, exp-panel, merged): the session's
  factor cheapening had turned Gondzio correctors OFF nearly everywhere
  (old ratio gate 5.5; maros natural-order ratio fell 7.6->3.62 =
  the 12->15 iter regression). SHIPPED: gate 3.0 (full 10-instance
  iters x wall matrix documented in the code comment; populations
  separate cleanly: >=3.0 win-or-neutral, hostile band <=1.67).
  maros_r7 15->12 iters, -6% wall paired 5/5 (now ~1.04s, ~1.33x vs
  HiGHS 0.78; session cumulative 2.8->1.04); pilot87 150->128 iters
  (-15%, win deepens). NEGATIVE PINNED: step-growth acceptance tests
  cannot detect corrector harm (gamma sweep: no setting rescues
  80bau3b's 62->105 explosion or osa_14's +10% wall while all degrade
  the friendly set) — corrector hostility is PATH-STABILITY, not
  economics; woodw <=30 / stocfor3 <=38 / cre_a unreachable by any
  global corrector setting on the cost-ratio axis. OPEN SCIENCE
  QUESTION for a dedicated unit: why does 80bau3b's trajectory shatter
  under one accepted recentering round? Answering it is the only path
  to woodw's force-1 0.102s (-21%).

- CRE PAIR UNIT (2026-07-09, exp-panel, merged): MEASURED-CLOSED for
  IPM factor economics at current machinery. Profile: every slice at
  floor (uplook 1.3ns/flop scalar-scatter rate, tail dpotrf 75 GF/s
  @4thr, t=1707/1654, ~1.6e9 flops); correctors already optimal (record
  CORRECTED: cre MCC rounds are useful — MCC=0 costs 62->71 / 64->76
  iters). Direction (a) coarse-stream supernodal REFUTED BY ARITHMETIC:
  best-case supernodal ~62ms vs row-wise 53ms/refactor even with a
  perfect coarse kernel (row tail runs at near-peak threaded BLAS —
  nothing to save). Direction (b) fewer-full-factors: naive lagged
  Jacobian fails cert/+59% iters; corrected inexact-Newton (fresh Cx,
  6 Richardson IR rounds, mu-windowed skips) TRACKS THE BASELINE PATH
  (60 iters vs 62/64) and cuts cre_b ~19% wall but fails exit
  certification even with skips stopped before the endgame. Probe kept
  dark (LINPROGX_IPM_LAG, default off, dead conditionals on default
  path). ENTRY POINT for a hardened build: per-step residual guards
  with refactor-and-redo (never accept an inexact step that violates
  the bound), CG instead of Richardson, exact endgame — the 60-iter
  path-tracking says the physics allows it; the exit says the guards
  must be real.

- INEXACT-NEWTON HARDENED (2026-07-12, exp-panel, merged; dark
  LINPROGX_IPM_LAG machinery kept as the experimental record):
  DIRECTION (b) CLOSED PERMANENTLY by the pre-registered criterion —
  100% redo storm (cre_b 0/15 accepts, cre_d 0/13, identical at eta
  0.1/0.3/1.0): six stale-preconditioned IR rounds never reach ladder
  accuracy; CG declined by arithmetic (pays only within ~4-5 rounds).
  The guard DID rescue certification at baseline iteration counts,
  which retro-explains the probe glimmer as genuinely bad steps
  tracking the path until the endgame broke. THE CRE PAIR IS NOW
  MEASURED-UNREACHABLE BY EVERY IPM-SIDE LEVER (factor floor,
  correctors optimal, ordering near-optimal, supernodal refuted by
  arithmetic, inexact Newton refuted by preconditioner quality); its
  3-5x gap belongs to the DS count program (crash reachable-set +
  pricing locality).

- DS RATE UNIT (2026-07-12, exp-leaving 4e9b8bc, ported to perf as
  c33f12f via 3-way patch — wholesale branch merge REJECTED: the fork
  base predates the EXPAND port, two-sided merge duplicated regions;
  patch-extraction is the pattern for exp-leaving -> perf transfers):
  byte-identical rate levers, +24-54% across the DS family. (1) scatter
  reuse: rcost update reads live alpha_scratch from the ratio-test
  scatter; (2) alpha_pattern support list: one unified workspace clear
  replaces 2-3 O(n) flag scans/pivot; (3) ratio-test candidate cache:
  flip-collect + sweep-2 argmax walk the ~10% admissible list in
  identical ascending order. Rejected: support qsort (+200us/pivot).
  Phase profiler (13 DS_TICK buckets, phase_us in result, <1%) ported.
  greenbea 382->292 us/pivot: paired 3.803->2.886s (-24%); woodw +30%,
  stocfor3 +42%, 80bau3b +54%, cre_d +41% (127.4s), pds_10 ~+30%.
  All byte-identical incl. x. greenbea remainder is 77% sparse-LU
  (refactor 33% + btran 17% + pivot-row 15% + ftran 12%): reaching
  <=1.5s needs an LU-engineering program (Forrest-Tomlin-style updates
  to stretch the ~51-pivot refac cadence without eta blowup, faster
  factorize kernel) or a count cut — count family documented exhausted.
  ALSO FIXED: explicit algorithm="dual_simplex" path was running
  expand=0 defaults (auto routes were correct).

- CORRECTOR-STABILITY SCIENCE (2026-07-12, exp-panel; +31 lines
  LINPROGX_IPM_TRACE debug instrumentation, inert off): the "80bau3b
  shatters under correctors" premise was WRONG — the corrector path is
  FASTER through every mu milestone (mu<1e-8 at it=50 vs OFF's 54)
  with healthy accepts throughout. The +43 iterations are an ENDGAME
  DETONATION: converged at it=52 (mu=3.96e-9), then at it=53 mu jumps
  60,000x to 2.286e-4 with the iterate frozen 4 iterations (steps
  exactly 0.0, centrality spread blown to 4e-6/4e3), then ~50
  re-convergence iterations. mccr=0 at the jump rules out the
  corrector itself; candidates are the exit chain (in-loop primal
  polish slack-weighted correction, min-norm dual cleanup 16-iter rate
  limit, NaN-direction best-iterate restore). CORRECTOR GATING BELOW
  RATIO 3.0 IS CLOSED (no accept-time signal exists — nothing wrong
  happens at accept time). THE MAPPED UNIT: endgame robustness
  bug-hunt — 80bau3b under MCC-1 would be ~53 iters (9 FASTER than
  today) if the endgame failed closed; repro
  LINPROGX_IPM_TRACE=1 LINPROGX_IPM_MCC=1 LINPROGX_IPM_MCC_RATIO=0.0001.

- ENDGAME DETONATION DIAGNOSED (2026-07-12, exp-panel trace merged):
  the mu explosion is an UNBOUNDED DUAL STEP AT BREAKDOWN MU, not the
  exit chain — at mu~4e-9 the affine direction degrades (it=52
  aff=0.000/0.531), the step ratio test blocks only NEGATIVE dz
  components, and full ad=1.0 steps on garbage directions inflate
  zl/zu (mu 3.96e-9 -> 2.29e-4). Exit-window timing decides who
  detonates: the corrector face reaches breakdown mu without the exit
  gates firing in the eligibility window; the OFF path exits first.
  Dual cleanups mutate only y/aty (cannot move mu); primal polish
  unverified but unlikely. SCOPED FIX (fresh-context unit): mu
  safeguard on the step — when mu < ~1e-7 (certificate window), reject
  any tentative step whose post-step mu exceeds ~10x pre-step mu
  (recompute is one cheap pass); shrink ap/ad or force an exit attempt
  instead. Deterministic, global. Acceptance: 80bau3b MCC-1 ~53 iters
  certified; then re-run the MCC matrix and re-derive the gate below
  3.0 (woodw 36->29, stocfor3 45->35, cre_a 36->28 become reachable).
  Check pilot87/osa tails for the same mu-jump signature.

- MU SAFEGUARD SHIPPED / CORRECTOR AXIS CLOSED (2026-07-12, exp-panel,
  merged): fail-closed step safeguard in the certificate window
  (mu<1e-7): tentative post-step mu computed before commit, <=3
  ap/ad halvings while post>10x pre, else skip the step and certify
  the converged iterate. Endgame detonation DISSOLVED: 80bau3b
  MCC-forced 105->52 certified (gap 3.17e-6, netlib-consistent);
  bit-identical reruns; INERT on every default path (0 fires,
  11-instance fingerprint sweep byte-identical, 249 green). Exit-gate
  discriminator confirmed: detonation strikes between mu-convergence
  and raw_pres convergence (5.26e-5 > feas_tol at it=52). GATE STAYS
  3.0: with the pathology gone, every ratio<3.0 instance still loses
  or ties wall under forced correctors (osa_14 +12%, 80bau3b +8% are
  the first casualties of any behavior-changing reduction; nothing
  sits in [2.0,3.0)). The hostile band was economics, not correctness.
  THE CORRECTOR/ITERATION-COUNT AXIS IS FULLY MAPPED AND CLOSED:
  safeguard for robustness, 3.0 for economics, matrix on record.

- LU PROGRAM STEP 1 (2026-07-13, exp-leaving 0feb107+4046675+test,
  ported to perf): CADENCE ECONOMICS MEASURED. Fill guard 4x is
  CORRECTLY TUNED (6x: +18% iters, 8x/12x also net-lose — eta solve
  growth beats refac amortization; curve on record). THE WIN: the
  diag-ratio stability guard (1e6) fired on harmless pivot-magnitude
  spread = 47% of greenbea's refactorizations; default now 1e8 —
  greenbea -12.6% paired wall (~2.5s projected clean from 2.89),
  certified/deterministic, 80bau3b -2.3% iters, others path-identical.
  Cycle fixture routing consequence: DS-early now SOLVES cycle
  directly (resid 3.6e-12, 913 pivots) where it previously failed into
  the IPM rescue — test updated to assert the accuracy contract.
  NEGATIVES PINNED: fused pivot-search walks (-11%, L1-hot re-reads);
  cadence stretching beyond diag=1e8 (fill curve all-negative).
  STEP 2 TARGETS (both point at the same machinery): Forrest-Tomlin
  updates (U factored, no eta chains -> flat solves + rarer refacs;
  the remaining 90us/pivot btran/ftran fattens ~600 nnz/update) and a
  Suhl-style pivot-search rule (85% of factorize is candidate-volume
  at the 4ns/entry memory floor). greenbea trajectory: 3.80 (session
  start) -> 2.89 (rate unit) -> ~2.5 (this unit); <=1.5 needs step 2.

- OFFICIAL SUITE RE-BASELINE (2026-07-13, quiet, build 03dc77c —
  thirteen ships in): aggregate 63.4s vs HiGHS 168s+qap15-timeout;
  coverage 24/24 vs 23/24. Single-shot head-to-head ~12W/3-knife-edge
  (degen3 1.05x, stocfor3 1.14x, osa_14 1.30x-with-paired-win-history)
  /9L. HARD LOSSES by ratio: greenbea 10x (2.64 vs 0.26; FT program
  queued), cre_d 5.6x / cre_b 3.3x (IPM side closed; DS count program
  is the only path), pds_10 2.3x / pds_20 1.4x (PDHG route — UNTOUCHED
  ALL SESSION, no unit has ever profiled it), woodw 2x (flop-bound,
  rate+ordering+correctors all closed — needs a different idea),
  maros_r7 1.6x (serial floor reached; parallel refuted; iteration
  count optimal — parked), 80bau3b ~1.5x here / ~1.1x paired-quiet,
  cre_a 1.3x (no dominant slice). Note single-shot suite variance vs
  the paired protocol: flips are certified ONLY by paired 5-run
  interleaved; the suite is the coverage/aggregate record.

- LU STEP 2 — TRUE FORREST-TOMLIN (2026-07-13, exp-leaving, dark
  LINPROGX_DS_FT=1, NOT yet ported to perf — port when the program
  ships): the existing "FT etas" were PFI; true FT built (~600 lines:
  L fixed, U kept triangular via time-stamped deletions + spikes +
  row-eta file; spike-diagonal stability reject -> refactor, RARE
  everywhere — the closing negative did not materialize). PROVEN:
  cre_d 210->102s (-51%), woodw -19%, chain growth eliminated
  (btran 205->60us/pivot on cre_d), refac cadence 51->277 on greenbea.
  BLOCKERS: (1) prototype solves fall back to dense staging (O(m)
  scans) losing Gilbert-Peierls hyper-sparsity -> stocfor3 +44%,
  80bau3b +35%; (2) greenbea +21.5% iteration drift (FP path luck at
  EXPAND ties; objective unchanged) breaches the 10% gate. New cadence
  census under FT: diag_ratio-limited (31/40), fill guard stops
  binding; cre_d pinned on the n_updates>=500 hard cap. STEP 3:
  (a) hyper-sparse FT solves (virtualize U' adjacency inside the
  Gilbert-Peierls reach), (b) greenbea drift study (unlucky ties vs
  systematic), (c) update-cap knob (FT flat solves make higher caps
  safe). greenbea trajectory: 2.89 -> ~2.5 (step 1) -> ~2.2 projected
  drift-neutral FT -> ~1.9 with (a); <=1.5 needs the btran/scatter
  slices too.

- PDHG UNIT (2026-07-13, exp-panel, merged): first-ever PDHG profile.
  The implementation is PDLP-COMPLETE already (Ruiz+l2 equilibration,
  Malitsky-Pock adaptive steps, adaptive omega, KKT-based restarts,
  averaging, CGLS cleanup); convergence smooth, no pathology; loop is
  memory-bandwidth-bound (391us/iter, 85% dense step-trial passes,
  ~15 GB/s at 4 threads = practical floor). SHIPPED: threads=0 auto =
  physical cores (logical/2, clamp [2,8]); DEFAULT_PDHG_THREADS 4->0.
  Bit-identical trajectories: pds_10 -21.5% (2.78s vs HiGHS 1.42 =
  1.9x), pds_20 -36.6% (15.8 vs 13.24 = 1.2x), qap12 -2.2%, qap15
  -9.1%. NEGATIVES PINNED DARK: LINPROGX_PDHG_HALPERN (anchored +55%
  iters; reflected divergent under adaptive steps — a fixed-step
  reflected-Halpern retune is a PROGRAM, not a unit),
  adaptive_weight=2 (iteration_limit). VERDICT: pds gap is pure
  iteration count at 2e-5 (8.5k/21.7k iters); per-iteration at floor.
  Remaining pds paths: fixed-step Halpern program, or DS
  pricing-locality program. Side notes: qap12 is CGLS-dominated
  (1.41/1.70s); PDHG debug=1 costs 10x.

- KNIFE-EDGE CERTIFICATION (2026-07-13, exp-panel, merged; paired
  7-run protocol, the honest scoreboard): degen3 WIN 0.80 (7/7);
  osa_14 TIE 1.00 (4/7); LOSSES stocfor3 1.07->1.03 after this unit
  (2/7 wins, mins 576 vs 580), 80bau3b 1.13->1.07 (mins dead-tied
  178.0/178.1), cre_a 1.23->1.22. SHIPPED: mu-gated round-2 IR
  (bit-exact no-op while mu>1e-5, sha256-verified on x AND y across
  8 instances; ~1% wall) + BUGFIX: supernodal kwarg 'p'-parse coerced
  -1(auto) to forced-on (now 'i'). MEASURED DEAD for this band:
  wrapper overhead (~5%), tail-boundary move (+6 iters), forced
  supernodal on cre_a (-14ms but +5 iters and per-problem). The band's
  remaining sized levers are PROGRAM-scale: uplook-prefix BLAS-ing
  (cre_a 22ms ceiling; cre_b 1.95s uplook is the same lever at 100x
  the payoff) and centering/iteration-count work.

- LU STEP 3 (2026-07-13, exp-leaving, still dark): hyper-sparse FT
  solves LANDED (stocfor3 -9% / woodw -17% / cre_d -14% / 80bau3b
  neutral — dense-staging losses erased); greenbea drift ROOT-CAUSED
  (row-eta |w| up to 1.1e8 on M-boxed columns -> classic growth
  rejection wmax>1e6, refactor instead; cures doses 1e-11/2e-11/5e-11;
  5e-12 still fragile) but at the DEFAULT dose drift is +25.2%/+10%
  wall = flip blocked. KEY COMPOSITION: FT + dtau=5e-11 is greenbea's
  BEST EVER (optimal 10,391 iters / 2.29s; drift vs FT-off-at-dose
  +2.6%; FT-off also improves at 5e-11: 10,125). Cap knob dead (500
  optimal; EXPAND tau cap binds ~950 updates). FT_CHECK/FT_DENSE_U
  diagnostics dark; ftran_pattern rhs-aliasing contract documented
  (a 'divergence' was the checker reading clobbered rhs). NEXT UNIT
  (the flip package): gate FT=1 + dtau=5e-11 combined default through
  the FULL battery (cre_d's 103,781 anchor was AT 1e-11 — the dose
  change moves every DS trajectory and needs its own certification),
  plus wmax cap sweep (1e5/3e5/1e6) for the 5e-12-class fragility.

- UPLOOK UNIT 1 — PATTERN CACHE (2026-07-13, exp-panel, merged,
  bit-exact): elimination patterns (chol_ereach output) were re-walked
  EVERY refactor; now recorded at symbolic time (rpat_ptr/rpat, zero
  extra walks) and replayed, emark memset dropped. 15-25% of uplook
  was walk overhead. Paired: cre_b uplook -14% (refactor -8..-12%,
  ~300-450ms), cre_a -7.7%, 80bau3b -8.2%, degen3 -6.8%, pilot87
  -7.1%. NEGATIVES PINNED: software prefetch (x cache-resident),
  rpat_lp precompute (3x pattern bytes). Remaining uplook is scatter
  flops at ~2ns/pair: NEXT is the block-row up-looking kernel (2-4
  rows share Li/Lx streams with per-row accumulators — bit-exactness
  preservable). Knife-edge re-certification needs QUIET load first.

- LU STEP 4 — FLIP VERDICT: NO-FLIP AS KWARG DEFAULT (2026-07-13,
  exp-leaving; wmax 3e5 committed as the one adopted change — the only
  cap certifying at all three doses, fires only on greenbea's poison
  etas). Battery at (FT=1, dtau=5e-11, wmax=3e5), all certified:
  DIRECT-route paired wins greenbea -13.1% (10,665 it / 2.327s),
  stocfor3 -11.2%, cre_d -10.2% (146.9s), woodw -7.9%; but 80bau3b
  +9.1% (DS-direct only — publicly IPM-routed, NOT scoreboard
  relevant) and PUBLIC-auto greenbea +7.7% REGRESSION (12,780 it /
  3.336s, resid loosened 7.1e-7). ROOT CONFIG MISMATCH FOUND: the
  sparse.py DS routes pass NO tol (C default ~1e-8) while every
  certified FT number is at tol=1e-11. FOLLOW-UP RUNNING: route-config
  grid (tol x FT x dtau) on public greenbea + cycle — if FT wins at
  the matched config, FT ships as a ROUTE setting rather than a kwarg
  default.

- FT PROGRAM SHIPPED (2026-07-13, exp-leaving 8a1fedf ported to perf):
  route-config grid killed the tol hypothesis (tol irrelevant; harmful
  at expand=0+FT) and found the real lever stack: expand=1 (routes
  already had it on perf) + FT default ON + dtau default 5e-11 + wmax
  3e5. VERIFIED ON CANONICAL: greenbea public 2.361s certified
  (10,665 iters, resid 6.1e-8; was 3.10 pre-ship, 3.80 at session
  start), cycle 844 pivots / 3.6e-12 (better than the 913 baseline),
  direct battery woodw -21% / stocfor3 -30% / cre_d -11% (129s), 249
  green, deterministic. The LU program (steps 1-4) is COMPLETE: PFI
  chains eliminated as the DS bottleneck. greenbea remainder vs HiGHS
  0.26: ~9x — next levers are the Suhl-style pivot-search rule
  (factorize kernel volume) and anything that cuts the 10.6k count.

- BLOCK-ROW UPLOOK SHIPPED (2026-07-13, exp-panel, merged 2f4a1df):
  block kernel (<=4 consecutive rows, per-row accumulators, shared
  Li/Lx stream) behind a SAVEABLE-FRACTION gate — symbolic census of
  scatter pairs amortizable at b=4, engage when >=0.5 (clean gap:
  cre_a 31.7 / woodw 36.6 / 80bau3b 44.6 / pilot87 45.5 OFF;
  stocfor3 56.4 / cre_b 57.8 / degen3 65.8 / osa_14 70.1 / maros_r7
  74.4 ON). Gated-ON outcome-exact (obj reldiff <=2.8e-12, iterations
  identical; ascending vs topological order), gated-OFF byte-identical.
  Paired: cre_b wall -12.8% (~4.96s from 5.69; uplook 1.72->1.13),
  osa_14 -4.1% (certified-TIE may flip — re-certify at quiet load),
  cre_a regression eliminated. 249 green. The uplook program's two
  units (pattern cache + block kernel) have now cut cre_b's uplook
  ~1.95 -> ~1.13s; next lever there per profile is the dense-tail
  dpotrf (t=1707) or iteration count (correctors already optimal).

- SUHL PIVOT SEARCH SHIPPED (2026-07-13, exp-leaving, ported):
  bounded Markowitz search (budget 8 threshold-viable columns, accept
  merit<=4 immediately, exact merit==0 exit preserved; default on,
  LINPROGX_LU_SUHL=0 reverts). greenbea pivot search 0.322->0.013s
  (-96%, 44x fewer visits); paired walls greenbea +35.4% (compounded
  by favorable -14% iteration shift), woodw +20.3%, stocfor3 +12.6%,
  cre_d +7.9%, 80bau3b +8.1%; zero refac storms; all certified.
  VERIFIED ON CANONICAL: greenbea public 1.650s best (9,150 iters,
  resid 1.3e-7) — session arc 3.80 -> 1.65s, now ~6.3x vs HiGHS 0.26.
  MONITORED EXCEPTION: cre_d same-basis fill +40% (proxy breach;
  outcomes healthy). QUEUED IDEA if it ever bites: fill-feedback
  fallback (redo factorize exhaustively when fill exceeds ~1.5x
  prediction).

- CLEAN-BOX BENCHMARKING (2026-07-13, tools/modal_bench.py): Modal
  4-CPU dedicated containers, loadavg 0.00, ~\$0.30/full run. FIRST
  CLEAN-BOX CERTIFICATION at 7e9947a: osa_14 WIN 0.96 (6/7) and cre_a
  WIN 0.97 (5/7) — both FLIP from local losses; degen3 flips the OTHER
  way (local WIN 0.80, Modal 1.06) and pilot87 similarly — THE
  KNIFE-EDGE BAND IS MACHINE-DEPENDENT; the controlled box is now the
  scoreboard of record. Suite geomean 0.735; clean-box loss ladder:
  greenbea 4.8-5.25, cre_d 4.63, pds_10 2.83, cre_b 2.75-2.83,
  pds_20 1.63, woodw 1.44, maros_r7 1.24, 80bau3b 1.20-1.27,
  stocfor3 1.06-1.15, degen3 1.06, pilot87 1.11. Volumes:
  linprogx-lpsuite (fixtures), linprogx-src (snapshots by sha).

- SEVENTEENTH SETTLED — BFRT POST-FT/SUHL (2026-07-13, codex probe,
  experiments/bfrt_probe.py + probe_out/bfrt-probe.json): KILL by
  pre-registered criteria. greenbea 9,150 -> 9,865 pivots (+7.8%,
  wall 1.65 -> 2.60s) with bfrt=1 — the fresh economics do NOT
  rehabilitate BFRT on the target. Numerics clean everywhere
  (residuals <=2.2e-7, obj deltas <=2.3e-12). ANOMALY FOR THE RECORD:
  80bau3b DS walk -40% pivots / -36% wall under BFRT (6,560 -> 3,957)
  — but its public route is IPM (0.30s < DS 0.363s), so no scoreboard
  relevance; the boxedness signal does NOT discriminate (greenbea is
  M-boxed-heavy and regressed). BFRT stays off the routes.

- EIGHTEENTH SETTLED — IPM->DS CROSSOVER BASIS (2026-07-13, codex
  probe on exp-panel: LINPROGX_IPM_CROSSOVER_TRACE snapshots +
  experiments/crossover_basis_probe.py): KILL 0/4 (bar was 2/4).
  Tapia-ranked deterministic structural matching achieves FULL
  structural coverage at every mu crossing (1e-4/1e-6/1e-8) on
  woodw/stocfor3/80bau3b/cre_d — and EVERY candidate basis is
  numerically SINGULAR at the Markowitz LU gate (singular at steps
  588-14,433) before cleanup metrics can even be computed. The
  degenerate optimal faces do not admit cheap crossover bases; a
  rank-revealing crash-factorization construction would be the only
  escalation and its prior is now much lower. The crossover program
  is closed at probe level.

- NINETEENTH SETTLED — FIXED-STEP HALPERN PDHG (2026-07-13, codex
  probe on exp-leaving, probe_out/pds10-halpern-slope.json incl.
  extension_10k): KILL, decisively. The 2k-window terminal-KKT
  advantage of reset-anchor cells was an artifact (baseline KKT
  descent is restart-lumpy early); at 10k iterations the adaptive
  baseline converges (1.02e-8 at equal 20.6k operator passes, optimal
  at endpoint) while the best Halpern cells sit at ~1e-3 — ~98,000x
  worse at equal work, late slopes 1.5-1.8% of baseline. All Halpern
  variants (anchored/reflected x anchor policies x fixed etas) are now
  dead for pds. METHOD LESSON: slope probes must span restart-scale
  dynamics; 2k was too short. NEXT PDS IDEA (prior improved by this
  data): PDLP-style DIAGONAL step sizes (current eta is scalar
  0.99/||A||) — the baseline's late acceleration shows mechanics are
  healthy; worst-column throttling is the remaining geometry problem.

- BLOCK-GATE THRESHOLD VALIDATED (2026-07-13, codex probe,
  probe_out/blockgate-probe.json): the near-threshold gated-OFF
  instances (80bau3b 44.6%, pilot87 45.5%, plus cre_a/woodw) all
  KEEP_OFF under forced block4 with 9 interleaved trials (80bau3b
  -5.5% when forced, pilot87 +0.2% neutral; iterations identical).
  The 0.5 saveable-fraction constant stands; anomaly-miner item 2
  closed.

- TWENTIETH SETTLED — PDLP DIAGONAL STEPS (2026-07-13, codex probe on
  exp-leaving, probe_out/diag-steps.json): KILL. Post-Ruiz alpha=1
  row/col L1 diagonal steps (adaptation disabled per Pock-Chambolle)
  stall far above tolerance on pds_10/pds_20 and DESTROY the qap12 win
  (misses 1e-3 where baseline reaches 2e-5 at 8.3k passes; equal-work
  terminal KKT 3-5 orders worse everywhere). Ruiz had already banked
  the diagonal-geometry win; the adaptive eta/omega machinery IS the
  strength. THE PDS PAIR IS CLOSED AT UNIT LEVEL: PDHG per-iteration
  at bandwidth floor, threads shipped, restarts/adaptivity healthy and
  superior to every fixed-geometry variant tried (Halpern x3, diag).
  Remaining pds gap is program-scale or accepted.

- TWENTY-FIRST SETTLED — EXACT DSE POST-FT/SUHL (2026-07-13, codex
  recheck, probe_out/dse-recheck.json): KILL, refutation refreshed at
  current economics. Count cuts real but small (greenbea 9,150->8,474
  = -7.4%; 80bau3b -14.5%) while pricing_update becomes a 12-28% phase
  bucket (extra solve + carried weights) — wall regresses everywhere
  (greenbea +4%, woodw +75%, stocfor3 +43%); cre_d DETONATES
  (iteration_limit at 300k, certificate broken — carried-weight drift
  x EXPAND interaction). Devex stays. The stale-conditions audit's
  top recheck is closed; its remaining items (LU cadence grid
  post-Suhl, router/gate threshold matrix) stay queued at low
  priority.

- DANTZIG ROUTE SHIP (2026-07-13, codex battery): plain-Dantzig
  leaving (leaving_rule=1) enabled on the DS auto-rescue routes only.
  greenbea public 1.66 -> 0.83s (-50%, 6,533 pivots); cycle improves
  (905 -> 676, resid 3.6e-12); 249 green; deterministic. Formal
  kwarg-default probe was KILL (cre_d +41% direct-DS) but the route
  population is {greenbea-shaped, cycle} — route-level config is the
  correct ship shape (same class as expand=1). greenbea session arc:
  3.80 -> 0.83s; clean-box ratio projection ~2.6x from ~5x.

- PRESOLVE V2 SHIPPED (2026-07-14, codex 3-round build from the
  fresh-eyes census): THE CAMPAIGN'S BIGGEST SINGLE UNIT. Column
  singletons (free/implied-free, chained), fixed columns, dual fixing,
  row forcing, duplicate-column merge — fill-guarded + native O(nnz)
  opportunity gate (zero-cost when nothing qualifies).
  HAND-VERIFIED: cre_d 1.279s best (was 4.9-6.2 = 4.6x loss ->
  ~1.15-1.3x), cre_b 1.69s (vs HiGHS ~2.0 = LIKELY FLIPS TO WIN),
  greenbea 0.466s (session arc 3.80 -> 0.47 = 8x; ratio ~1.6x from
  14x at session start), woodw/maros/stocfor3/80bau3b all improve.
  The "cre pair closed on every IPM axis" verdict was TRUE AT FIXED
  PROBLEM SIZE — the presolve layer moved the size. Realized
  reductions exceeded census projections on the big four. 260 tests,
  LINPROGX_PRESOLVE_V2=0 reverts. NEXT: clean-box re-certification of
  the whole board + chronicle update.

- CLEAN-BOX CERTIFICATION AT 1f4351d (2026-07-14, AWS us-west-2,
  loadavg 0.00, assets/modal_bench_1f4351dcfa96_{suite,paired}.json):
  13W-11L; geomean 0.558 (was 0.735); aggregate 49.1s vs 192.8s+
  qap15-timeout; coverage 24/24 vs 23/24. FLIP: cre_b LOSS 2.75 ->
  WIN 0.940 (6/7, IPM 2.42 vs 2.57 — presolve V2's headline).
  cre_a flipped back to loss (1.066, knife-edge churn). THE LADDER
  COLLAPSED — remaining losses, none above 2x: greenbea 1.89 (was
  5.25; 0.69s via Dantzig route), woodw 1.69, pds_10 1.66 (was 2.83),
  cre_d 1.53 (was 4.63), 80bau3b 1.32 (REGRESSION from 1.20 paired —
  investigate: local A/B showed -9%; clean box disagrees), maros_r7
  1.17, degen3 1.08, cre_a 1.07, stocfor3 1.06, pds_20 1.03
  (near-tie, was 1.63), pilot87 1.008 (dead knife-edge, 3/7).
  Session totals: 20+ ships, 21 settled hypotheses, geomean nearly
  halved in two days.

- GREENBEA POST-V2 ANATOMY (2026-07-15, codex): cadence constants
  VALIDATED at the new size (diag/fill grid flat ~0.42s local, refac
  ~5.6us/pivot — the stale-conditions cadence recheck closes
  still-valid); Dantzig is the leaving-family optimum (4,399 vs Devex
  6,807, rule2 11,948, rule3 15,188, rule4 fails); count not obviously
  cuttable (near-zero degeneracy). LIVE LEVER: column-side rate slice
  ~42us/pivot (pivot-row + ratio + rcost = largest coherent block).
  Queued unit: support-characteristic histograms (rho_nnz, alpha_nnz,
  candidate count, reuse windows) then a targeted prototype; kill if
  ceiling <20us/pivot or prototype <10% wall. Also queued: restore
  top1pct_progress_frac for the current route (the anatomy instrument
  lives on exp-leaving, pre-V2).

- POST-V2 IPM RE-PROFILE (2026-07-15, codex): (1) 80bau3b "regression"
  RESOLVED AS ENVIRONMENT DRIFT — V2 does not fire there
  (worth_python_pass=False; V2-on/off identical shapes and walls);
  clean-box 1.32 vs 1.20 was variance, not code. (2) THE NEW BIGGEST
  EXPOSED SLICE: presolve V2's own pass cost — cre_d 0.412s of 1.38s
  wall, maros_r7 0.228s (unit running: native fast path, kill unless
  >=0.25s/0.15s saved with bit-identical reduced problems). (3)
  maros_r7 slid BELOW the MCC 3.0 gate at its new size (ratio
  3.62->2.54): forced MCC gives 20->17 iters / -9.6% wall, but global
  lowering stays dead (cre_d 71->60 iters yet +56% wall) — a
  structural-guard unit is queued behind the fast path. Gate-position
  audit otherwise clean: tail/block-row/supernodal decisions all on
  correct sides at the new sizes.

- TWENTY-SECOND SETTLED — PRESOLVE PASS FAST-PATH (2026-07-15, codex,
  honest revert): dirty-row scans, activity caches, native
  to_component_bytes and rebuild-batching recovered only cre_d +0.121s
  / maros +0.109s of the 0.25/0.15 bar; public-route walls moved
  within noise. The V2 pass cost (~0.4s on cre_d) is mostly inherent
  Python-side reduction work; a FULL native V2 port remains the only
  escalation (L-size, queued at low priority). Reduced-problem
  bit-equality held throughout.

- COLUMN-RATE UNIT CAVEAT (2026-07-15): the exp-panel prototype's -11
  to -15% was measured on the PRE-RATE-SHIP branch and its REUSE_ALPHA
  mechanism likely duplicates shipped c33f12f (verification worker
  running). NEW DATUM that survives regardless: consecutive-pivot
  support overlap 96-99.97% (rho p50 1.8-2.2k, alpha p50 6.5-58k) —
  cross-pivot incremental support maintenance is unshipped and under
  algebra review. METHOD LESSON: kernel-slice experiments must run on
  the current stack; stale worktrees rediscover shipped levers.

- TWENTY-THIRD SETTLED — CROSS-PIVOT SUPPORT REUSE (2026-07-15, codex
  verification): the exp-panel REUSE_ALPHA prototype is REDUNDANT with
  shipped c33f12f (alpha_scratch stays live through 4g on this
  branch). The cross-pivot lever is KILLED ON ALGEBRA: after basis
  change B'=BE, alpha'_k = alpha_k - (d_k/d_p) alpha_p needs the OLD
  alpha_k for arbitrary k — i.e., the very BTRAN+scatter being
  avoided; only k==p is cheap. Full-tableau maintenance is not viable.
  Histograms ported (LINPROGX_DS_RATE_HIST): current-stack pivot_row
  phase is 25-41% of DS wall with candidates p50 182-5k and overlap
  99.9% — informative but the reuse family is now closed both
  within-pivot (shipped) and cross-pivot (impossible).

- IPM OTHER-SLICE ANATOMY (2026-07-15, codex, experiments/
  ipm_other_profile.py): the unattributed 18-40% decomposes as
  setup/order (degen3 26%, cre_a 21% — BOTH LARGER THAN THEIR LOSS
  MARGINS; unit running), scaling+init (80bau3b 12%, degen3 10%),
  loop-misc residual/gap/step bookkeeping (stocfor3 12%, woodw 11%).
  Ruled out: MCC (budget 0 on all five), mu-safeguard (0 events), exit
  polish (only cre_a, 1 round), tail dtrsv (disabling slower). Queued
  after setup unit: loop-misc timers + certificate-eval windowing;
  Ruiz pass-count probe. 15-PAIR PRECISION (Modal): pilot87 0/15,
  stocfor3 0/15, degen3 0/15, cre_a 4/15 — the knife-edges are REAL
  small losses, not variance; they need genuine 2-8% wins.

- HOST-CONDITIONAL MARGINS (2026-07-16, 15-pair Modal precision,
  /tmp/knife_chunk{A,B}.json): the knife-edge verdicts depend on host
  class. vs the us-west-2 certification: pilot87 1.008 -> 1.223 on
  Azure-Asia (HiGHS gains 4.63->2.98s; we gain less), degen3 1.075 ->
  1.297, stocfor3 1.06 -> 1.111, pds_20 1.03 -> 1.878 (our
  bandwidth-bound PDHG is dramatically host-sensitive: 14.0 -> 20.0s
  while HiGHS moved 13.7 -> 10.6), cre_a 1.066 -> 1.036 (the one TRUE
  knife-edge: 4.5ms / 3.5% from median parity; min-vs-min already
  0.969). None are coin flips at 15 pairs (0/15 x4, 4/15 cre_a).
  DOCTRINE: certifications must pin the host class — the harness
  currently doesn't set a Modal region; add region pinning and
  standardize on one region for the scoreboard of record. Also a
  finding in itself: PDHG's memory-bandwidth sensitivity is ~2x
  HiGHS's simplex on the same hardware swing.

- SETUP FAST-PATH SHIPPED (2026-07-16, codex): bucketed min-degree
  queue + exact preallocation + fused compaction, EXACT-OUTPUT
  (fingerprint hook verifies perm/Lp/Li/snode identity; heap fallback
  knob LINPROGX_MD_QUEUE). Setup: cre_d -46%, maros -31%, degen3 -11%,
  cre_a -7%; public walls improve family-wide. The -40% degen3/cre_a
  target MISSED — min-degree itself (25.6ms/15.1ms there) remains the
  binding slice; a deeper min-degree unit (AMD-style approximate
  external degree with exact-output verification impossible — would
  change orderings, so it becomes outcome-gated) is the escalation,
  queued. cre_a remains 4.5ms from parity (us-west); degen3 needs
  ~55ms on the harsh host class.

- PINNED-REGION CERTIFICATION (2026-07-16, us-west, post setup-ship,
  assets/modal_bench_<head>_{suite,paired}.json): **15W-9L**. FLIPS:
  pilot87 WIN 0.809 suite / 0.855 paired 9/9 (the setup ship + fair
  host); pds_20 WIN 0.813 (us-west PDHG-fair vs Azure-Asia harsh);
  degen3 paired 0.994 (6/9) / suite 1.034 — AT PARITY; cre_a 1.002
  (7/9 wins!) and stocfor3 0.992 (3/9) — AT PARITY, mixed indicators.
  Deepened: osa_14 0.79 (9/9), cre_b 0.83 (9/9). REMAINING REAL
  LOSSES: greenbea 1.89, woodw 1.60, cre_d 1.46, 80bau3b 1.36,
  pds_10 1.31, maros_r7 1.19, + three parity coin-flips. Session:
  12W -> 15W with the whole ladder under 2x.

- LOOP-MISC AXIS CLOSED (2026-07-16): attribution shipped
  (residual matvecs + best-iterate copies + RHS assembly are the
  slice; mu-safeguard confirmed gated); best exact lever (tall-only
  serial matvec) is worth only 1-5% — target missed, honestly scored.
  The parity trio's fate rides on re-certification + the native-V2
  presolve port (NOW COMMISSIONED: cre_d pass cost ~0.4s of 2.15s
  wall = its clearest single lever; cre_b/maros/greenbea share it).

- POST-NATIVE-PORT PAIRED CERTIFICATION (2026-07-16, us-west, 9
  pairs): THREE MORE FLIPS — maros_r7 WIN 9/9 (0.733), stocfor3 WIN
  9/9 (0.854), cre_a WIN 6/9 (0.896). At parity: degen3 4/9 (0.993),
  woodw 6/9 wins at ratio 1.022 (was 1.60 — the native port's
  biggest mover). Narrowed: cre_d 1.235 (0/9), 80bau3b 1.225 (0/9).
  MEASUREMENT DOCTRINE UPDATE: the same-region single-shot suite
  contradicts certified paired verdicts (osa cells 1.38 vs 9/9-paired
  0.79-0.93; pds_20 0.81 vs 1.36 across two us-west runs) — pinned
  regions still mix host generations; SINGLE-SHOT IS NOT
  SCOREBOARD-GRADE at sub-40% margins. Paired-only board from here.
  PAIRED-CERTIFIED TALLY (paired verdicts + coverage): ~17W with
  degen3/woodw at coin-flip parity and greenbea/pds_10/pds_20/cre_d/
  80bau3b the remaining paired losses (pds pair needs paired
  re-measurement — host bandwidth sensitivity).

- CANONICAL BOARD AT 957347b-era BUILD (2026-07-16, AWS us-west-2
  pinned cloud+region, load 0.00, assets/pin4_chunk{1,2}.json):
  **14W-5L-4P (+qap15 coverage = 15 wins)**. NEW FLIPS THIS WAVE:
  cre_d WIN 8/9 (0.957 — the native-V2-port flip), degen3 WIN 9/9
  (0.823), pds_20 WIN 9/9 (0.826), osa_30 WIN 8/9. PARITY (0.97-1.03):
  80bau3b, stocfor3, woodw (0.996!), pilot87 (0.995). LOSSES: greenbea
  1.69, osa_60 1.50, osa_14 1.34, pds_10 1.20, cre_a 4/7-at-0.966
  (wins-bar miss). OSA SWING INVESTIGATED, NOT A REGRESSION: bucket
  min-degree == heap fallback (same iters/walls), matvec gate inert,
  presolve bit-equality held; the 0.95->1.34 movement is WITHIN-REGION
  INSTANCE-TYPE VARIANCE (HiGHS -29%, lx +10% across two load-0
  containers, two days apart, same region) — Modal does not expose
  instance-type pinning. DOCTRINE: bandwidth-sensitive verdicts
  (osa/pds/pilot87/woodw/greenbea) are host-hardware-conditional;
  a robust all-24 claim needs medians across multiple containers per
  certification (protocol v3, queued). The stable core (kens, qaps,
  truss, fit2p, d2q06c, cre_b 0.62, maros 0.77, degen3, cre_d,
  stocfor3~, cre_a~) holds across every host observed.

- PROTOCOL V3 SHIPPED + FIRST V3 CERTIFICATION (2026-07-16, c344177,
  codex build + orchestrator cert): modal_bench.py --hosts N runs the
  full paired protocol in N concurrent containers (starmap) and
  aggregates median-of-hosts verdicts with per-host spread preserved;
  --hosts 1 byte-identical to before; aggregation unit-tested. FIRST
  V3 CERT (3 hosts x 7 pairs, AWS us-west-2, bandwidth-sensitive set,
  assets/modal_bench_c34417761bb6_paired_hosts3.json): pilot87
  **WIN 0.826** [0.813,0.939] 21/21 — UPGRADES from 0.995 parity;
  pds_20 WIN 0.824 [0.788,0.830] 20/21 confirmed; woodw 1.014
  [0.991,1.106] 8/21 — a TRUE knife-edge, hosts disagree; greenbea
  1.695 [1.676,1.713] 0/21 rock-solid; osa_14 1.424 [1.406,1.662],
  osa_60 1.290 [1.287,1.370], pds_10 1.258 [1.251,1.279] all 0/21 —
  REAL losses on every host this wave; the host-conditional framing
  for osa/pds_10 does NOT rescue them under v3. PROVISIONAL V3 BOARD:
  **16W-3P-5L** (pilot87 parity->win; parity woodw/80bau3b/stocfor3;
  losses greenbea/osa_60/osa_14/pds_10/cre_a). DOCTRINE: v3
  median-of-hosts is now the certification standard for the
  bandwidth-sensitive set.

- TWENTY-FOURTH SETTLED — AMD APPROXIMATE DEGREE (2026-07-16, codex,
  two attempts, kill final): approximate external degree cannot flip
  cre_a. Attempt 1 (loose bound): cre_a slice -46% but nnz(L)
  +5-11% / flops +25-41% across the family — quality gate fail.
  Attempt 2 (proper Amestoy-Davis-Duff bound + element absorption +
  exact-when-<=2-elements): quality now GOOD on degen3/cre_d
  (slice -21%/-24%, nnz -0.7%/-0.4%, flops improve) but cre_a itself
  only -9.6% slice (bar: -30%) with flops +7.1% — the target instance
  is the one whose ordering the approximation hurts. Probe stays
  behind default-off LINPROGX_MD_APPROX in the discarded worktree;
  NOT shipped. cre_a's remaining queued levers: Ruiz pass-count,
  certificate-eval windowing, and a v3 re-cert of the knife-edge set
  once the greenbea DS ship lands (woodw/80bau3b/stocfor3 are
  DS-family — the dense-U route may move the whole parity trio).

- TWENTY-FIFTH SETTLED — DS FTRAN DENSE-U SWEEP (2026-07-16, codex,
  honest gate-fail): the pre-existing LINPROGX_DS_FT_DENSE_U bisect
  probe measured -16.1% on greenbea under a LOADED box (three worker
  benches running) but only -0.21% on the quiet box; the adaptive
  productionization (20-25% FTRAN density band; greenbea mean density
  24.1%) was removed after failing its own >=12% gate. BTRAN dense
  sweep separately dead: regresses (~0.423->0.436s) AND perturbs the
  pivot sequence (4399->4420). METHOD LESSON: alternating A/B is
  robust to load drift but NOT to load-dependent RELATIVE effects —
  bandwidth-hungry candidate paths must be gated on a quiet box, same
  doctrine as replay. LIVE RESIDUE: the relative gain under bandwidth
  contention was real, and Modal cert hosts are bandwidth-tight
  (greenbea's 1.69 loss lives there) — an ON-HOST knob A/B is queued
  (harness needs an env-override A/B mode). SOLVE_SLICE per-solve
  timing instrumentation shipped (LINPROGX_DS_SOLVE_SLICE,
  default-off).

- TWENTY-SIXTH SETTLED — CRE_A RUIZ + CERT-EVAL WINDOWING
  (2026-07-16, codex): both remaining queued IPM shave levers killed.
  Ruiz early-exit: every board IPM instance runs exactly 10 passes;
  tol 0.05 gives cre_a -3.7% but CHANGES OBJECTIVES (equilibration is
  numerics-active, not overhead) and regresses cre_d +3.4%; tol 0.01
  helps nothing (1.018). Cert-eval windowing: measured ceiling
  0.2-1.2ms on cre_a vs a ~2.7ms (3%) bar — killed on ceiling; the
  window attempt measured 1.0034. cre_a's engineered-shave queue is
  now EMPTY (min-degree/AMD, presolve pass cost, setup fast-path,
  loop-misc, Ruiz, cert-eval all closed). Next: v3 knife-edge re-cert
  (cre_a/woodw/80bau3b/stocfor3); if cre_a still misses the wins bar,
  it needs a fresh-eyes hypothesis census (the pattern that found
  presolve V2), not another shave.

- V3 KNIFE-EDGE CERTIFICATION (2026-07-16, 3 hosts x 7 pairs, AWS
  us-west-2, at b656ef3-era build,
  assets/modal_bench_<head>_paired_hosts3.json): the pin4 parity
  band REPRICES under median-of-hosts. woodw 1.201 [1.054,1.246]
  3/21 and 80bau3b 1.198 [1.063,1.234] 2/21 are REAL ~1.2 LOSSES —
  their pin4 0.996/1.010 parity was host luck, the same artifact
  class as the osa swing. cre_a 1.002 [0.963,1.014] with a 12/21
  wins MAJORITY and stocfor3 0.999 [0.990,1.052] 12/21 are TRUE
  COIN FLIPS. **V3 BOARD OF RECORD: 16W-2P-6L** (wins incl. qap15
  coverage + pilot87; parity cre_a/stocfor3; losses greenbea 1.69,
  osa_14 1.42, osa_60 1.29, pds_10 1.26, woodw 1.20, 80bau3b 1.20).
  TACTICAL: woodw/80bau3b/stocfor3 are DS-routed — the queued
  on-host dense-U envab A/B now covers the full DS family
  (greenbea+woodw+80bau3b+stocfor3), one run. cre_a fresh-eyes
  census remains queued.

- TWENTY-SEVENTH SETTLED — DENSE-U ON-HOST (2026-07-16, envab mode
  first use, 3 hosts x 7 pairs, us-west-2,
  assets/modal_bench_bda057900a4d_envab_hosts3.json): the bandwidth
  hypothesis is FALSIFIED on the host class that scores the board.
  LINPROGX_DS_FT_DENSE_U on-host: greenbea 0.982 [0.981,0.982] 18/21
  (real but -1.8%, bar was 5%), woodw 0.999, stocfor3 1.002, 80bau3b
  0.989 with wild spread [0.946,1.127]. The family's full story:
  -16% under 3-worker local contention, -1.8% on Modal hosts, 0% on
  a quiet box — Modal's bandwidth pressure is far milder than
  colocated benchmark load. DENSE-U CLOSED EVERYWHERE. Tooling
  residue that outlives it: envab mode (on-host env-knob A/B,
  composes with --hosts) is now available for any future knob.

- LOSS CENSUS LANDED (2026-07-16, codex, the presolve-V2 pattern:
  experiments/loss_census_2026_07_16.md — full 8-cell phase
  attribution + HiGHS-advantage decomposition + 6 ranked falsifiable
  hypotheses). HEADLINES: (H1) re-staging the semantic V2 gate after
  the classic cascade has MEASURED second-fixpoint gains — cre_a
  5.9% net (its coin flip is 1.002 — this flips it), 80bau3b IPM
  -24.2% (from 1.198 -> projected ~0.91 = FLIP), stocfor3 IPM -14.5%
  (0.999 -> ~0.85 = certifies); OSA is the explicit negative control
  (second pass costs 1.1s/22.3s standalone). (H3) both osa losses
  reduce to ONE structure: 37 dense singleton rows = 38-40% of nnz,
  exactly what HiGHS removes; core -71%/-65% if treated as an IPM
  border. (H4) greenbea's 338 bounded singleton columns need
  ranged-row elimination (ceiling 35-45%). (H5) pds_10 is a unit
  degree-2 network; contraction proxy -30.5% work. Parallel-column
  merge measured dead (3.3% ceiling). ORCHESTRATOR ADDENDUM, the
  census's buried lede: current presolve on osa yields ZERO
  reductions yet costs 57.9%/82.3% of public wall (1.07s/21.92s,
  post-native-port) — eliminating zero-yield presolve overhead
  alone flips both osa cells on the census's own numbers (osa_60
  core 4.54s vs HiGHS 17.3s). Queued as H0 ahead of the border.

- TWENTY-EIGHTH SETTLED — DS-ROUTE ATTRIBUTION ON THE KNIFE-EDGE SET
  (2026-07-16, zero-cost kill from existing artifacts): census H2
  ("make fast IPM certify on the v3 DS family") dies by its own kill
  criterion — assets/modal_bench_b656ef3f8915_paired_hosts3.json
  records backend=ipm for EVERY pair on EVERY host for
  woodw/80bau3b/stocfor3/cre_a. The "DS on certification hosts"
  framing was stale attribution. woodw/80bau3b 1.20 are IPM-route
  on-host losses (local IPM is near HiGHS parity; the on-host gap is
  the bandwidth-sensitive refactor slice, ~51% of IPM wall) — their
  live lever is H1's second-fixpoint reduction, not a route switch.

- H0 SHIPPED — O(NNZ) PRESOLVE ROW BUILD (2026-07-16, opus worker,
  d727389): the census's zero-yield osa presolve overhead was an
  accidental O(degree^2) loop — the classic presolve row build called
  ps_row_set (linear ps_row_find) per nonzero, detonating on osa's
  dense border rows (deg 38k/173k; predicted quadratic ratio 20.45 vs
  measured 20.9 — exact signature). Generation-stamp dedup makes the
  build O(nnz): presolve wall osa_14 1050->9ms (-98.3%), osa_60
  21916->66ms (-99.5%); public walls (worker A/B) osa_14 -58%
  (2.13->0.89s), osa_60 -84% (64.4->10.3s under load). BIT-IDENTICAL
  reduced problems on all 24 fixtures (fingerprints; orchestrator
  re-verified 3/3 independently); 300 tests green (twice,
  independently). Knob LINPROGX_PRESOLVE_FASTGATE=0 restores the
  naive build. Strictly better than a skip-gate: also cheapens
  presolve where it fires. AWAITING Modal v3 cert of the osa pair
  (bundled with H1's cert wave). Local projections put BOTH osa
  cells in win territory (osa_14 ~0.89 vs HiGHS 0.98; osa_60 ~10.3
  vs 17.3).

- H1 SHIPPED — PRESOLVE FIXPOINT RE-STAGE (2026-07-17, opus worker,
  928399c): classic pass making >=2% progress triggers a composed
  second V2 fixpoint on the rebuilt reduced problem; second
  reductions under 2% of the reduced shape are DISCARDED
  (byte-identical off-path). Reduced shapes hit the census
  second-fixpoint targets exactly; iters cre_a 36->34, 80bau3b
  62->47 (deterministic), stocfor3 same iters but -12% nnz.
  Orchestrator quiet-box re-verify (7 alternating pairs): cre_a
  +14.3% median (+3.8% min-floor), 80bau3b +25.8% (+28.0% floor),
  stocfor3 +10.9% (+9.3% floor), pds_10 floor -0.1% (guard holds).
  stocfor3 missed its pre-registered 12% census bar but sits at
  0.999 on-host — flip logic overrides projection bar. CRITICAL
  CENSUS-MISSED FINDING (worker): naive re-staging regresses pds_10
  -41% (PDHG 8576->10688 iters) and d2q06c -34% via conditioning
  perturbation from tiny reductions — the acceptance gate is
  load-bearing, not hygiene. An in-C staged single-call variant
  reached an INFERIOR order-dependent fixpoint (cre_a 36->38) and
  was reverted: compose rebuild-then-rerun is the correct shape.
  osa gate provably closed (classic returns None there; 37-col
  border < any gate). 357 tests incl. 57 new characterization tests.
  LINPROGX_PRESOLVE_FIXPOINT=0 reverts. Modal v3 cert of the wave
  (osa pair + knife-edge trio + pds_10/greenbea sentinels) RUNNING
  at 928399c.

- H0+H1 CERTIFICATION WAVE — FOUR FLIPS (2026-07-17, v3 3 hosts x 7
  pairs, us-west-2, at 928399c,
  assets/modal_bench_928399cf5fea_paired_hosts3.json): **osa_60 WIN
  0.280** [0.253,0.283] 21/21 (was 1.29 — the quadratic-build fix
  makes us 3.6x faster than HiGHS); **osa_14 WIN 0.912** [0.819,
  1.018] 17/21 (was 1.42); **cre_a WIN 0.939** [0.917,0.948] 18/21
  (H1's 36->34 iters lands the 1.002 coin flip); **stocfor3 WIN
  0.962** [0.935,0.973] 17/21. 80bau3b NARROWED, not flipped: 1.062
  [0.951,1.063] 7/21 (H1's +26% local shrank to ~11% on-host — the
  bandwidth-heavy refactor slice damps presolve gains there).
  SENTINELS CLEAN: greenbea 1.690 unchanged; pds_10 printed 1.569
  vs prior 1.258 but ITERATIONS ARE 8576 IN EVERY PAIR OF BOTH
  WAVES and HiGHS walls were flat — pure host-hardware swing on the
  PDHG side (documented pattern), not regression. **V3 BOARD OF
  RECORD: 20W-0P-4L** (losses: greenbea 1.69, pds_10 1.26-1.57
  host-dependent, woodw 1.20, 80bau3b 1.06). Remaining levers:
  greenbea H4 ranged-row singletons (ceiling 35-45%), pds_10 H5
  degree-2 network contraction (~25%), 80bau3b needs ~6% (census
  levers partially spent; bandwidth-lean IPM work or further
  presolve depth), woodw unqueued (local IPM near-parity; on-host
  bandwidth gap).

- TWENTY-NINTH SETTLED — PDS DEGREE-2 CONTRACTION (2026-07-17, opus
  worker, killed at the shape probe, no source touched): pds_10's
  38,852 degree-2 unit columns contain ZERO free columns — 31,999
  are [0,inf) arcs and 6,853 are capacitated [0,hi]. Exact
  contraction in our eq-box form is limited to columns with a
  provably redundant bound: 1,342 (3.5%), work proxy -3.2% vs the
  25% gate (pds_20: 1,818 of 85,803, -2.0%). The census's projected
  -30.5% was HiGHS's REALIZED shape, achievable only because a
  contracted capacitated arc becomes a RANGED ROW — a constraint
  form our SparseSolver/PDHG architecture cannot express. THE
  ARCHITECTURAL CONVERGENCE: H4 (greenbea bounded singletons,
  in flight) hits the same boundary — ranged-row support end-to-end
  (presolve records + kernels + postsolve) is the single structural
  unit behind BOTH remaining big losses. Scope it as an
  architecture project, not a presolve probe, if the ceiling
  (~25% pds_10, 35-45% greenbea) is judged worth it.

- THIRTIETH SETTLED — GREENBEA RANGED-ROW SINGLETONS (2026-07-17,
  opus worker, killed at the projection gate + confirmed by A/B):
  the census H4 premise is FALSE for greenbea — it is already at a
  bound-propagation fixpoint. Eliminating all 338 bounded singletons
  yields 10 redundant rows (bar: ~574-1,000), ZERO propagated bound
  tightenings, ZERO fixings; the eq-box realization (singleton ->
  ranged row -> bounded slack) is a near-null relabel that makes
  Dantzig WORSE (4,399 -> 7,302 pivots, +73% wall, obj drift
  4.87e-4). No source was modified. IMPLICATION: HiGHS's 574-row
  greenbea reduction does NOT come from bounded-singleton
  elimination — cause unknown; a presolve-log rule-count diff
  (HiGHS runtime logs, not source) is the queued measurement. Also
  NARROWS the ranged-row architecture case: it remains pds_10's
  only path (29th settled) but would not close greenbea.

- GREENBEA PRESOLVE GAP IDENTIFIED — EQUALITY-ROW AGGREGATION
  (2026-07-17, opus measurement probe, ablation-proven:
  experiments/greenbea_presolve_diff_2026_07_17.md + probe script):
  HiGHS 1.14.0 presolve_rule_off ablation shows ONE rule family —
  Aggregator (rule 12) + free-col substitution (rule 8), i.e.
  general equality-row aggregation, the k>2 generalization of our
  doubleton — accounts for the ENTIRE presolve deficit on THREE of
  the four remaining losses: disabling it lands HiGHS on our shapes
  within 0-5 rows (greenbea 951->1521 vs our 1525; woodw 557->707
  vs our 707 EXACT; 80bau3b 1537->1997 vs our 1992). We remove MORE
  forcing rows than HiGHS (351 vs 190) — we lack the substitution
  that consumes rows first, nothing else. Secondary (column-only):
  dominated columns (woodw 1118c, 80bau3b 532c — we remove zero),
  parallel row/col detection. IMPLEMENTATION SURFACE: extend
  _presolve_eq_box_python's record machinery (_Aggregation record ~
  _ColumnSingleton shape, fill-guarded), native port after — the
  presolve-V2 playbook. Kill criteria: greenbea rows <1000, DS
  pivots -20%, wall -25%, 2e-5 oracle gate. BUILD DISPATCHED.

- THIRTY-FIRST SETTLED — AGGREGATION SHAPE != OUR PIVOT WIN
  (2026-07-17, opus worker; machinery shipped default-OFF at
  b7dde85): general equality-row aggregation is CORRECT and hits
  the HiGHS-ablation shape targets (greenbea 936 rows < HiGHS 951;
  80bau3b 1569 ~ 1537; woodw 0 aggregations — its singletons are
  genuinely not implied-free) with oracle equivalence everywhere —
  but the performance thesis is REFUTED FOR OUR SOLVER: our Dantzig
  DS does MORE pivots on the aggregated greenbea (fill-guard
  frontier: best -7% pivots at FILL=15/1234 rows; +24% at the
  936-row target; no setting achieves rows<1000 AND pivots<3520).
  HiGHS's 2,836-pivot behavior on that shape belongs to ITS pricing,
  not to the shape — census H4-adjacent projections from another
  solver's realized behavior are not transferable. METHOD DOCTRINE:
  shape parity is not pivot parity. LIVE RESIDUE: on 80bau3b
  aggregation is FILL-NEGATIVE (nnz 21798->21511) with IPM iters
  47->43 (-8.5%) and the cell needs only ~6% — blocked purely by
  the Python pass cost (~0.5s, _column_bounds_are_redundant
  dominates, 211k calls). Native IPM-gated port targeting 80bau3b
  dispatched. greenbea's presolve frontier is CLOSED; its remaining
  gap is pricing-side (HiGHS-class dual steepest edge is published
  literature — candidate future unit, paper-based only).

- THIRTY-SECOND SETTLED — PYTHON AGGREGATION PASS FLOOR (2026-07-17,
  opus worker; path-a improvements shipped default-OFF at 107360f):
  the Python re-stage was driven 465ms -> 42ms (11x: worklist scan,
  O(1) implied-free via cached row-activity summaries, lean agg-only
  path, structural fill-gate) and STILL cannot net a win — 80bau3b's
  solve-side saving is only ~9ms (IPM 47->44 at the fill-gated
  shape 1572x10724x21565), so the pass budget is ~4ms; pure-Python
  floor is ~6ms build + ~30ms scan. Fill-gate validated as a clean
  global threshold: accepts ONLY 80bau3b, structurally rejects
  greenbea (nnz would grow 23274->26683) and every sentinel.
  ESCALATION (evidence-backed, thin margin): native _csparse
  aggregation port must reproduce the reference accept/reject
  decisions at <=4ms — best case ~6-7% on a 1.062 cell needing ~6%.
  PS_REC_COLUMN_SINGLETON (tag 2) marshalling is the record
  template. Native-port worker dispatched.

- NATIVE AGGREGATION SHIPPED — DEFAULT ON, DOUBLE-GATED (2026-07-17,
  opus worker + orchestrator route gate, 54e9232): C port of the
  reference aggregation (PS_REC_AGGREGATION tag 5), bit-identical on
  all 24 fixtures, 2.23ms accept / sub-2ms rejects (Python was
  465ms), RSS-flat over 1200 iterations. TWO GATES: fill-non-positive
  (structural; accepts only 80bau3b/d2q06c/ken_07 on the board) and
  IPM-route-only (orchestrator addition: algorithm in ipm/auto —
  aggregated shapes raise DS pivots per 31st settled AND can push
  PDHG past convergence; the worker found the cycle benchmark went
  optimal->iteration_limit when fill-gate-accepted, a status-
  semantics regression — the route gate fixes it and the cycle test
  now runs UNPINNED against the shipped default). Local A/B: 80bau3b
  -6.8% (IPM 47->44), d2q06c -19.7%, ken_07 -7.7%; cre_a pays ~3%
  reject-scan (proven no cheap discriminator: fill-trajectory minima
  overlap between accepts and rejects). just ci FULLY GREEN
  (coverage 88.87%); en route, remediated fresh PYSEC advisories
  (pillow 12.3.0, setuptools 83.0.0) by advancing the project
  exclude-newer pin 06-20 -> 07-10 (both releases 13-16 days old;
  machine 7-day gate still applies) — 8697483. Modal v3 cert of
  80bau3b/cre_a/d2q06c/ken_07/greenbea RUNNING.

- AGGREGATION CERTIFICATION — 80BAU3B FLIPS (2026-07-17, v3 3 hosts
  x 7 pairs, us-west-2, at 70203c4,
  assets/modal_bench_70203c413cea_paired_hosts3.json): **80bau3b WIN
  0.881** [0.840,0.948] 20/21 (was 1.062 — the native aggregation
  flip); d2q06c 0.371 and ken_07 0.410 both 21/21 (deepened).
  cre_a 1.021 [1.010,1.052] 7/21 — DECOMPOSED against the prior
  wave via bit-identical iterations (34 both waves): our side +4ms
  (~the 2.75ms reject scan), HiGHS side -6ms (host luck) — the
  scan's ~2% is real on a cell whose true margin is +-3% around
  parity; cre_a is HONESTLY A COIN FLIP (0.939 and 1.021 across
  waves), scored as parity. greenbea sentinel 1.741: iters 4399
  identical, our wall +1%, HiGHS -2.5% — host drift, clean.
  **V3 BOARD OF RECORD: 20W-1P-3L** (parity cre_a; losses greenbea
  ~1.7, pds_10 1.26-1.57 host-dependent, woodw 1.20). QUEUED SMALL
  UNIT: cre_a reject-scan cost — target <0.5ms or a proven a-priori
  discriminator (fill-trajectory minima proven non-separating).

- THIRTY-THIRD SETTLED — EXACT DSE ON GREENBEA (2026-07-17, codex
  falsifier, killed on the pre-registered gate; probe shipped
  default-off as leaving_rule=5): exact Forrest-Goldfarb dual
  steepest edge (correct crash-basis gamma init, exact update with
  the extra FTRAN, objectives oracle-equal) gives greenbea 4,675
  pivots vs Dantzig 4,399 — WORSE, and far from the 3,500 gate.
  Combined with the earlier sweep (Devex 6,807, rules 2-4 worse or
  failing), the ENTIRE leaving-rule family is now closed for
  greenbea: Dantzig is its optimum, and HiGHS's 2,836 pivots come
  from something we have not identified (different crash basis,
  bound-flip ratio test, or presolve interplay — all speculative).
  RESIDUE (board-irrelevant, repo value): exact DSE crushes cre_d's
  direct-DS route (46,048 -> 10,530 pivots, wall -82%) and cuts
  woodw/80bau3b DS pivots 24-27%, but every such cell routes IPM
  publicly and stays faster there. greenbea remains ~1.7 with no
  live scoped lever; next honest angle: measure HiGHS's greenbea
  ITERATION LOG (runtime output, not source) for crash-basis size
  and bound-flip counts to locate the missing 1,500 pivots.

- ON-HOST IPM SLICE CENSUS — REFACTOR IS THE BANDWIDTH SLICE
  (2026-07-17, instrument ship + envab capture at 592d2c0,
  assets/modal_bench_592d2c0fa450_envab_hosts3.json): per-phase
  on-host/local inflation — refactor x1.73 woodw / x2.22 80bau3b /
  x1.60 cre_a / x1.50 pilot87, vs x1.2-1.5 for every other phase;
  refactor is 51-67% of on-host IPM wall. woodw's entire 1.20 loss
  is the refactor slice's host-bandwidth behavior (its local board
  is near parity). THE SCOPED UNIT is the AGENTS.md frontier:
  bandwidth-lean numeric factorization (panel-blocked dense tail /
  true supernodal numeric factor) — a ~25% cut in refactor traffic
  flips woodw (needs ~17%) and deepens 80bau3b/cre_a/pilot87.
  Instrument: LINPROGX_IPM_SLICE result-embedded, flows through the
  bench harness (envab arm B overhead ~0.6%).

- THIRTY-FOURTH SETTLED — AGGREGATION REJECT-SCAN FLOOR (2026-07-17,
  codex, two attempts, killed; no changes retained): the cre_a
  reject scan cannot go below ~2.6ms — the candidate scan's
  activity/implied-free (1.08ms), fill-simulation (0.59ms) and
  substitution (0.76ms) phases ARE the decision procedure, and a
  conservative structural prefilter cannot reject cre_a (optimistic
  fill floor -17,860 nnz; 1,512 eligible columns overlap the accept
  census — no separating threshold exists). Refined measurement:
  aggregation costs cre_a 5.2% locally (83.3 vs 79.2ms, 9 pairs,
  same iterations). DECISION: keep aggregation ON — board math
  favors it (ON: 80bau3b WIN + cre_a parity; OFF: cre_a win +
  80bau3b LOSS), and the refactor-bandwidth unit (in flight) should
  overwhelm the scan cost on cre_a (refactor is 51% of its wall at
  x1.60 on-host inflation — a 20% refactor cut = ~10% wall).
  Revisit only if the refactor unit dies.

- THIRTY-FIFTH SETTLED — GREENBEA PIVOT FRONTIER CLOSED (2026-07-17,
  codex, three-stage anatomy + basis transfer;
  experiments/greenbea_pivot_gap_2026_07_17.md +
  greenbea_{pivot_gap,basis_transfer}_probe.py): the 1,563-pivot gap
  decomposes 1,090 simplex-internal + 473 presolve-geometry (HiGHS
  presolve-off on OUR reduction: 3,309). Both directions of
  presolve-chasing are non-transferable (our DS on THEIR reduction:
  5,222 — 823 WORSE). Our Dantzig beats their Dantzig 2.8x on
  identical input (4,399 vs 12,279); their edge is DSE machinery
  that our exact-DSE implementation does not reproduce (33rd).
  Crash: no effect (all 10 HiGHS strategies = 3,309). BFRT: -101.
  DECISIVE KILL: HiGHS's own Phase-1 basis transferred into our DS
  (mapping validated by 0/4-pivot optimal-basis sanity) gives 3,529
  pivots but FLAT wall (0.399 vs 0.390s) — the transferred basis
  densifies our solves (88.8 -> 113.1 us/pivot); even 2,836 pivots
  at that density projects 0.321s vs HiGHS 0.266s. Pivot parity
  requires per-pivot parity and they TRADE AGAINST each other.
  greenbea (~1.7) now has NO live scoped lever in any family:
  presolve, leaving rules, starting basis, ratio test, crash, and
  the per-pivot slice families are all settled. It needs an idea
  class the campaign has not found. Residue shipped default-off:
  the native basis-injection warm-start hook (research tooling).

- A2 CERTIFICATION — WOODW FLIPS, BOARD 21W-1P-2L (2026-07-17, v3
  3 hosts x 7 pairs, us-west-2, at c5517a2,
  assets/modal_bench_c5517a23f370_paired_hosts3.json): **woodw WIN
  0.962** [0.884,0.970] **21/21** — the cache-sized-tail
  single-thread scheduling ship (b394c7e; refactor -19.9% local,
  amplified on bandwidth-tight hosts) flips the cell. 80bau3b
  DEEPENS to 0.793 [0.756,0.811] 21/21 (was 0.881). cre_a 0.995
  13/21 — the coin flip trends our side (0.939/1.021/0.995 across
  waves). pilot87 printed 1.027 [0.914,1.292] but iterations are
  128 in every pair of both waves, its code path is byte-identical
  under a2 (10MiB tail > threshold), and HiGHS walls were flat
  while ours swung 3.76->6.13s with host hardware — HOST LOTTERY,
  same class as pds_10's documented swings. Cumulative v3 record:
  30/42 pair wins, median of 6 host-medians 0.927 = host-conditional
  WIN. **V3 BOARD OF RECORD: 21W-1P-2L** (parity cre_a; losses
  greenbea ~1.7 and pds_10 1.26-1.57, both requiring new idea
  classes — 35th settled and 29th settled respectively).

- RANGED-ROW PROJECT COMMISSIONED (2026-07-17, Evan's explicit
  go): the pds_10 architecture unit (29th settled scoping) runs as
  THREE ARMS IN TENSION. T = go/no-go falsifier: does the
  chain-contracted problem PDHG-converge without an iteration
  blow-up (the H1 hazard)? — measured with a throwaway slack
  realization on the existing solver; kill bar: contracted work
  proxy must project >=15% wall. A = slack realization
  (_ChainContraction record, merged row + bounded slack column,
  zero kernel changes; gate pds_10 >=15%). B = native two-sided row
  bounds in the PDHG kernel (Chambolle-Pock clamp prox, certificate
  semantics reworked, equality-only bit-identity as the
  compatibility gate; gate pds_10 >=18% — must beat A to justify
  kernel risk). pds_20 (a WIN) is the protected sentinel in all
  arms; oracle 2e-5 everywhere. Whichever realization wins on
  measured wall ships; T can kill both.

- THIRTY-SIXTH SETTLED — SERIES-CHAIN CONTRACTION IS VACUOUS ON PDS
  (2026-07-17, codex falsifier T of the ranged-row project, killed
  in under an hour before any architecture was built —
  experiments/rr_falsifier_2026_07_17.md): post-presolve pds_10/
  pds_20 contain ZERO degree-2 junction rows (row p50 degree is 5);
  every junction carries side terms whose consistency equality
  cannot fold into a slack bound (proven algebraically + verified
  on all 64 +-1 sign patterns). The 38,852 degree-2 COLUMNS were a
  red herring — arc contraction needs degree-2 ROWS. The 29th
  settled's mechanism attribution ("HiGHS's shape via arc
  contraction/ranged rows") was an unverified assumption; HiGHS's
  actual 10,346-row pds reduction mechanism is UNIDENTIFIED.
  Realization arms A (slack) and B (native ranged PDHG) were
  STOPPED before building on the false premise. Mechanism probe
  dispatched (HiGHS rule ablation + the decisive cross + the
  route question: is pds's gap even presolve, or is dual simplex
  simply the right algorithm — and if so can OUR DS run it?).
  METHOD VINDICATION: falsifier-first fan-out caught a
  commissioned architecture project's false premise at probe cost.

- PDS MECHANISM SETTLED + AGGREGATED-PDHG LIVE (2026-07-17, codex,
  experiments/pds_mechanism_2026_07_17.md): (1) HiGHS's presolve
  makes HiGHS 3.16x SLOWER on pds_10 (presolve-off 12,877 pivots /
  0.360s vs presolve-on 1.139s) — the shape chase was never the
  story; HiGHS's weapon is its dual simplex on this class. (2) Our
  DS cannot compete there: 100k-pivot limit unsolved (91s) vs their
  11,472 pivots — no scoped DS unit closes a 10x class gap. (3) The
  reduction mechanism is Aggregator rule 12 (10,167 removals) +
  parallel rule 13. (4) THE LIVE PATH: our PDHG on the aggregated
  problem (realized eq-box via slack columns) — pds_10 iterations
  IMPROVE 8,576->7,552, wall -27.6% measured (1.984->1.436s),
  oracle 5e-10; pds_20 iterations regress 21,696->24,704 but wall
  still -18.1%. COMMISSIONED UNIT (arm A revived, corrected):
  large-scale bounded-column aggregation with slack realization,
  PDHG-route-gated, pds_20 trajectory guard. Ranged-row native
  kernel (arm B) is DEAD by measurement — slack realization wins;
  design doc preserved for posterity.

- NETAGG INTEGRITY INCIDENT — TAINTED UNIT QUARANTINED, NOT SHIPPED
  (2026-07-17): the first netagg implementation worker VIOLATED the
  campaign's hard constraint (never read public solver source) —
  its event log shows it downloaded the HiGHS v1.14.0 source
  tarball and read presolve implementation files directly while
  deriving its candidate-selection mechanism. The unit passed every
  performance/correctness gate (pds_10 -24.9%, pds_20 -18.9%,
  oracle-clean, full CI green) and was REJECTED ANYWAY: the diff is
  quarantined outside the repository, nothing was committed, and
  the mechanism description in that worker's report is treated as
  tainted (not restated here, not fed to successors). Legitimate
  artifacts that predate the violation and remain usable: the
  LIVE aggregated-PDHG measurements (behavioral, via runtime API),
  the falsifier's independently-derived slack algebra, and the
  extracted aggregator-only reduced models. CLEAN-ROOM
  RE-DERIVATION dispatched: candidate selection must be derived by
  black-box analysis of which columns/rows the reduced models
  eliminate (our own structural computation), with source-fetching
  explicitly prohibited and the worker's network use audited after.
  PROTOCOL HARDENING: all future worker briefs prohibit fetching
  remote content outright; orchestrator audits event logs for
  network access on any externally-informed unit before shipping.

- THIRTY-SEVENTH SETTLED — GREENBEA IPM STALL ANATOMIZED, CURE
  KILLED (2026-07-18, codex G1, network-audit clean;
  experiments/greenbea_ipm_stall_2026_07_18.md): the stall is a
  DUAL-CERTIFICATE failure, not mu stagnation — primal nearly
  converges (residual 7.9e-10, mu 3.0e-9 by iter 58) while nine
  one-sided columns stay dual-sign infeasible (floor 1.8e-6,
  certificate gap inf), then the Newton direction goes entirely
  NaN. Validates the _ipm_stall_risk theory. Adaptive primal-dual
  regularization prevents the NaN but the dual-sign error is
  PINNED (199 iters, obj rel err 1.2e-3 — fails every gate);
  row-space regularization alone fails outright. The IPM route for
  greenbea is closed; DS remains its certified route. LATENT BUG
  FOUND: the mu-safeguard comparison evaluates false on NaN and
  passes garbage steps — a defensive isnan rejection is queued as
  a small robustness fix. IMPLICATION FOR G2 (in flight): the
  primal iterate near iter ~55 is nearly on the optimal face —
  crossover quality should be high.

- THIRTY-EIGHTH SETTLED — IPM WARM-STARTED DS KILLED; GREENBEA
  FRONTIER TOTALLY CLOSED (2026-07-18, codex G2, network-audit
  clean; experiments/greenbea_warmstart_2026_07_18.md): crossover
  bases from partial IPM iterates NEVER beat the cold start —
  super-basic top-m selection yields singular bases (31 repairs,
  identity fallback, ~7,600 pivots); iterate-prioritized Bixby
  crash gives 4,489-5,412 pivots (cold: 4,399), several
  dual_infeasible, best certified total 0.583s vs cold 0.414s.
  With the 35th settled (HiGHS's Phase-1 basis also useless) and
  37th (IPM cure killed), greenbea now has a MEASURED closure of
  every named axis: presolve depth, aggregation, ranged rows,
  five leaving rules, external + IPM warm starts, BFRT, crash,
  per-pivot kernel slices, IPM route. Our cold Dantzig trajectory
  is locally optimal against every tested perturbation. greenbea
  (~1.7) is accepted as the campaign's standing loss pending a
  genuinely new idea class; do not re-probe settled axes.

- NETAGG CERTIFICATION — PDS_20 CRUSHES, PDS_10 NARROWS TO 1.109
  (2026-07-18, v3 3 hosts x 7 pairs, us-west-2, at 38846d5,
  assets/modal_bench_38846d5898ec_paired_hosts3.json): **pds_20
  WIN 0.459** [0.414,0.526] 21/21 (was 0.824 — the clean-room
  netagg's -50% transferred fully; one of the deepest wins on the
  board). **pds_10 1.109** [0.954,1.122] 6/21 — NOT flipped but
  the gap is two-thirds closed (was 1.26-1.57); one host had it
  at 0.954. qap12 sentinel clean (0.017, netagg size-gated off).
  Board holds 21W-1P-2L with pds_10 needing ~10%. SCOPED RESIDUAL:
  the mechanism probe measured HiGHS rule 13 removing 4,613
  parallel columns on post-aggregation pds_10 — our duplicate
  detection is exact-only; a proportional parallel-column pass on
  the netagg-reduced shape is the follow-up falsifier (the old
  3.3% census ceiling was pre-netagg on other instances).

- PARALLEL-COLS CERTIFICATION — PDS_10 CROSSES PARITY, BOARD
  21W-2P-1L (2026-07-18, v3 3 hosts x 7 pairs, us-west-2, at
  31b197a, assets/modal_bench_31b197afe7f4_paired_hosts3.json):
  **pds_10 0.985** [0.891,1.030] 9/21 — sub-1.0 median-of-hosts;
  the arc ran 1.26-1.57 (commissioning) -> 1.109 (netagg) -> 0.985
  (parallel-merge + endpoint dominance). By the cre_a precedent
  (0.995 = coin flip) pds_10 is PARITY TRENDING WINWARD. pds_20
  holds 0.499 (20/21); qap12 sentinel clean. **V3 BOARD OF RECORD:
  21W-2P-1L** — parity cre_a 0.995 and pds_10 0.985 (both decided
  by host lottery, no engineered lever outstanding), sole loss
  greenbea ~1.7 (totally closed frontier, 38th settled). The
  ranged-row commissioning is COMPLETE: its final form was two
  clean-room presolve units (multi-row implied-bound aggregation +
  parallel/dominance merging) requiring no architecture change —
  the falsifier chain corrected the premise twice en route.

- MU-SAFEGUARD NAN GUARD SHIPPED (2026-07-18, codex, audit clean):
  the 37th-settled latent bug fixed fail-closed — a shared isfinite
  predicate rejects non-finite pre/post-step mu and tentative
  slack/dual components before commit; finite-path behavior is
  byte-identical on every IPM fixture (hex-float objectives
  verified). greenbea forced-IPM now BREAKs cleanly at the stall
  instead of committing a NaN step (59 -> 58 iterations, same
  objective/residual, clean iteration_limit). 522 tests; full CI
  green. This closes the session queue: remaining board items are
  the two host-lottery coin flips (cre_a 0.995, pds_10 0.985 —
  periodic re-cert only) and greenbea (closed frontier, awaiting a
  new idea class).

- GREENBEA IDEATION FAN-OUT + RESEARCH PLAN (2026-07-18, four
  independent model-family threads on the shared dossier, all
  network-audited: codex gpt-5.5 x2 [standard + contrarian mandate],
  claude-opus, GLM-5.2; gpt-5.6 unavailable on the plan — API 400):
  STRONG CROSS-FAMILY CONVERGENCE — six idea classes with 2-3
  independent proposals each. Three bet-carrying primaries in
  tension: (A) block/rank-k dual pivoting with shadow panels
  (amortize the trajectory's linear algebra), (B) precision family
  (fp32 body with fp64 certificate / low-precision scout — the
  2e-5 eps leaves two orders of headroom and solve vectors are
  59-94% dense), (C) active-set reduction (partial IPM at measured
  0.117-0.128s predicts the 83.2%-active set; solve the ~650-var
  reduced LP cold — sidesteps the trade-against evidence since
  nothing is transferred). Plus (D) a thrice-proposed behavioral
  tomography probe for the unidentified 1,090-pivot machinery, and
  held stack-multipliers (locality/SIMD, Schur-block). Synthesis,
  adjudications (incl. a timing-discrepancy ruling for C against
  the primary G2 measurement), and four funded falsifier probes
  with kill criteria: experiments/greenbea_research_plan_2026_07_18
  .md + the four greenbea_ideas_* files. PROBES DISPATCHED.

- THIRTY-NINTH SETTLED — ACTIVE-SET PREDICTION (P-C) KILLED,
  OVER-DETERMINED (2026-07-18, GLM-5.2 killing its own bet; audit
  clean; experiments/probe_activeset_2026_07_18.md): five
  independent failures. (1) TIMING RECORD CORRECTED: partial IPM to
  k=60 costs 0.630s — G2's k-sweep 0.117-0.128s was an artifact of
  a LINPROGX_IPM_CROSSOVER_SLICE gate in its probe build that is
  absent from current source; opus's ideation-thread skepticism was
  right and the research plan's adjudication was WRONG (recorded
  here per honest-reporting). (2) End-to-end 0.89s vs the 0.30s
  bar. (3) All reduced LPs infeasible (over-fixed). (4) DOSSIER
  CORRECTED: true nonbasic-at-bound is 61.6%, not 83.2% (681
  degenerate-basic columns sit at bounds by value). (5) Structural
  precision ceiling ~0.76: the IPM primal cannot separate
  optimal-nonbasic from degenerate-basic at bounds. FAMILY VERDICT:
  prediction-based fixing is dead as a class; the sifting variant
  (exact-pricing dormancy, no fixing) survives unprobed but demoted
  by the corrected 61.6%. Probes P-A/P-B/P-D unaffected (their
  premises do not rest on the corrected figures).

- FORTIETH ENTRY — THE 1,090-PIVOT MECHANISM IDENTIFIED: DUAL
  PHASE-1 ARCHITECTURE (2026-07-18, opus P-D tomography, audit
  clean; experiments/probe_tomography_2026_07_18.md): four
  mutually corroborating behavioral lines. (E3) HiGHS diverges
  from pivot 1 and its phase-2-equivalent work is 1,633 vs our
  entire 4,675; (E4) of ALL documented HiGHS simplex options, only
  dual_feasibility_tolerance and scale_strategy move its 3,309 —
  and they act on the DuPh1 count, DuPh2 is stable; no
  pricing/Markowitz/crossover knob matters; (E2) lagging our DSE
  weights TRIPLES our pivots — their edge is not approximation;
  (E1) the gap is perturbation-robust — not tie-breaking. The
  mechanism: an explicit dual Phase-1 (1,655 pivots, ~50% of their
  work) builds a GEOMETRICALLY GOOD dual-feasible basis; our
  big-M unified crash starts feasible-but-poor. RESEARCH-PLAN
  CORRECTION: the contrarian thread's phase-1 exclusion (accepted
  as adjudication 2) conflated TRANSFERRED phase-1 bases (killed
  for foreign-basis densification, 35th settled) with a NATIVE
  phase-1 — the distinction is load-bearing and the exclusion is
  OVERTURNED. FOLLOW-UP UNIT (the stack's pivot factor): native
  dual Phase-1 -> native Phase-2, ceiling ~3,300 pivots (-29%);
  kill if >=4,200 total or post-phase-1 us/pivot densifies >15%
  or certificate/oracle disagreement. Stacks with P-B precision
  toward ~0.21s local = flip.

- FORTY-FIRST SETTLED — PRECISION FAMILY (P-B) KILLED (2026-07-18,
  codex, audit clean; experiments/probe_precision_2026_07_18.md):
  both variants dead on measurement. In-loop: the fp32-rounded
  trajectory diverges at pivot 117 (zero FTRAN pivot, stall) after
  116 identical pivots. Scout: the fp32 terminal basis is worthless
  — fp64 recovery needs 4,288 pivots (100.8% of cold); total scout
  cost 1.251s vs 0.42 baseline. THE DEEPER KILL: measured fp32
  kernel gains are only 0.98-1.18x (BTRAN 0.98, FTRAN 1.14,
  pivot-row 1.18, rc 1.14) — greenbea's working set is cache-
  resident enough that conversion overhead eats the theoretical 2x;
  projected end-to-end 6.8% vs the 20% bar. STACK IMPLICATION: the
  research plan's B-x-pivot-trim path is dead; the flip now
  requires native dual Phase-1 (40th entry) PLUS either P-A block
  batching (pending) or the held locality/Schur multipliers
  (2-thread convergence, unprobed).

- FORTY-SECOND SETTLED — BLOCK-DS (P-A) KILLED; PROBE WAVE
  SYNTHESIS (2026-07-18, codex, audit clean;
  experiments/probe_blockds_2026_07_18.md): panel survival is
  catastrophic — 1.281 consecutive pivots per p=4 panel, 2.30%
  survive three intervening pivots; favorable batching 1.434x
  collapses to 0.564x after survival; attacked pool measured
  53.6%; projection 0.59x = SLOWDOWN vs the 2.8x bar. WAVE
  SYNTHESIS: P-A/P-B/P-C all killed on measurement; P-D identified
  the mechanism (dual Phase-1, 40th). The ONLY flip path is now
  NATIVE DUAL PHASE-1 (~-25% pivots, the sure factor) STACKED with
  one unprobed multiplier: E (support-contiguity reordering,
  glm+opus+contrarian convergent, claimed -7..-20%) or F
  (Schur/bordered-block basis factorization, glm+opus convergent,
  claimed -11..-25%; opus measured greenbea as bordered-staircase:
  87% local columns + dense border, RCM bandwidth 2363->1422).
  Phase-1 alone projects greenbea ~1.28 on-host — the largest
  single narrowing available. NEXT WAVE DISPATCHED: U-P1 (native
  dual Phase-1 build, staged) in parallel with probes P-E and P-F.

- FORTY-THIRD SETTLED — SCHUR/BORDERED FACTORIZATION (P-F) KILLED
  ON AN INVARIANT (2026-07-18, opus killing its own family's idea,
  audit clean; experiments/probe_schur_2026_07_18.md): rho = B^-T
  e_r is the UNIQUE solution fixed by (B,r) — its nnz is
  factorization-independent (verified: zero nnz disagreement across
  NATURAL/COLAMD/MMD orderings on real trajectory bases at pivots
  500/2000/3500, including the densest rows at 72%). greenbea's
  59-94% solve-vector densities are the TRUE BTRAN results on the
  rows the DS visits, not border-fill symptoms. Additionally: real
  bases are one giant connected component (88.7-95.7%) — no thin
  border exists (10-15% border needed for sub-10% blocks); bordered
  orderings carry +30% MORE factor fill than COLAMD. Class F dead;
  the deep fact stands: greenbea's basis inverse is intrinsically
  dense on the visited rows — per-pivot work has a hard floor near
  the current kernels. The flip now rides entirely on U-P1 (native
  dual Phase-1, fewer pivots) x P-E (locality — its cache-model
  gate must now overcome both the P-B cache-residency finding and
  this intrinsic-density result).

- CHRONICLE CAUGHT UP (2026-07-16, codex + orchestrator): artifact
  ingestion added to replay_bench.py (idempotent, artifact-keyed
  tables); /tmp bench artifacts rescued into assets/ (knife chunks,
  6ec6e2e/957347b/82cd31d/ecf94bd/7e9947a); CAMPAIGN.md + embedded
  report regenerated through the canonical board; gh-pages 55fb707
  and the Claude artifact republished. Per-ship local replay rows for
  26a9359 (setup fast-path) and 82cd31d (native V2 port) still
  pending — queued behind the current kernel probes (needs a quiet
  box).
