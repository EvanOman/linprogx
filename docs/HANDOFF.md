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
