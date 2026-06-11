# Sparse PDHG Performance Handoff

**Date:** Wednesday, June 10, 2026 (parallel-experiment update)
**Primary worktree:** `/home/evan/dev/linprogx`
**Primary branch:** `sparse-support`
**Supersedes:** the May 18, 2026 handoff (column equilibration / tuned polish era)

## June 10 Update: Parallel CYCLE-gap Experiments

Three experiments ran in parallel worktrees and were integrated:

1. **Doubleton-row presolve (WIN, integrated as `src/linprogx/presolve.py`).**
   Dependency-free presolve: empty rows, cascading singleton rows, and
   doubleton-row substitution `x_p = (b_i - d*x_q)/a` with bound mapping and
   postsolve replay. Fill limit `max_fill=5` is critical (fill 2 and 10 both
   fail to help; conditioning, not size, is what matters). Wired into
   `SparseSolver` behind `presolve=True` (default). CYCLE: removes 388 rows /
   360 cols and converges via FULL KKT at 36k iterations, delta 2.8e-6.
   DFL001: removes 15/15, 27.9k iterations, delta 0.16.
2. **Plateau early-exit (integrated, dormant insurance).** Ring buffer of
   best-seen relative KKT per eval; if <2% improvement over the last 80 evals
   and the best iterate is within 50x tol primal residual, adopt the best
   iterate and exit. `plateau_window`/`plateau_threshold` kwargs; result dict
   reports `plateau_exit`. With presolve active it never fires on the
   benchmarks; it exists for degenerate shapes presolve cannot fix.
3. **eval_interval 64 -> 40 (reverted; interaction lesson).** A clear win on
   the UNPRESOLVED CYCLE (full KKT at 39.9k iters) but a loss on both
   presolved problems (CYCLE plateau-exits prematurely at delta 1.1e-3,
   DFL001 regresses to 35k iters). The restart trajectory is chaotic in
   these parameters; tune them only jointly with presolve. The
   `eval_interval_override`, `restart_*` and `debug` kwargs from this
   experiment were kept as tooling. The diagnosis stands: long restart
   epochs let omega drift fatally (e.g. 0.027 -> 0.004) when the sufficient
   criterion is unreachable and the artificial restart is ~17k iters away.

**Current committed results:** DFL001 5.34s delta 1.6e-1 (HiGHS 6.38s,
Clarabel 8.06s — linprogx fastest); CYCLE 1.39s delta 2.8e-6 via full KKT
(HiGHS 0.18s, Clarabel 0.22s — still ~7x slower, expected simplex territory).

### Second round: remaining CYCLE ideas, all exhausted

A follow-up round swept every remaining identified idea on the PRESOLVED
problems. Every knob is already at its optimum and every dynamics variant
regressed; the conclusion is that ~36k iterations is what this restarted
PDHG costs on CYCLE, and further gains need a different algorithm class
(e.g. a simplex/crossover endgame), not more tuning.

- Restart constant sweeps (sufficient 0.1-0.4, necessary 0.7-0.9,
  artificial 0.2-0.55, eval_interval 48-128): the defaults
  (0.2/0.8/0.36/64) are the optimum of a chaotic landscape; most neighbors
  fail outright. rs=0.10 saves ~2% (noise).
- max_fill sweep (3-12): 3-5 give identical reductions and the best result;
  6+ degrade quality and can break convergence.
- omega seeds (0.01-10): best non-default (0.1) saves ~6%, within noise.
- The debug trajectory shows where CYCLE's iterations go: ~60% of the run
  (iters ~3k-25k) is a second omega down-spiral to 3.5e-5 before omega
  climbs back to its correct level ~2e-2, after which convergence is
  explosive (KKT 3.8e-3 -> 8.4e-5 in ~500 iterations). Two targeted fixes
  for that spiral BOTH regressed:
  - replacing the movement update with residual-balance steps when the KKT
    error is lopsided (CYCLE infeasible at 50k; the early omega descent
    needs the movement signal),
  - clamping the per-restart movement update to a factor 4 (CYCLE
    infeasible at 50k, DFL001 +7% iterations).
- Duplicate-row removal in presolve: finds ZERO rows on either benchmark
  after the singleton/doubleton cascade (the 13 raw duplicates get consumed
  by it) and only adds presolve time. Implemented, measured, removed.

### Third round: a different algorithm class (June 10, continued)

Tried and negative:

- **Active-set crossover from the PDHG point** (3 prototype variants in
  scipy/lsqr: primal-proximity faces, dual-reduced-cost faces, combined).
  On these degenerate problems the predicted face never stabilizes
  (CYCLE: free count < row count -> inconsistent least squares, pres ~1e1+;
  DFL001: dual residual stuck ~0.6). Doing crossover properly requires
  basis management (rank-revealing LU, pushes, ratio tests) — i.e. building
  simplex itself. Not a quick add-on.
- **Halpern anchoring** (blend accepted step toward the cycle anchor with
  weight 1/(k+2), replacing within-cycle averaging): CYCLE infeasible at
  50k, DFL001 +46% iterations. Reverted; averaging + restarts wins again.

Validated and promising:

- **The C result dict now exposes the dual vector `y`** (original units).
- **Mehrotra predictor-corrector IPM prototype**
  (`experiments/ipm_prototype.py`, scipy splu on the regularized normal
  equations, Ruiz + cost scaling, native boxes, zero-width boxes pinned):
  - CYCLE: **34 iterations, delta 1.7e-7, true residual 1.8e-11, 2.4s in
    Python** — HiGHS-class accuracy; a C port should land ~0.2-0.5s.
  - DFL001: normal equations fill badly (98s in Python, wobbly tail,
    delta 2.6e4) — IPM is the wrong tool there, PDHG already beats HiGHS.
  - Conclusion: portfolio architecture (IPM for small/degenerate, PDHG for
    large sparse), which is standard practice, not overfitting.

**Next planned build (multi-session): dependency-free C IPM**
1. Sparse Cholesky module: minimum-degree ordering, elimination-tree
   symbolic factorization, up-looking numeric factorization, triangular
   solves, static + dynamic regularization. Test against scipy on the
   benchmark ADA' matrices.
2. Mehrotra driver in C mirroring the validated prototype exactly.
3. `algorithm="ipm"` in SparseSolver plus size-based routing
   (small/degenerate -> IPM, large -> PDHG), benchmarks, README.

## High-Level Status

The sparse PDHG solver was rewritten from a hand-tuned fixed-step Chambolle-Pock
loop into a restarted average PDHG with:

- Ruiz equilibration (10 inf-norm passes plus one l2 pass) replacing the old
  median-normalized column-norm scaling
- restarted iterate averaging with sufficient/necessary/artificial restart
  criteria on a scale-free relative KKT error
- an adaptive primal weight `omega` (movement-ratio update at restarts with a
  residual-balance safeguard) replacing the manual `objective_scale` tuning
- an adaptive step size (accept/shrink linesearch on the local bound
  movement/interaction, x-first update ordering)
- KKT-based termination (primal residual, dual residual, duality gap) measured
  in original problem units
- a final status convention that stays feasibility-based for backward
  compatibility: a primal-feasible end point reports `optimal`

All per-problem tuning is gone. Both Netlib benchmarks run with identical,
untuned solver settings (`max_iterations=50_000`, `eps=2e-5`).

## Current Benchmark Results (committed artifacts)

### Netlib DFL001 (6071 x 12230, 35632 nnz)

| Solver | Status | Delta vs published | Runtime |
| --- | --- | ---: | ---: |
| linprogx-sparse | optimal | 1.974e+00 (1.8e-7 relative) | 5.751s |
| SciPy/HiGHS | optimal | 3.286e-04 | 6.240s |
| Clarabel | optimal | 3.109e-02 | 7.986s |

**linprogx-sparse is now the fastest solver on DFL001 on this machine.** It
converges by full KKT termination at ~32k iterations with max equality residual
1.9e-05.

### Netlib CYCLE guardrail (1903 x 3371, 21234 nnz, degenerate, b = 0)

| Solver | Status | Delta vs published | Runtime |
| --- | --- | ---: | ---: |
| linprogx-sparse | optimal | 8.817e-04 | 2.035s |
| SciPy/HiGHS | optimal | 5.898e-12 | 0.184s |
| Clarabel | optimal | 8.174e-10 | 0.244s |

CYCLE solves with the same untuned settings (old committed result needed
`objective_scale=6e-5`, 110k iterations, a feasibility polish phase, and got
delta 6.6e-3 in 5.0s). It is still ~11x slower than HiGHS: the KKT gap
plateaus around 1e-3 relative and never passes the 2e-5-relative gap test
within 50k iterations, so the run consumes its full budget and certifies via
primal feasibility. This appears intrinsic to PDHG on this degenerate shape
(see negative results below).

## What Changed On `sparse-support` This Session

Commits:

- `ebb0670 Replace tuned sparse PDHG with restarted adaptive solver`
- `6183f39 Speed up sparse PDHG kernels with int32 operator and -O3`

Key code structure in `src/linprogx/_csparse.c`:

- `ScaledOp` struct: the equilibrated operator stored once per solve with
  **32-bit inner indices** and restrict-qualified `scaled_op_matvec` /
  `scaled_op_transpose_matvec` kernels.
- `evaluate_kkt(...)`: computes primal residual (max + l2), dual residual
  (inf + l2), primal/dual objectives, gap — all in original units — plus a
  scale-free relative `kkt` used for restart decisions and candidate
  selection. Two matvecs per call.
- `kkt_terminated(...)`: primal max residual <= tol (absolute), dual inf
  residual <= tol*(1+||c||_inf), |gap| <= tol*(1+|p|+|d|).
- Main loop (`CSRMatrix_solve_eq_box_pdhg`):
  - x-first update with extrapolated dual gradient `2*A*x_new - A*x`.
  - Adaptive step: trial step with current eta; accept if
    `eta <= movement/|interaction|`, else shrink `(1-(k+1)^-0.3)` and retry;
    accepted steps may grow eta by `(1+(k+1)^-0.6)`. Rejection rate measured
    at ~1-2% (`step_trials` in the result dict).
  - Cached `ax = A*x`, `aty = A'*y`; pointer swaps commit trial buffers; one
    matvec + one transpose matvec per accepted iteration.
  - Every 64 iterations (`eval_interval`): evaluate current and average
    iterates, pick the better by relative KKT, check termination, then apply
    restart rules (sufficient 0.2 / necessary 0.8 with stall / artificial
    0.36*total).
  - On restart: primal weight `omega <- exp(0.5*log(||dy||/||dx||) +
    0.5*log(omega))` plus the safeguard: if the relative primal residual
    exceeds 20x the relative dual residual **and** 20x the relative gap,
    `omega *= 2`; if the relative dual residual exceeds 20x the relative
    primal residual, `omega *= 0.5`. Clamped to [1e-8, 1e8].
  - `tau = eta/omega`, `sigma = eta*omega` throughout.
- `objective_scale` (Python kwarg) now seeds `omega` and is otherwise unused;
  `adaptive_weight` kwarg (int, default 1) exists for experiments
  (0 = frozen omega, 2 = residual-balance-only update — known bad, kept only
  as an experiment hook).
- Active-set CGLS cleanup retained as a fallback: `max_passes=12`,
  `max_iter=600` per pass, breaks when a pass improves the l2 residual by
  less than 1%.
- Result dict gained: `primal_weight`, `dual_residual`, `gap`, `restarts`,
  `step_trials`.
- `pyproject.toml`: both C extensions build with `-O3`.

Benchmarks/tests:

- `bench_large.py` / `bench_cycle.py`: both use `max_iterations=50_000`,
  `eps=2e-5`, no `objective_scale`.
- `tests/test_large_benchmark.py`: the tuned-polish CYCLE test became
  `test_cycle_sparse_pdhg_untuned_reaches_benchmark_quality` (50k budget, no
  tuning, optimal, residual <= 2e-5, delta <= 1e-2).
- `bench_sparse_fast.py` small cases all solve optimally now, including
  `random_feasible unit` which previously hit the 4000-iteration limit
  (now ~700 iterations, sub-millisecond).

## What We Tried And Learned This Session

### Confirmed useful (in merge order of impact)

1. Restarted averaging + adaptive primal weight: DFL001 9.5s -> ~7s untuned;
   removed all per-problem `objective_scale` tuning.
2. Scale-free relative KKT for restart/candidate decisions. The first
   implementation weighted the KKT error by omega; with tiny omega the
   "best" candidate could be wildly primal-infeasible. Normalizing each
   component (residuals by 1+||b||, 1+||c||; gap by 1+|p|+|d|) fixed
   incoherent restart decisions across omega updates.
3. Ruiz equilibration: the difference between CYCLE converging (objective
   delta 1e-6 territory at 120k fixed-step iterations) and plateauing at
   1e-1. Slightly worse for DFL001 alone, strongly net positive.
4. Residual-balance safeguard on the omega update: without it CYCLE's omega
   spirals down (low omega -> big primal steps -> ||dx|| dominates -> omega
   keeps falling, a runaway feedback loop; the movement ratio is
   ~omega^2-dependent so the fixed point is unstable). The safeguard fires
   only on 20x lopsided residuals; requiring the gap also be 20x smaller for
   the upward nudge is what kept DFL001 unaffected (variant C). CYCLE went
   from 125k iterations to ~30k-50k.
5. Adaptive step size: DFL001 45.9k -> 31.9k iterations. Rejections ~1%.
6. int32 operator indices + restrict + fused movement accumulation + -O3:
   ~10% wall clock.
7. Multi-pass CGLS cleanup (active-set refresh between bound-limited steps)
   with progress-based early exit; cheap insurance, certifies feasibility
   when the loop ends primal-infeasible but near-optimal.

### Tried but NOT a win (do not redo without new ideas)

- **Residual-balance-only omega update** (`adaptive_weight=2`): catastrophic
  on DFL001 (omega driven the wrong way by the gap term; stuck at delta 1e8).
- **omega-weighted KKT** for candidate selection/restarts: incoherent when
  omega changes between restarts; replaced by relative KKT.
- **Presolve v1 on CYCLE** (empty rows, cascading singleton rows -> fixed
  variables, duplicate rows up to scaling): removes 140 rows / 112 cols /
  112 fixed vars in 5 rounds, but iteration count and runtime are
  unchanged (still 50k iters, ~2.0s). CYCLE's slowness is not in this
  removable structure. Prototype only; no module built.
- **Boxing CYCLE's 7 free variables** with implied bounds (1e3 / 1e4 caps):
  either distorts the optimum or worsens conditioning. Dead end.
- **-march=native**: only ~4% over -O3 and perturbs FP enough to change
  iteration counts; not shipped.
- (Previous sessions: Rust port, l1 diagonal preconditioning, LSQR-only
  cleanup — see git history of this file.)

### Known quirks / debts

- CYCLE never satisfies the relative-gap test within 50k iterations (gap
  oscillates ~1e-3 relative from 30k to 120k+ regardless of step/weight
  settings probed). Its `optimal` status comes from the feasibility-based
  final convention. The benchmark budget (50k) is therefore the runtime.
- The eval-of-current-iterate inside the loop recomputes `ax`/`aty`
  needlessly (2 extra matvecs per 64 iterations, ~3%): harmless, easy
  micro-optimization if wanted.
- `iterations` counts accepted PDHG iterations; `step_trials` counts all
  linesearch trials.

## Environment / Commands

```bash
cd /home/evan/dev/linprogx
uv sync --extra dev
uv pip install -e . --force-reinstall   # ALWAYS after touching the .c file
just ci                                  # 70 tests
uv run python bench_large.py             # DFL001 artifacts
uv run python bench_cycle.py             # CYCLE artifacts
uv run python bench_sparse_fast.py --iterations 4000 --repeats 10
```

Stale-`.so` warning from the previous handoff still applies: rebuild the
editable install after every C edit before benchmarking.

Useful raw probe (full diagnostics incl. gap/omega/restarts):

```bash
uv run python -u - <<'PY'
from bench_cycle import DATA_PATH, EXPECTED_CYCLE_OBJECTIVE, _bounds, load_cycle
import time
data = load_cycle(DATA_PATH)
bounds = _bounds(data)
lo = [float("-inf") if l is None else float(l) for l, _ in bounds]
hi = [float("inf") if u is None else float(u) for _, u in bounds]
start = time.perf_counter()
r = data["A"].solve_eq_box_pdhg(data["c"].tolist(), data["b"].tolist(), lo, hi,
                                max_iter=50000, tol=2e-5, check_interval=50000)
print(time.perf_counter() - start, {k: v for k, v in r.items() if k != "x"})
PY
```

## Suggested Next Steps

1. **CYCLE tail convergence** is the only remaining parity gap. Ideas not yet
   tried: PDLP-style localized duality-gap restart measure (instead of the
   KKT-error ablation), doubleton-row substitution presolve (329 rows with
   b=0 allow `x_p = -(c/a) x_q` merges — the only presolve class with real
   mass left), or accepting that degenerate small LPs are simplex territory.
2. Consider terminating CYCLE-like runs early once the relative KKT stops
   improving across several restarts (plateau detection) instead of burning
   the full budget; would cut CYCLE to ~1.2s at identical quality.
3. Micro: skip the redundant current-iterate matvecs in the eval (cache
   pass-through), OpenMP for matvecs if the dependency-free story allows.
4. A second large LP (e.g. another Netlib instance at DFL001 scale) would
   guard against overfitting the restart constants to DFL001.

## Guardrails

- Keep DFL001 and CYCLE both in the table; never optimize one alone.
- Do not loosen `eps=2e-5`.
- `iteration_limit` with good objective still counts as not solved —
  equality residual is the gate.
- Keep HiGHS and Clarabel rows in every benchmark run.
- The C sparse path stays dependency-free.
