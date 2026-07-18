# greenbea IPM stall anatomy — 2026-07-18

## Probe contract

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`.
- Route under test: presolved native IPM, forced directly through the C
  `solve_eq_box_ipm` path corresponding to `SparseSolver(algorithm="ipm")`.
- Accuracy contract: `eps=2e-5`; the native solve uses `tol=1e-9` and
  `feas_tol=2e-5`.
- No solver source or network access was used.
- Reduced problem: 1,525 rows, 3,868 columns, 23,274 nonzeros.
- Baseline diagnostics: `LINPROGX_IPM_ANATOMY=1`, `LINPROGX_IPM_SLICE=1`,
  `LINPROGX_IPM_LOOP_PROFILE=1`, and native `debug=True`.

## Anatomy written before remedies

The failure is **dual-certificate infeasibility followed by a non-finite
Newton direction**. It is not mu stagnation, primal infeasibility, step
collapse, or the iteration-60 pace watchdog.

The primal/barrier trajectory converges strongly. Mu falls from `2.479e2`
at iteration 0 to `3.013e-9` at iteration 58. The original-unit primal
residual falls from `3.395e5` to `7.942e-10`. Steps are not collapsing:
iterations 55–57 take primal/dual steps `(0.924, 0.751)`, `(0.856, 0.982)`,
and `(1.000, 0.999)` before the fraction-to-boundary factor.

The dual certificate does not converge. The true Lagrangian gap is infinite
at every sampled iteration because at least one reduced cost points toward
an infinite bound. The scaled dual residual reaches `1.831e-6` at iteration
53, then floors near `1.815e-6`. The number of infinite-side sign violations
falls from 1,846 at iteration 0 to 15 at iteration 30, rises to 27 at
iteration 40, and ends at 9 at iteration 58. The worst violation falls to
`7.229e-7` at iteration 53 but then stalls at `4.287e-7` through iteration
58—far above the certificate sign tolerance (`1e-9 * (1 + |c_j|)`).

| Iteration | mu | raw primal residual | scaled dual residual | bad infinite-side reduced costs | worst violation | certified gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.479e2 | 3.395e5 | 1.016e0 | 1,846 | 1.513e-1 | inf |
| 10 | 8.635e1 | 4.750e4 | 1.775e-1 | 104 | 4.388e-2 | inf |
| 20 | 4.578e1 | 7.487e3 | 2.056e-2 | 69 | 5.577e-3 | inf |
| 30 | 1.851e1 | 7.750e2 | 1.215e-3 | 15 | 4.271e-5 | inf |
| 40 | 4.575e-1 | 8.965e-2 | 9.084e-5 | 27 | 3.082e-4 | inf |
| 50 | 2.447e-3 | 2.898e-6 | 2.853e-6 | 34 | 6.439e-6 | inf |
| 53 | 3.246e-4 | 4.701e-7 | 1.831e-6 | 25 | 7.229e-7 | inf |
| 55 | 3.401e-5 | 9.267e-7 | 1.818e-6 | 23 | 4.286e-7 | inf |
| 56 | 6.663e-6 | 8.807e-8 | 1.816e-6 | 31 | 4.287e-7 | inf |
| 57 | 5.016e-7 | 1.295e-8 | 1.815e-6 | 31 | 4.287e-7 | inf |
| 58 | 3.013e-9 | 7.942e-10 | 1.815e-6 | 9 | 4.287e-7 | inf |

At iteration 58 the affine solve returns a wholly non-finite direction:
3,868/3,868 `dx`, 1,525/1,525 `dy`, 3,868/3,868 `dzl`, and 257/3,868
`dzu` values are non-finite. Consequently `mu_aff` and `sigma` are NaN.
The ratio tests leave both steps at 1.0 because comparisons with NaN are
false. The existing endgame mu safeguard records one shrink but no break;
its `post_mu > 10 * pre_mu` comparison is also false for NaN, so the NaN
step is committed. Iteration 59 observes NaN residuals and exits, restoring
the iteration-58 best iterate. Exit state:

```text
status=iteration_limit
iterations=59
best_gap=inf
best_pres=1.091e-13
best_raw=7.942e-10
best_dres=1.815e-6
best_mu=3.013e-9
objective=-72464559.56495331
wall=0.657s (diagnostic run)
```

The loop profiler attributes about 0.112 s to exit-gate work. Slice timings
on the diagnostic run were 0.090 s refactor, 0.017 s triangular solves,
0.025 s residual matvecs, and 0.511 s other. These are anatomy timings with
heavy diagnostics and failed cleanup attempts, not the remedy performance
gate.

## Remedy decision and results

The anatomy supported one standard remedy family: adaptive primal-dual
regularization of the late normal equations. It did not support a looser
certificate, primal polish, a different mu-progress heuristic, or another
dual cleanup: primal feasibility and mu were already converged while the
dual sign certificate was not.

The implementation is behind `LINPROGX_IPM_ADAPTIVE_REG` and is default off.
Its trigger is global and problem-independent: a non-finite affine Newton
direction. On that signature it retries the same affine solve with a
geometric regularization ladder, increasing the bound-space Hessian floor
and the row-space diagonal together. Exhaustion fails closed before a NaN
direction reaches the ratio tests. `LINPROGX_IPM_ANATOMY` gates the diagnostic
trace and is also default off.

### Attempt 1: row-space regularization only

Increasing only the row-space diagonal did not make the iteration-58
direction finite at `delta = 3.013e-9`, `3.013e-7`, or `3.013e-5`. This
falsified a Cholesky-row-diagonal-only explanation and pointed to the
complementarity-derived `D` scaling upstream of the normal equations.

### Attempt 2: full primal-dual regularization

Increasing both regularization terms made the direction finite on the
second retry at iteration 58 (`delta = 3.013e-7`) and prevented the immediate
NaN. It did not repair the certificate. At iteration 59 the same nine sign
violations remained, with the same `4.287e-7` maximum. By iteration 60 the
dual residual was still `1.815e-6` and the gap was still infinite. Subsequent
steps collapsed, primal feasibility worsened, and the run reached the
200-iteration budget after 51 adaptive retries.

Uninstrumented result:

```text
status=iteration_limit
iterations=199
wall=12.716s
objective=-72464906.00541723
best_gap=inf
adaptive retries=51
```

This fails every greenbea remedy gate: no optimal certificate, more than 60
iterations, and much more than 0.30 s.

### Oracle and timing facts

Local SciPy/HiGHS execution (oracle use only; no source inspection) returned:

```text
status=optimal
objective=-72555248.12984593
wall=0.266s
```

The baseline IPM best point was `90688.565` high (`1.24992e-3` relative), and
the regularized point was `90342.124` high (`1.24515e-3` relative). Both miss
the `2e-5` oracle-equality gate by orders of magnitude. Three uninstrumented
baseline IPM runs were 0.747 s, 0.701 s, and 0.675 s, all
`iteration_limit` at iteration 59 with the identical objective
`-72464559.56495331`.

The unchanged public auto route used dual simplex and returned certified
optimality in 0.463 s, 4,399 iterations, original-unit equality residual
`1.769e-7`, and objective `-72555248.12984592`. No public routing change was
made; because the IPM remedy did not converge, the stall-risk predicate
must continue to route greenbea to dual simplex.

### Inertness and regression gates

With the experiment gate off versus on, seven other IPM fixtures had
identical status, iteration count, and objective. None activated the
non-finite-direction remedy before certification:

| Fixture | Status | Iterations off/on | Objective off/on |
| --- | --- | ---: | ---: |
| degen3 | optimal | 17 / 17 | -862.613643378328 |
| cre_a | optimal | 34 / 34 | 23595585.417344872 |
| d2q06c | optimal | 47 / 47 | 69390.8853005068 |
| ken_07 | optimal | 15 / 15 | -560107625.9972683 |
| 80bau3b | optimal | 44 / 44 | 529334.8866795273 |
| woodw | optimal | 32 / 32 | 1.3045073685961155 |
| maros_r7 | optimal | 20 / 20 | 693586.1673729181 |

Full pytest after the C rebuild:

```text
======================= 379 passed, 7 skipped in 38.83s ========================
```

## Verdict: KILLED

The failure class is now documented: greenbea reaches a primal/barrier
solution but not a dual-feasible Lagrangian certificate; the remaining
infinite-side reduced-cost error floors before the normal equations become
non-finite. Adaptive row-space regularization does not stabilize the solve,
and full primal-dual regularization stabilizes it without moving the pinned
dual sign error. Two principled attempts failed the certificate, iteration,
wall, and oracle gates, so the G1 probe is killed under the campaign rule.
