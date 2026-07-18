# Unit U-P1 — native dual Phase-1 (2026-07-18)

## Verdict

**KILLED — S1.** Neither of the two permitted artificial-bounds designs passed
the first falsifier. The better design used 10,203 Phase-1 pivots plus 1,176
Phase-2 pivots (11,379 total) and 1.362 s, versus the knob-off baseline's 4,399
pivots and 0.409 s. The required gate was fewer than 4,200 total pivots **and**
improved wall.

The implementation remains diagnostic-only behind `LINPROGX_DS_PHASE1=1`.
Unset and `LINPROGX_DS_PHASE1=0` use the historical path and result shape.

## Evidence and design choice

The tomography supports a phase-architecture mechanism through four independent
lines:

1. HiGHS exposes a real split on the identical reduction: DuPh1 1,655, DuPh2
   1,633, and 21 cleanup pivots. Our baseline has one 4,399-pivot Dantzig loop.
2. The HiGHS options that materially move total pivots move DuPh1: dual
   feasibility tolerance changes DuPh1 from 1,559 to 2,202 while DuPh2 stays in
   a much narrower band; scaling strategy 4 cuts DuPh1 to 1,288.
3. Densifying the 89.6%-sparse objective explodes HiGHS pivot counts but barely
   moves ours, identifying a cost-sensitive Phase-1 rather than a pricing-rule
   or tie-breaking effect.
4. A transferred HiGHS Phase-1 basis cuts our pivots to 3,529, proving that
   basis geometry matters, but its foreign geometry raises solve cost from 88.8
   to 113.1 us/pivot. A viable unit therefore had to construct the basis through
   our own factorization and pricing stack.

I chose the published **artificial-bounds / box** family rather than transferring
a basis or changing the pivot kernels. Missing sides of one-sided/free bounds
receive a temporary globally scaled finite box. That makes the crash basis dual
feasible for the working boxed problem while preserving the true cost vector,
so cost sparsity and `tol` participate in the trajectory. The box expands by a
global factor of ten when it blocks feasibility, up to the historical big-M
ceiling. There is no problem-name or instance-specific parameter.

At a successful boundary the code restores every true bound, refactorizes and
reprices the Phase-1 basis, rebuilds native DSE weights when requested, and then
uses the existing Phase-2 ratio test, updates, and unchanged original-unit exit
certificate. The returned diagnostic separates Phase-1/Phase-2 pivots and loop
wall.

## Characterization before behavior change

Before changing `_csparse.c`, the following knob-off behaviors were measured and
then encoded in `tests/test_dual_phase1.py`:

| case | status | objective | pivots | residual |
|---|---|---:|---:|---:|
| bounded optimum | optimal | 1 | 0 | 0 |
| contradictory finite bounds | infeasible | 10 | 2 | 0 |
| one-sided unbounded ray | dual_infeasible | -100,000 (parked big-M point) | 0 | 0 |
| free unbounded ray | dual_infeasible | -50,000 (parked big-M point) | 0 | 0 |

The tests also require unset and `LINPROGX_DS_PHASE1=0` to match exactly across
status, objective, iteration count, residual, `x`, and `y`. Knob-on synthetic
tests require the same optimal/infeasible/unbounded statuses and verify that the
reported phase counts sum to the returned iteration count.

## S1 — prototype and greenbea gate

Settings: the cached linprogx `dual_simplex` reduction (1,525 x 3,868 x 23,274),
`leaving_rule=1`, `expand=1`, `tol=1e-8`, `max_iter=50,000`, and public
certificate epsilon 2e-5. Wall is a foreground local direct solve.

| design | Phase-1 pivots | Phase-2 pivots | total pivots | wall | pivot change vs baseline | wall change vs baseline | status/objective |
|---|---:|---:|---:|---:|---:|---:|---|
| knob off | — | 4,399 | 4,399 | 0.409 s | — | — | optimal / -72,557,668.26492292 |
| attempt 1: solve tight-box auxiliary to primal optimum | 11,377 | 0 | 11,377 | 1.489 s | +158.6% | +264.5% | optimal / identical |
| attempt 2: hand off at first true dual-feasible native basis | 10,203 | 1,176 | 11,379 | 1.362 s | +158.7% | +233.4% | optimal / rel. delta 5.1e-16 |
| HiGHS existence target | 1,655 | 1,633 (+21 cleanup) | 3,309 | — | -24.8% | — | tomography reference |

Attempt 1 showed that waiting for the boxed auxiliary to become primal optimal
merely re-solves the full objective inside the box. Attempt 2 used the proper
dual Phase-1 stopping condition: immediately restore true bounds once the basis
admits true-bound dual-feasible nonbasic assignments, leaving primal
infeasibility for Phase-2. That creates the intended split and cuts Phase-2 to
1,176 pivots, but the 10,203-pivot Phase-1 is 516% above HiGHS's 1,655 and makes
the total much worse than baseline.

**S1 gate: FAIL.** Both required predicates fail: 11,379 is not below 4,200 and
1.362 s does not improve on 0.409 s. Per the two-design cap, no third formulation
was attempted.

## S2 — post-Phase-1 solve-rate diagnostic

S2 was not formally entered because S1 is the mandatory stopping gate. The
attempt-2 instrumentation nevertheless exposes the recurring trade-against:

| metric | knob off | attempt 2 | change |
|---|---:|---:|---:|
| direct/Phase-2 loop us per pivot | 92.9 | 108.2 | +16.5% |
| whole-run FTRAN mean density | 0.24085 | 0.34465 | +43.1% |
| whole-run BTRAN mean density | 0.42794 | 0.60246 | +40.8% |

The density counters combine both candidate phases, so they are diagnostic
rather than a valid isolated S2 adjudication. The isolated 1,176-pivot Phase-2
loop timing is already 1.5 percentage points beyond the +15% kill threshold.

## S3 — generality and safety

**Not run as a performance stage: stopped at S1.** In particular, no knob-on
DS-rescue fixture battery and no alternating public-route median-of-9 campaign
were justified after the S1 kill.

Safety work completed despite the early stop:

| check | result |
|---|---|
| knob-off greenbea after implementation | 4,399 pivots, objective -72,557,668.26492292, residual 1.026e-7 |
| unset vs `LINPROGX_DS_PHASE1=0` characterization | exact status/objective/iterations/residual/`x`/`y` match |
| knob-on synthetic optimal | certificate preserved; phase counts consistent |
| knob-on contradictory finite bounds | `infeasible`, unchanged |
| knob-on one-sided and free rays | `dual_infeasible`, unchanged |
| oracle/certificate greenbea attempt 2 | optimal; objective rel. delta 5.1e-16; residual 1.007e-8 |

## S4 — public A/B ship gate

**Not run: S1 killed the unit.** Therefore there is no median-of-9 public-route
wall result, no default-on recommendation, and no SHIP-CANDIDATE claim. The
recommended default is **OFF**; retain the env gate only for diagnostic study.

## Validation

Required C rebuild commands were run from the repository after each C editing
round:

```text
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
```

Offline validation:

```text
ruff check: passed
ruff format --check: passed (56 files already formatted)
ty check: passed
Bandit medium+: passed
pytest + coverage: 531 passed, 7 skipped in 38.70s
coverage: 89.16% (floor 85%)
```

`pip-audit` / `just ci` was intentionally not invoked because dependency-audit
resolution may contact package indexes, violating this unit's no-network rule.
No web or network tool was used; build dependencies resolved from the local uv
cache. No Git operation was run.

## Files

- `src/linprogx/_csparse.c`: env-gated artificial-bounds Phase-1, clean native
  handoff, geometric box safeguard, and phase diagnostics.
- `tests/test_dual_phase1.py`: pre-change characterization, knob-off identity,
  phase accounting, and optimal/infeasible/unbounded safety tests.
- `experiments/unit_dual_phase1_2026_07_18.md`: this falsifier report.
