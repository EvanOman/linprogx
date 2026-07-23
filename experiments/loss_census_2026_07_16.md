# V3 loss census — 2026-07-16

## Scope and protocol

This is a fresh local measurement pass over `greenbea`, `osa_14`, `osa_60`,
`pds_10`, `woodw`, `80bau3b`, `cre_a`, and `stocfor3`. The v3 board selected
the set, but no board timing or ratio is reused below as a measurement.

The environment was rebuilt before measurement with:

```text
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
```

Every solver invocation was sequential and bounded at 120 seconds. Timings are
single-host, mostly single-shot census measurements, not scoreboard-grade
paired medians. The throwaway runner lived at `/tmp/loss_census_runner.py`; no
solver source was changed. `osa_60` exceeded the aggregate 120-second budget of
`ipm_other_profile.py` because that script performs several solves, so its IPM
cell uses one full debug run instead. No individual solve was allowed past 120
seconds.

## Where the wall goes

`linprogx public` is the uninstrumented `SparseSolver(algorithm="auto")` wall.
Direct route profiles exclude the Python wrapper and current presolve. The
three instances labelled DS by the campaign took IPM on this local rebuild;
both the local IPM and a direct Dantzig profile are therefore shown.

| Instance | linprogx public, route, iterations | Direct phase attribution from this run | HiGHS default, route, iterations | Advantage decomposition |
| --- | --- | --- | --- | --- |
| greenbea | 0.4160s, DS/Dantzig, 4,399 | 0.4418s direct: pivot row 108.45ms (24.8%), BTRAN 82.91ms (18.9%), FTRAN 78.30ms (17.9%), ratio 65.47ms (14.9%), reduced-cost update 42.63ms (9.7%), LU update 26.84ms (6.1%), refactor 23.94ms (5.5%); 1/4,399 degenerate pivots | 0.2446s, dual simplex, 2,836 | HiGHS uses 35.5% fewer pivots. End-to-end wall per pivot is 86.2us for HiGHS versus 94.6us public / 100.4us direct for linprogx, so both pivot count and per-pivot work contribute. |
| osa_14 | 1.8506s, IPM, 55 | 0.7905s direct: refactor 0.400s (50.6%), Newton solves 0.140s (17.7%), loop misc 0.126s (15.9%), setup 90.6ms (11.5%); current presolve separately measured 1.0721s | 0.9805s, dual simplex, 2,895 | The algorithms' iteration counts are not comparable. Our IPM core is already faster than the entire HiGHS solve; the local loss is created before it, with current presolve equal to 57.9% of the 1.8506s public wall. |
| osa_60 | 26.6196s, IPM, 56 | 4.5405s direct: refactor 1.920s (42.3%), setup 1.0335s (22.8%), Newton solves 0.950s (20.9%); current presolve measured 21.9160s | 17.3023s, dual simplex, 13,623; forced HiGHS IPM 13.2374s, 40 IPM + 136 crossover iterations | Our IPM core is 3.8x faster than HiGHS default, but current presolve is 82.3% of public wall. HiGHS also chooses the slower of its two measured routes here: forced IPM is 23.5% faster than its default DS. |
| pds_10 | 2.2708s, PDHG, 8,576 | 2.0034s direct: PDHG step 1.6516s (82.4%), transpose 0.2175s (10.9%), eval 69.5ms (3.5%), accumulate 42.7ms (2.1%); 20 restarts and 8,604 step trials | 1.2651s, dual simplex, 7,508 | The 0.2674s public/direct difference is 11.8% of public wall. The remaining wall is the repeated sparse step, against a problem HiGHS presolves much further. |
| woodw | 0.1004s, local IPM, 32 | Local IPM 0.0784s direct: refactor 40ms (51.0%), Newton 10ms (12.8%), loop misc 18.8ms (24.0%), setup 6.2ms. Direct certified-route Dantzig: 0.1051s, 1,338 pivots; pivot row 31.5%, ratio 21.7%, BTRAN 12.1%, FTRAN 12.3%, reduced-cost update 11.8%; 0 degenerate pivots | 0.0904s, dual simplex, 828 | Local IPM is within 10.1ms end-to-end. On Dantzig, linprogx has lower direct wall/pivot (78.6us) than HiGHS end-to-end (109.1us), but uses 61.6% more pivots. |
| 80bau3b | 0.1791s, local IPM, 62 | Local IPM 0.1544s direct: refactor 80ms (51.8%), Newton 30ms (19.4%), loop misc 31.1ms (20.1%), setup 9.2ms. Direct certified-route Dantzig: 0.5616s, 7,244 pivots; ratio 31.5%, pivot row 24.8%, BTRAN 16.8%, reduced-cost update 9.6%, FTRAN 8.2%; 0 degenerate pivots | 0.1752s, dual simplex, 3,285 | Local public IPM is within 3.9ms. On Dantzig, HiGHS uses 54.7% fewer pivots and 53.3us end-to-end per pivot versus 77.5us direct for linprogx. |
| cre_a | 0.0946s, IPM, 36 | 0.0838s direct: refactor 40ms (47.7%), Newton 20ms (23.9%), setup 17.3ms (20.6%); current presolve 6.9ms | 0.0864s, dual simplex, 1,493 | The local public gap is 8.2ms. HiGHS' forced IPM is slower at 0.1819s (35 IPM + 59 crossover iterations); its advantage is presolve plus its DS route, not a better IPM loop. |
| stocfor3 | 0.6691s, local IPM, 45 | Local IPM 0.5181s direct: refactor 0.250s (48.3%), Newton 0.140s (27.0%), loop misc 50.5ms (9.7%), setup 51.7ms. Direct certified-route Dantzig: 1.4352s, 9,630 pivots; ratio 34.0%, BTRAN 24.7%, FTRAN 15.5%, refactor 9.2%, LU update 8.0%; 0 degenerate pivots | 0.5781s, dual simplex, 12,313 | Dantzig uses 21.8% fewer pivots than HiGHS but costs 149.0us/pivot versus HiGHS' 47.0us end-to-end. The much larger linprogx matrix, not degeneracy or pivot count, is the first-order distinction. |

The DS rate histograms reinforce the structural split. `greenbea` has median
rho/alpha/ratio-candidate counts of 897/3,625/182; `80bau3b` has
96/770/96; `stocfor3` has 7/68/7. The DS family is therefore not one kernel
shape: `greenbea` is support-dense, while `stocfor3` loses on matrix size and
per-pivot overhead despite tiny pivot supports.

## Presolved structure and why V2 stops

Degree percentiles and density are for linprogx's current presolved problem.
“Active” is the fraction of the returned original-space solution within
`1e-7 * max(1, |bound|)` of a finite bound. Every residual bipartite matrix had
one nontrivial connected component, so disconnected-component splitting has a
measured ceiling of zero on this set.

| Instance | Raw -> current presolved `(m,n,nnz)`; density | Degree pattern after presolve | Bound/degeneracy proxy | What current presolve did | Measured reason it stops |
| --- | --- | --- | --- | --- | --- |
| greenbea | (2,392, 5,598, 31,070) -> (1,525, 3,868, 23,274); 0.3946% | row p50/p99 5/91.8; col p50/p99 6/17; 338 singleton columns | 83.2% active; 1 degenerate DS pivot | removed 867 rows/1,730 cols: 1,102 forcing columns, 438 doubletons, 113 fixed columns, 47 singleton rows, 23 column singletons, 7 duplicate columns | all 338 remaining singleton columns have nonredundant bounds; 4 doubletons hit the fill guard; post-V2 candidate scan is 0 rows/0 cols |
| osa_14 | (2,337, 54,797, 317,097) -> unchanged; 0.2476% | row p50/p99/max 22/2,256/38,336; col p50/p99 6/6; 2,337 singleton columns | 94.7% active | removed 0 rows/0 cols; presolve wall 1.0983s | 37 column-singleton eliminations are visible after the gate, but 37/2,337 rows and 37/54,797 cols are below the 8% count gate; 2,300 other singleton columns have nonredundant bounds |
| osa_60 | (10,280, 243,246, 1,408,073) -> unchanged; 0.05631% | row p50/p99/max 21/35/173,366; col p50/p99 6/6; 10,280 singleton columns | 79.1% active | removed 0 rows/0 cols; presolve wall 21.9160s | 37 column-singleton eliminations are visible after the gate; 10,243 other singleton columns have nonredundant bounds |
| pds_10 | (16,558, 49,932, 107,605) -> (14,438, 47,812, 103,230); 0.01495% | every coefficient has `|a|=1`; row p50/p99 5/30; col p50/p99 2/4; 38,852 degree-2 columns | 83.4% active; 14,669 zero-cost columns | removed 2,120 rows/2,120 cols: 1,540 doubletons and 580 singleton rows | post scan finds 242 fixed columns and 88 singleton-row cascades; 1,051 remaining singleton columns have nonredundant bounds; a forced second pass removes only 1,056 nnz (1.0%) |
| woodw | (1,098, 8,418, 37,487) -> (707, 5,363, 19,807); 0.5224% | row p50/p99 16/337.7; col p50/p99 3/13; 165 singleton columns | 80.5% active; 0 degenerate Dantzig pivots | removed 391 rows/3,055 cols: 3,050 forcing columns, 386 forcing rows, 4 column singletons, 1 doubleton | all 165 singleton columns have nonredundant bounds; 4 doubletons hit fill guard; post-V2 scan is 0/0 |
| 80bau3b | (2,262, 12,061, 23,264) -> (2,079, 11,878, 22,923); 0.09283% | row p50/p99 7/58.2; col p50/p99 2/6; 4,791 singleton columns | 49.0% active; 0 degenerate Dantzig pivots | removed 183 rows/183 cols: 158 doubletons and 25 singleton rows | post scan sees 530 fixed cols, 39 singleton-column candidates, 8 forcing rows/43 cols, 21 empty cols, and 3 duplicate cols; 4,752 singleton cols have nonredundant bounds |
| cre_a | (3,516, 7,248, 18,168) -> (3,041, 6,861, 17,274); 0.08279% | row p50/p99 4/36; col p50/p99 2/11; 2,835 singleton columns | 4.0% active | removed 475 rows/387 cols: 381 doubletons, 88 empty rows, 6 singleton rows | post scan sees 3 fixed cols and 6 forcing rows/56 forcing cols; the forced cascade is larger (90 rows/212 cols); 2,835 singleton cols have nonredundant bounds and 18 doubletons hit fill guard |
| stocfor3 | (16,675, 23,541, 72,721) -> (14,633, 21,499, 68,419); 0.02175% | row p50/p99 3/15; col p50/p99 2/17; 9,313 singleton columns | 1.0% active; 0 degenerate Dantzig pivots | removed 2,042 rows/2,042 cols: 2,010 doubletons and 32 singleton rows | post scan sees 513 singleton-column candidates, which cascade to 769 rows/cols; 8,800 singleton cols have nonredundant bounds and 48 doubletons hit fill guard |

Two attractive-looking structural ideas have low measured ceilings. All eight
problems remain one connected component. Cost-compatible proportional-column
aggregation finds only 126 excess columns on `greenbea` (3.3% of current
columns) and 388 on `80bau3b` (3.3%), and zero on the other six; the much larger
same-support counts on `woodw` and `80bau3b` do not share compatible normalized
coefficients and costs.

## HiGHS advantage: presolve, route, iterations, or per-iteration cost

| Instance | linprogx current presolved | HiGHS presolved | HiGHS reduction beyond raw | What the comparison says |
| --- | ---: | ---: | ---: | --- |
| greenbea | 1,525 x 3,868 x 23,274 | 951 x 3,158 x 23,609 | 1,441 rows, 2,440 cols, 7,461 nnz | HiGHS has 37.6% fewer rows and 18.4% fewer cols than linprogx but 1.4% more nnz; its 2,836 vs 4,399 pivots is the strongest measured advantage. |
| osa_14 | 2,337 x 54,797 x 317,097 | 2,300 x 54,760 x 196,716 | 37 rows, 37 cols, 120,381 nnz | HiGHS removes exactly the 37 dense singleton-column rows visible behind our gate, cutting nnz 38.0%. Route differs (HiGHS DS vs linprogx IPM), but our direct IPM core is faster. |
| osa_60 | 10,280 x 243,246 x 1,408,073 | 10,243 x 243,208 x 849,355 | 37 rows, 38 cols, 558,718 nnz | The same 37-row border accounts for 39.7% of nnz. Our forced V2 shape is 10,243 x 243,209 x 849,356—one column/nnz from HiGHS. |
| pds_10 | 14,438 x 47,812 x 103,230 | 4,092 x 32,646 x 78,216 | 12,466 rows, 17,286 cols, 29,389 nnz | HiGHS leaves 71.7% fewer rows, 31.7% fewer cols, and 24.2% fewer nnz than linprogx. This size gap feeds the PDHG step that occupies 82.4% of direct wall. |
| woodw | 707 x 5,363 x 19,807 | 557 x 4,095 x 15,823 | 541 rows, 4,323 cols, 21,664 nnz | Relative to linprogx's output, HiGHS has 21.2% fewer rows, 23.6% fewer cols, and 20.1% fewer nnz. Its Dantzig advantage is pivot count, while local linprogx IPM is already close. |
| 80bau3b | 2,079 x 11,878 x 22,923 | 1,537 x 9,876 x 20,012 | 725 rows, 2,185 cols, 3,252 nnz | HiGHS has 26.1% fewer rows, 16.9% fewer cols, and 12.7% fewer nnz than linprogx; on DS it also uses 54.7% fewer pivots. |
| cre_a | 3,041 x 6,861 x 17,274 | 1,271 x 4,971 x 16,446 | 2,245 rows, 2,277 cols, 1,722 nnz | HiGHS removes 58.2% more of the residual rows and 27.5% of residual cols while barely changing nnz; the win is dimension/setup plus DS route, not a faster IPM iteration. |
| stocfor3 | 14,633 x 21,499 x 68,419 | 6,713 x 13,579 x 50,844 | 9,962 rows, 9,962 cols, 21,877 nnz | HiGHS has 54.1% fewer rows, 36.8% fewer cols, and 25.7% fewer nnz. linprogx DS uses fewer pivots, so the 3.17x per-pivot wall difference is structural/per-iteration cost. |

## Ranked falsifiable hypotheses

### 1. Re-stage the semantic V2 gate after the classic cascade

**Mechanism.** The first opportunity decision is count-based and happens before
classic singleton/doubleton reductions create new fixed, forcing, empty, and
column-singleton opportunities. Re-evaluate the gate at the reduced shape (or
make the first pass reach the combined semantic fixpoint). This is not the
settled pass-speed fast path: it deliberately changes the reduced problem.

**Affected instances and measured size.** A throwaway call to the existing
native V2 reducer on the current presolved problem produced:

| Instance | Current -> second-fixpoint shape | Extra pass | Current direct -> second-fixpoint direct | Net projected gain after paying pass |
| --- | --- | ---: | --- | ---: |
| cre_a IPM | 3,041 x 6,861 x 17,274 -> 2,951 x 6,649 x 16,734 | 2.23ms | 36 iters / 83.81ms -> 34 / 76.63ms | 5.9%; 4.95ms, or 60% of the fresh 8.19ms local loss margin |
| 80bau3b IPM | 2,079 x 11,878 x 22,923 -> 1,992 x 11,155 x 21,798 | 4.41ms | 62 / 154.39ms -> 47 / 112.67ms | 24.2% |
| 80bau3b Dantzig | same | 4.41ms | 7,244 / 561.65ms -> 6,987 / 451.90ms | 18.8% |
| stocfor3 IPM | 14,633 x 21,499 x 68,419 -> 13,864 x 20,730 x 59,964 | 8.06ms | 45 / 518.08ms -> 45 / 434.91ms | 14.5% |
| stocfor3 Dantzig | same | 8.06ms | 9,630 / 1,435.23ms -> 9,604 / 1,162.41ms | 18.5% |

`osa_14` and `osa_60` are negative controls for a naive second pass: although
their direct IPM walls fall 0.7905->0.2274s and 4.5405->1.5917s, the standalone
extra reductions cost 1.1114s and 22.2655s. Do not generalize this gate to them
without a separate end-to-end win.

**Falsification probe.** Implement only the phase-order/gate change behind an
env flag, run original-space postsolve/certificate checks, and do alternating
public A/B on `cre_a`, `80bau3b`, and `stocfor3` in both IPM and Dantzig where
applicable.

**Kill criterion.** Kill if public wall improves less than 4% on `cre_a` or
less than 12% on either `80bau3b`/`stocfor3`, if either OSA instance regresses
when the gate is supposed to stay closed, or if objective/residual/status is
not equivalent at the existing `2e-5` external-oracle gate.

### 2. Make the fast IPM route certify reproducibly on the v3 DS family

**Mechanism.** The rebuilt local public route certified through IPM on
`woodw`, `80bau3b`, and `stocfor3`, even though the campaign route is DS on the
certification hosts. The next question is not a DS kernel shave: it is which
original-unit certificate or numerical branch makes the faster IPM route
host-conditional.

**Affected instances and measured size.** On the same reduced problems, direct
IPM versus direct Dantzig was 0.0784s vs 0.1051s on `woodw` (25.4% core gain),
0.1544s vs 0.5616s on `80bau3b` (72.5%), and 0.5181s vs 1.4352s on `stocfor3`
(63.9%). The local public IPM residuals were `4.46e-10`, `4.74e-11`, and
`5.23e-11`. Relative objective deltas versus HiGHS were `2.38e-5`, `4.11e-9`,
and `4.66e-8`; `woodw` is therefore the risky member, not evidence that the
route is already universally safe.

**Falsification probe.** On the same three-host v3 protocol, record the exact
IPM exit/certificate branch and run forced-IPM versus auto in each pair. Keep
the original-space residual, relative objective delta, iteration count, and
paired wall.

**Kill criterion.** Kill per instance if forced IPM fails the existing
certificate on any host, if relative objective error exceeds `2e-5`, or if the
median paired public gain is below 10%. Kill the whole family hypothesis if
the v3 hosts already take IPM and the route label was only stale attribution.

### 3. Treat the 37 OSA dense singleton rows as an IPM border, not a second presolve scan

**Mechanism.** Both OSA problems have exactly 37 eliminable singleton columns
whose rows form a very dense border: maximum row degree is 38,336 on `osa_14`
and 173,366 on `osa_60`. An IPM setup that analytically removes/reconstructs
that border (or forms its small Schur complement) could obtain the reduced
operator without paying another full V2 scan.

**Affected instances and measured size.** Removing the border cuts nnz by
120,381 (38.0%) and IPM 55->33 iterations / 0.7905->0.2274s (71.2%) on
`osa_14`; it cuts nnz by 558,717 (39.7%) and IPM 56->34 /
4.5405->1.5917s (64.9%) on `osa_60`. Against fresh public loss margins, those
core savings would cover 64.7% and 31.7% respectively; as a whole-public-wall
ceiling they are 30.4% and 11.1%.

**Falsification probe.** Build a throwaway implicit bordered operator in IPM
setup that emits the measured 2,300 x 54,760 x 196,716 and
10,243 x 243,209 x 849,356 cores and reconstructs the 37 variables. Time setup
plus solve, not solve alone.

**Kill criterion.** Kill if total public gain is below 15% on `osa_14` or 8%
on `osa_60`, if setup consumes more than the measured 0.563s/2.949s core
savings, or if postsolve changes objective/residual/status. The already-run
standalone V2 probe fails this gate and is not the implementation to pursue.

### 4. Add ranged-row/bound-propagation presolve for greenbea's bounded singletons

**Mechanism.** `greenbea` has 338 column singletons that current V2 cannot
remove because their bounds are not redundant. Eliminating such a column
turns its equality into a ranged row; native row bounds plus propagation could
unlock the missing HiGHS-style cascade without touching the closed DS reuse
families.

**Affected instances and projected size.** HiGHS reaches 951 x 3,158 versus
linprogx's 1,525 x 3,868. It then needs 2,836 pivots versus 4,399 while
greenbea has only one measured degenerate pivot. Matching only the pivot-count
reduction projects a 35.5% wall gain, worth about 86% of the fresh local loss
margin; dimension-driven per-pivot savings put the hypothesis ceiling around
35-45% wall.

**Falsification probe.** Prototype ranged-row elimination for the 338 bounded
singletons (or export an equivalent reduced model) and feed the unchanged
Dantzig solver. Record reduced shape, pivot count, every DS phase, objective,
and original equality/bound residuals.

**Kill criterion.** Kill if pivots do not fall at least 20% (below 3,520),
public wall does not fall 25%, or the ranged-row reconstruction misses the
existing certificate. A mere 3.3% parallel-column merge is not enough.

### 5. Contract the unit degree-2 network core before pds_10 PDHG

**Mechanism.** `pds_10` is a graph-like unit matrix: every coefficient has
absolute value 1, and 38,852/47,812 columns (81.3%) have degree 2. A
bound-aware graph contraction/cycle-space reduction should attack structure,
not the already-closed PDHG matvec loop.

**Affected instance and projected size.** HiGHS reduces to
4,092 x 32,646 x 78,216 from linprogx's 14,438 x 47,812 x 103,230. The simple
work proxy `m+n+nnz` falls 30.5%; applied to the 82.4%-of-wall PDHG step, that
projects a 25.1% whole-direct-wall gain, about 50% of the fresh local loss
margin before any iteration-count effect.

**Falsification probe.** Contract only provably eliminable degree-2 unit
columns behind a throwaway preprocessor, reconstruct in original space, and
run the unchanged PDHG trajectory.

**Kill criterion.** Kill if the contracted work proxy falls less than 25%,
measured public wall falls less than 15%, or the 2e-5 objective/residual gate
fails.

### 6. Low priority / effectively killed: generalized parallel-column merge

**Mechanism.** Extend duplicate-column merging to proportional coefficients
and one-sided bounds while preserving normalized objective coefficients.

**Affected instances and projected size.** The probe finds only 126 mergeable
excess columns on `greenbea` and 388 on `80bau3b`, both 3.3% of current columns,
and zero on the other six. A linear column-work ceiling is therefore 3.3%, well
below either fresh loss margin.

**Falsification probe.** A no-solve reducer should report the exact resulting
`m,n,nnz` and estimated factor work before any solver implementation.

**Kill criterion.** Keep this candidate killed unless that shape-only probe
demonstrates at least 10% nnz or factor-work reduction on one affected
instance.

## Recommendation

Start with hypothesis 1. It has an exact structural cause, exercises already
shipped semantics rather than a new numerical algorithm, and produced paid-for
gains on both certified routes. Run hypothesis 2's on-host route trace in the
same certification wave because it can invalidate or greatly enlarge the DS
scope. Hypothesis 3 is the next large architecture unit if the gate change
ships cleanly; its downstream ceiling is real, but the preparation budget is
the entire problem.
