# Proportional parallel-column falsifier on netagg-reduced `pds_10`

Date: 2026-07-18

## Verdict

**KILLED-AT-PROBE.** The requested provably exact, compatible-cost class
projects only a **2.27% column reduction** and a **2.05% nnz reduction** on the
shipped default-ON netagg output. Both are below the pre-registered 5% live
gate. No solver implementation, C change, A/B timing, or downstream
certification was attempted.

## Information used

The probe used only:

- the repository's own `src/linprogx/presolve.py`, `src/linprogx/_csparse.c`,
  `src/linprogx/sparse.py`, and existing presolve/netagg tests;
- the behavioral measurements already recorded in `docs/HANDOFF.md`, especially
  `PDS MECHANISM SETTLED + AGGREGATED-PDHG LIVE` and `NETAGG CERTIFICATION`;
- the local fixture `/tmp/lpsuite/lp_pds_10.mat`;
- textbook linear-programming algebra derived below.

No network access was used. The environment was built only with the permitted
local-cache `uv sync`; the permitted reinstall was unnecessary because no C
file changed. No external solver source code was read or fetched. No Git
operation was run.

## Exact merge algebra

Let nonzero columns satisfy

\[
A_j = t A_k, \qquad t \ne 0,
\]

with bounds \(x_j\in[\ell_j,u_j]\) and
\(x_k\in[\ell_k,u_k]\). Keep column \(A_k\) and define

\[
y = x_k + t x_j.
\]

Then the constraint contribution is preserved exactly:

\[
A_jx_j + A_kx_k = A_k(t x_j+x_k)=A_ky.
\]

The mapped interval of \(t x_j\) is

\[
[a,b]=[\min(t\ell_j,t u_j),\max(t\ell_j,t u_j)],
\]

so the exact merged bounds are the Minkowski sum

\[
y\in[\ell_k+a,u_k+b].
\]

For the probed signs this specializes to:

- \(t=+1\): \([\ell_k+\ell_j,u_k+u_j]\);
- \(t=-1\): \([\ell_k-u_j,u_k-\ell_j]\).

The objective is

\[
c_jx_j+c_kx_k
=c_ky+(c_j-tc_k)x_j.
\]

Therefore the allocation-independent compatibility condition for a single
linear-cost merged variable is

\[
c_j=t c_k,
\]

and it is necessary whenever the feasible split can vary. Under this condition
the retained coefficient is \(c_k\). Fixed-allocation and bound-dominance
exceptions are characterized below. For a class, orient every column to one
canonical column \(B\): \(A_j=s_jB\), \(s_j\in\{-1,+1\}\).
All members are mutually compatible precisely when their oriented costs
\(s_jc_j\) agree. The merged variable is
\(y=\sum_j s_jx_j\), its bounds are the sum of the oriented member intervals,
and its cost is the common oriented cost.

### Exact postsolve split

For a pair, given a feasible reduced value \(y\), choose

\[
z=t x_j \in [a,b]\cap[y-u_k,y-\ell_k].
\]

The intersection is nonempty exactly because \(y\) lies in the merged
Minkowski-sum interval. A deterministic split for the probed one-sided bounds
is, for example,

\[
z=\max(a,y-u_k)
\]

(with the equivalent upper clamp applied defensively), followed by

\[
x_j=z/t,\qquad x_k=y-z.
\]

This respects both original boxes, reconstructs \(y\), preserves every row,
and preserves objective under the compatibility condition. In the fully
general extended-bound case, choose any finite point in the nonempty
intersection, explicitly handling an infinite endpoint rather than evaluating
an indeterminate infinity. A class split repeats this interval-intersection
operation member by member while passing the remaining aggregate to the final
member. The probe's 1,529 compatible members all have finite lower and infinite
upper bounds.

## Shape probe method

The input was obtained with the shipped public call
`presolve_matrix(..., algorithm="pdhg")`, leaving default-ON netagg enabled.
The resulting CSR matrix was transposed to CSC. Each nonempty column was keyed
by its exact row-index tuple and an exact coefficient tuple canonically oriented
so the first coefficient was positive. This detects only \(t=\pm1\); no
tolerance or coefficient rounding was used. Structural classes were then split
by exact oriented cost \(s_jc_j\). Only groups of size at least two were counted
as compatible merge classes.

Every one of the 84,923 post-netagg coefficients is exactly \(+1\) or \(-1\).
All parallel pairs found have \(t=+1\); the census found zero \(t=-1\) pairs.

## Class census

| Fact | Count |
| --- | ---: |
| Raw `pds_10` shape | 16,558 × 49,932 × 107,605 |
| Shipped netagg shape | 4,955 × 38,329 × 84,923 |
| Net aggregations recorded | 9,483 |
| Structural exact-parallel classes | 399 |
| Columns in structural classes | 4,417 |
| Structurally removable columns | 4,018 |
| Structural parallel pairs | 44,927 |
| Structural \(t=+1\) pairs | 44,927 |
| Structural \(t=-1\) pairs | 0 |
| Compatible-cost groups | 660 |
| Columns in compatible-cost groups | 1,529 |
| Compatible-cost removable columns | **869** |
| Compatible-cost pairs | 1,100 |
| Compatible class sizes | 473 × size 2; 165 × size 3; 22 × size 4 |

The structural class-size histogram was: size 2: 102 classes; 3: 33; 4: 66;
5: 11; 6: 22; 8: 11; 10: 11; 14: 11; 18: 11; 19: 11; 20: 22;
21: 11; 24: 22; 27: 11; 28: 11; 29: 11; 35: 11; 36: 11.

## Projected compatible-cost shape

| Metric | Netagg input | Projected | Reduction |
| --- | ---: | ---: | ---: |
| Rows | 4,955 | 4,955 | 0 (0%) |
| Columns | 38,329 | 37,460 | 869 (**2.267%**) |
| Nonzeros | 84,923 | 83,185 | 1,738 (**2.047%**) |
| Columns + nnz | 123,252 | 120,645 | 2,607 (**2.115%**) |
| Rows + columns + nnz | 128,207 | 125,600 | 2,607 (**2.033%**) |

The compatible class explains 869 of the 4,018 structurally removable
columns. It is far smaller than the 4,613-column behavioral rule-ablation
signal and fails the 5% live threshold on every relevant shape denominator.

## Incompatible-cost remainder

There are 43,827 structurally parallel but cost-incompatible pairs. At fixed
aggregate \(y\), their exact projected objective is

\[
c_ky + (c_j-tc_k)x_j,
\]

minimized over the bound-feasible split interval. This is generally a convex
piecewise-linear value function of \(y\), not one linear merged-column cost, so
blind merging changes the LP.

The textbook dominance exception follows by a feasible exchange/weak-duality
argument. In canonical contribution variables \(z_i=s_ix_i\), suppose one
member has smaller unit cost. Moving contribution from a more expensive member
to the cheaper member preserves \(Ax\) and weakly decreases the objective until
a bound blocks the exchange. A single-column reduction is exact when the same
endpoint blocks for the whole aggregate interval—for example, the retained
cheaper contribution is upper-unbounded and every more expensive contribution
has a finite lower bound, so all expensive members can be fixed at their lower
bounds. The sign-symmetric case retains a lower-unbounded expensive
contribution and fixes cheaper members at finite upper bounds. Without such a
global endpoint condition, allocation changes at breakpoints and requires the
piecewise value function or multiple columns.

For characterization only, 377 structural classes contain incompatible costs.
Of those, 300 satisfy the simple exact endpoint-dominance condition above
(all via an upper-unbounded lowest-cost contribution), accounting for 3,006
additional cost-group eliminations. The remaining 77 incompatible classes
(220 member columns) do not satisfy that sufficient condition. Per the unit's
explicit scope and kill rule, none of these incompatible-cost opportunities was
counted toward the compatible-class live gate or implemented.

## Commands and validation

Environment build:

```text
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation
Resolved 57 packages; built and installed linprogx successfully.
```

The census was run twice through `uv run python`, and a separate bound/sign
cross-check confirmed the pair counts and one-sided bounds. The shipped netagg
path's focused test module passed:

```text
tests/test_presolve_netagg.py .......................................... [ 62%]
.........................                                                [100%]
============================== 67 passed in 1.96s ==============================
```

Full pytest and the performance/certificate gates were not run: the sequence
requires an immediate stop at a sub-5% shape projection, and no solver or test
source was modified.

## Files touched

- `experiments/pds_parallel_cols_2026_07_18.md` (this report)

---

# Follow-up: endpoint-dominated parallel columns

## Verdict

**LIVE-SHIPPED-CANDIDATE, default OFF.** Compatible merging plus provably exact
endpoint dominance projects and realizes a **9.139% nnz reduction**, clearing
the 8% probe gate. The public `pds_10` alternating median improves **29.93%**
and iterations fall 7,104 -> 6,400. The implementation remains dark behind
`LINPROGX_PRESOLVE_PARALLEL_COLS=1` pending promotion.

## Exact dominance condition

Orient one exact structural class to a common nonzero column `B`:

\[
A_j=s_jB,\qquad z_j=s_jx_j,\qquad s_j\in\{-1,+1\}.
\]

The oriented variable has interval

\[
[L_j,U_j]=[\min(s_j\ell_j,s_ju_j),\max(s_j\ell_j,s_ju_j)]
\]

and unit cost \(\gamma_j=s_jc_j\). Constraints depend on the class only
through \(y=\sum_jz_j\). Equal-\(\gamma\) members first form an exact
compatible group, with interval endpoints summed as extended bounds.

For minimization, the implemented lower-endpoint dominance condition is:

1. all costs and every selected fixed endpoint are finite;
2. the minimum-cost compatible group has \(U=+\infty\);
3. every strictly higher-cost group has a finite lower endpoint.

The symmetric upper-endpoint condition is:

1. all costs and every selected fixed endpoint are finite;
2. the maximum-cost compatible group has \(L=-\infty\);
3. every strictly lower-cost group has a finite upper endpoint.

NaN/invalid boxes are rejected. A cheap upper-unbounded member paired with an
expensive lower-unbounded member (or its sign-symmetric counterpart) is not
reduced: that geometry can contain a cost-improving recession direction rather
than a finite selected endpoint.

### Safety argument

For the lower-endpoint case, take any feasible class allocation `z` and set
every higher-cost group `g` to its finite lower endpoint `L_g`. Add all released
amount to the retained minimum-cost group:

\[
z'_0=z_0+\sum_{g>0}(z_g-L_g),\qquad z'_g=L_g.
\]

The retained group can absorb the nonnegative amount because its upper endpoint
is infinite. The aggregate is unchanged, so `Ax=b` is unchanged, and every box
is respected. The objective change is

\[
\sum_{g>0}(\gamma_0-\gamma_g)(z_g-L_g)\le0.
\]

Thus every feasible solution has a no-more-expensive feasible representative
with all dominated groups fixed. Applying this map to an optimal solution
produces another optimal solution satisfying the fixes: if it were strictly
better, the original was not optimal. The upper-endpoint proof is identical
with inequalities reversed. This preserves feasibility and at least one
optimal solution; postsolve simply restores each fixed finite endpoint.

Compatible members of the retained group use the signed `_ParallelColumn`
record. If `A_j=t A_k`, postsolve intersects

\[
x_k\in[\ell_k,u_k]\cap[y-b,y-a],\quad
[a,b]=[\min(t\ell_j,tu_j),\max(t\ell_j,tu_j)],
\]

then sets \(x_j=(y-x_k)/t\). The 64-pattern battery covers coefficient signs,
`t=+/-1`, positive/negative compatible costs, and finite/one-sided boxes.

## General non-parallel dominated columns

A clean constructive sufficient condition exists, but independent row-activity
intervals do not establish it. To fix `x_j` at its finite lower bound, one
would need a jointly realizable exchange vector `v` over other columns such
that

\[
A_{-j}v=A_j,\qquad c_{-j}^Tv\le c_j,
\]

with `v_i>0` only where `u_i=+inf` and `v_i<0` only where `l_i=-inf`. Then
`x'_{-j}=x_{-j}+(x_j-l_j)v` is feasible and no more expensive. Row-activity
intervals can show that each row is separately compensable, but the row-wise
choices need not be one common `v`; treating them as joint would be unsound.
Finding `v` requires an auxiliary conic/LP feasibility problem per candidate,
not the requested clean activity-bound pass. No non-parallel columns were
therefore counted or reduced.

## Combined shape census

| Fact | Count |
| --- | ---: |
| Netagg input | 4,955 x 38,329 x 84,923 |
| Structural parallel classes / member columns | 399 / 4,417 |
| All structurally removable columns | 4,018 |
| Compatible-only groups / removable columns / nnz | 660 / 869 / 1,738 |
| Cost-incompatible structural classes | 377 |
| Endpoint-dominance classes | 300 (all lower-endpoint) |
| Members in dominance classes | 4,131 |
| Extra cost-group removals from dominance | 3,006 |
| Unresolved incompatible classes | 77 |
| **Combined removable columns** | **3,875 (10.110%)** |
| **Combined removable nnz** | **7,761 (9.139%)** |
| **Combined projected/realized shape** | **4,955 x 34,454 x 77,162** |

The native record stream contains 176 signed compatible merges and 3,699
endpoint fixes, totaling the same 3,875 removals. This is less than 869 merge
records because compatible members inside a dominated higher-cost group are
fixed directly rather than first merged and then fixed.

## Implementation and gates

The new native stage runs only when all of these global conditions hold:

- route is explicit PDHG or auto;
- netagg actually produced a composed reduction (therefore inheriting its raw
  size and 10% netagg-yield gates);
- post-netagg rows are at most 10,000;
- realized parallel/dominance reduction removes at least 8% of current nnz;
- `LINPROGX_PRESOLVE_PARALLEL_COLS=1` (default is `0`).

The 10,000-row maximum is a global reduced-network size gate. It makes
`pds_20` (10,495 post-netagg rows) and larger networks skip the native scan
entirely; there is no fixture-name dispatch.

| Gate | Raw result | Verdict |
| --- | --- | --- |
| Shape probe | nnz 84,923 -> 77,162 (-9.139%) | PASS >=8% |
| `pds_10` public wall, alternating 9 | 1.845878s off -> 1.293357s on (-29.93%) | PASS >=8% |
| `pds_10` iterations | 7,104 -> 6,400 | PASS |
| `pds_20` final alternating 9 | 12.812281s off -> 11.464505s on (-10.52%) | PASS <=+1% |
| `pds_20` trajectory | shape 10,495 x 84,872 x 187,223; 14,016 iters both | PASS, inert |
| qap12 sentinel | status/backend/3,392 iters/objective/full `x` identical | PASS, inert |
| qap15 sentinel | status/backend/3,456 iters/objective/full `x` identical | PASS, inert |
| Knob OFF | `pds_10` reduced components and all vectors exactly restored | PASS |
| Sign/cost algebra | 64 merge patterns + both dominance endpoints/signs | PASS |
| Cost-improving recession exclusion | native stage returns `None` | PASS |
| Full CI | lint/format/type/security + 518 passed, 7 skipped | PASS |

The first `pds_20` run, before adding the maximum-row gate, paid a rejected
full scan and measured +8.48% amid 6-16s host swings, so it failed and prompted
the gate. The final run was structurally inert. It was still extremely noisy
(individual identical-path solves ranged 7.36-45.76s), but its registered
median passes; shapes, objectives, solution trajectories, and iterations were
identical independently of timing.

### `pds_10` raw wall samples

```text
OFF: 1.448292 1.442855 1.459666 1.589032 1.889097 1.847727 1.845878 4.489006 2.750379
 ON: 1.244477 1.253030 1.290933 1.430985 1.293357 1.280173 5.743386 3.280386 1.711520
```

### Certificate and original-space residuals, knob ON

| Fixture | Objective-relative error | Equality residual | Bound residual | Status |
| --- | ---: | ---: | ---: | --- |
| pds_10 | 1.909e-9 | 4.690e-6 | 1.129e-6 | optimal |
| pds_20 | 1.685e-9 | 8.059e-6 | 4.333e-6 | optimal |
| qap12 | 5.928e-6 | 1.857e-5 | 0 | optimal |
| qap15 | 5.906e-6 | 6.960e-6 | 0 | optimal |

The qap objectives use the existing local Clarabel oracle results because the
local HiGHS qap15 record is a timeout. All errors and original-space residuals
are below `eps=2e-5`; `eps` was not changed.

## Follow-up information and files

Information used was restricted to repository source/tests, the prior census
in this report, the named behavioral entries already recorded in the repository,
the four local `/tmp/lpsuite` fixtures, existing local oracle artifacts, and
textbook LP algebra. No network access occurred; no external solver source was
read or fetched; no Git operation was run.

Files touched in the target worktree:

- `src/linprogx/_csparse.c`
- `src/linprogx/presolve.py`
- `tests/test_presolve_parallel_cols.py`
- `tests/test_presolve_fixpoint.py`
- `experiments/pds_parallel_cols_2026_07_18.md`

Final build/pytest tail:

```text
Required test coverage of 85% reached. Total coverage: 89.11%
======================= 518 passed, 7 skipped in 38.24s ========================
```
