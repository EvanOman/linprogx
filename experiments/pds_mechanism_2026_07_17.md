# pds presolve mechanism and algorithm cross — 2026-07-17

## Ranked conclusion

1. **The remaining `pds_10` wall gap is primarily algorithmic: HiGHS dual
   simplex is the right algorithm class.** HiGHS solves raw `pds_10` with
   presolve **off** in 12,877 pivots and 0.360s. Its normal presolve reduces the
   pivot count to 7,508 but raises total wall to 1.139s. On linprogx's presolved
   input, HiGHS likewise takes 11,472 pivots/0.343s with presolve off versus
   7,272/1.217s with presolve on. The large structural reduction is a net wall
   regression for HiGHS, not the cause of its advantage over linprogx PDHG.
2. **The unidentified structural rule is HiGHS `Aggregator` (toggleable rule
   12, bit 4096).** It is real and large: the per-rule report attributes 10,167
   row and 10,167 column removals to Aggregator on raw `pds_10`. Disabling it
   leaves `13,161 x 42,574 x 90,937` instead of
   `4,092 x 32,646 x 78,216`. Applied to linprogx's presolved input, the
   HiGHS rule set with parallel detection disabled projects
   `14,438 x 47,812 x 103,230` to `4,186 x 36,943 x 87,172`; enabling the full
   rule set reaches `4,086 x 32,651 x 78,289`.
3. **`Parallel rows and columns` (rule 13, bit 8192) is the secondary
   structural mechanism.** It owns most of the post-aggregation column and nnz
   collapse: disabling it on raw `pds_10` leaves
   `4,200 x 36,737 x 86,618`. The per-rule report attributes 43 rows and 4,613
   columns to it. It is not the source of the 10k-row collapse.

Therefore the classification is **(c) both, ranked (b) then (a)**: linprogx
lacks HiGHS Aggregator's large-pds coverage/fixpoint and its parallel reduction,
but those rules do not explain the measured wall advantage. The performance
gap is HiGHS's dual-simplex implementation—start, pricing, ratio-test/bound-flip,
and basis-update economics—versus linprogx PDHG. Routing `pds_10` to linprogx's
current dual simplex is not viable: a direct run on the 14,438-row presolved
model hit its 100,000-pivot limit after 91.248s without solving.

All HiGHS evidence below comes only from highspy 1.14.0 runtime options and
logs. No solver source was inspected.

## 1. Per-rule attribution

Options: `presolve_rule_logging=True`, `log_dev_level=2`. Counts are the rule
table's attributed removals over the full presolve cascade.

| Rule reported by HiGHS | `pds_10` rows | `pds_10` cols | Calls | `pds_20` rows | `pds_20` cols | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empty row | 0 | 0 | 0 | 76 | 0 | 76 |
| Singleton row | 580 | 580 | 580 | 851 | 851 | 851 |
| Fixed column | 0 | 242 | 242 | 0 | 484 | 484 |
| Dominated col | 0 | 19 | 19 | 0 | 6 | 6 |
| Free col substitution | 125 | 125 | 125 | 257 | 257 | 257 |
| Doubleton equation | 1,540 | 1,540 | 1,540 | 2,745 | 2,745 | 2,745 |
| Dependent equations | 11 | 0 | 1 | 11 | 0 | 1 |
| **Aggregator** | **10,167** | **10,167** | **29** | **20,909** | **20,909** | **24** |
| Parallel rows and columns | 43 | 4,613 | 5 | 41 | 2,933 | 5 |
| **Total** | **12,466** | **17,286** |  | **24,890** | **28,185** |  |

Full raw shapes:

| Fixture | Raw | HiGHS presolved |
| --- | ---: | ---: |
| `pds_10` | 16,558 x 49,932 x 107,605 | 4,092 x 32,646 x 78,216 |
| `pds_20` | 33,874 x 108,175 x 232,647 | 8,984 x 79,990 x 188,070 |

The Aggregator attribution is not series-chain contraction. The prior exact
chain probe established that the linprogx-presolved pds matrices have zero
total-degree-2 junction rows.

## 2. Toggleable-rule ablation

Each row is a fresh HiGHS presolve of the raw fixture with exactly the named
rule disabled through `presolve_rule_off`. Shapes are `rows x cols x nnz`.

| Bit | Disabled rule | `pds_10` resulting shape | `pds_20` resulting shape |
| ---: | --- | ---: | ---: |
| none | Full presolve | 4,092 x 32,646 x 78,216 | 8,984 x 79,990 x 188,070 |
| 6 (64) | Forcing row | 4,092 x 32,646 x 78,216 | 8,984 x 79,990 x 188,070 |
| 7 (128) | Forcing col | 4,092 x 32,646 x 78,216 | 8,984 x 79,990 x 188,070 |
| 8 (256) | Free col substitution | 4,078 x 32,641 x 78,503 | 8,996 x 80,002 x 188,069 |
| 9 (512) | Doubleton equation | 4,118 x 32,675 x 78,271 | 9,035 x 80,041 x 188,045 |
| 10 (1,024) | Dependent equations | 4,108 x 32,673 x 78,604 | 8,997 x 79,998 x 188,411 |
| 11 (2,048) | Dependent free columns | 4,092 x 32,646 x 78,216 | 8,984 x 79,990 x 188,070 |
| **12 (4,096)** | **Aggregator** | **13,161 x 42,574 x 90,937** | **30,015 x 103,919 x 222,208** |
| **13 (8,192)** | **Parallel rows and columns** | **4,200 x 36,737 x 86,618** | **9,095 x 82,743 x 193,864** |
| 14 (16,384) | Sparsify | 4,092 x 32,646 x 78,216 | 8,984 x 79,990 x 188,070 |
| 15 (32,768) | Probing | 4,092 x 32,646 x 78,216 | 8,984 x 79,990 x 188,070 |
| 16 (65,536) | Enumeration | 4,092 x 32,646 x 78,216 | 8,984 x 79,990 x 188,070 |

The small counterintuitive changes when rules 8–10 are disabled are cascade
effects; none approaches Aggregator's ownership. Rule 12 is necessary for the
row collapse. Rule 13 is necessary for most of the remaining column collapse.

### Projection from linprogx's exact presolved input

This cross feeds HiGHS the matrices output by `presolve_matrix(...,
algorithm="pdhg")`.

| Fixture | linprogx presolved | HiGHS on that input, rule 12 off | HiGHS on that input, rule 13 off | HiGHS full on that input |
| --- | ---: | ---: | ---: | ---: |
| `pds_10` | 14,438 x 47,812 x 103,230 | 13,161 x 42,574 x 90,937 | 4,186 x 36,943 x 87,172 | 4,086 x 32,651 x 78,289 |
| `pds_20` | 30,202 x 104,579 x 225,131 | 30,015 x 103,919 x 222,208 | 9,110 x 82,931 x 194,092 | 8,996 x 79,990 x 187,910 |

For `pds_10`, Aggregator plus the minor rules, with parallel detection held
off, projects a 22.5% reduction in `m+n+nnz` and a 15.6% nnz reduction. The
full HiGHS rule set projects a 30.5% reduction in `m+n+nnz` and a 24.2% nnz
reduction. Iteration behavior under linprogx PDHG remains unmeasured; the prior
conditioning falsifier still applies.

The rule table on linprogx's `pds_10` input attributes 10,167 row/column
removals to Aggregator and 50 row plus 4,616 column removals to parallel rows
and columns. On the `pds_20` input it attributes 20,894 row/column removals to
Aggregator and 41 row plus 2,933 column removals to parallel detection.

## 3. Decisive algorithm/presolve cross

Every run forces HiGHS `solver="simplex"`, `simplex_strategy=1` (serial dual
simplex), and changes only `presolve="off"|"on"`. Wall is the complete
foreground `Highs.run()` call. All runs returned `Optimal` at the exact
published objective.

| Fixture | Input model | HiGHS presolve | DS pivots | Wall | Objective |
| --- | --- | --- | ---: | ---: | ---: |
| `pds_10` | raw, 16,558 x 49,932 x 107,605 | off | 12,877 | **0.360s** | 26,727,094,976 |
| `pds_10` | raw | on | 7,508 | 1.139s | 26,727,094,976 |
| `pds_10` | linprogx presolved, 14,438 x 47,812 x 103,230 | off | 11,472 | **0.343s** | 26,727,094,976 |
| `pds_10` | linprogx presolved | on | 7,272 | 1.217s | 26,727,094,976 |
| `pds_20` | raw, 33,874 x 108,175 x 232,647 | off | 32,612 | **1.817s** | 23,821,658,640 |
| `pds_20` | raw | on | 17,388 | 9.603s | 23,821,658,640 |
| `pds_20` | linprogx presolved, 30,202 x 104,579 x 225,131 | off | 28,353 | **1.456s** | 23,821,658,640 |
| `pds_20` | linprogx presolved | on | 17,206 | 10.483s | 23,821,658,640 |

Presolve substantially lowers HiGHS pivot counts but loses wall:

| Fixture/input | Pivot reduction from presolve | Presolve-on wall / off wall |
| --- | ---: | ---: |
| `pds_10` raw | 41.7% | **3.16x slower** |
| `pds_10` linprogx-presolved | 36.6% | **3.54x slower** |
| `pds_20` raw | 46.7% | **5.29x slower** |
| `pds_20` linprogx-presolved | 39.3% | **7.20x slower** |

`pds_20` exposes the cause most clearly: HiGHS's dependent-equations search
alone took about 7.7s in the logged presolve, while the raw, unpresolved LP
solved by dual simplex in 1.817s total.

Thus HiGHS's reduction does not matter positively for its pds solve. Its DS
kernel handles the larger unit-network LP so efficiently that the reduction is
a sideshow—and, at these sizes, an expensive one.

## 4. linprogx dual-simplex routing probe

The bounded direct run used linprogx's current presolved `pds_10` matrix and
existing native dual simplex with `max_iter=100_000`, `tol=2e-5`, and
`expand=1`.

| Shape | Status | Pivots | Wall | Bound flips | Artificial ejections | Degenerate pivots | Refactorizations |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14,438 x 47,812 x 103,230 | iteration limit | 100,000 | 91.248s | 3,169,649 | 13,437 | 0 | 526 |

The run did not solve. It is already about 41x slower than the approximately
2.2s PDHG route at the bound and about 266x slower than HiGHS DS on the same
linprogx-presolved matrix, while still unfinished. Immediate routing is
therefore rejected. The actionable route is not a dispatch change; it is the
existing DS pricing/locality and basis-update program needed to make linprogx's
dual simplex competitive on pds.

## Bottom line

- **Mechanism identified:** HiGHS `Aggregator` (rule 12) owns the 10k-row
  `pds_10` reduction; `Parallel rows and columns` (rule 13) owns most of the
  remaining column/nnz reduction.
- **Performance attribution:** algorithm first. HiGHS dual simplex with
  presolve off is faster than its deeply presolved solve on both pds fixtures.
- **Architecture implication:** an Aggregator-equivalent large-pds pass is a
  legitimate structural project if it can preserve PDHG iterations, but it is
  not the explanation for HiGHS's current wall advantage.
- **Routing implication:** dual simplex is the right algorithm class, but
  linprogx's current implementation cannot be routed there until its
  pds-specific pivot economics improve by orders of magnitude.

## 5. Final falsifier: Aggregator-only model under linprogx PDHG

### Verdict: LIVE

The Aggregator-only `pds_10` model makes linprogx PDHG **both smaller and
better conditioned**. Iterations drop loudly from **8,576 to 7,552** (-11.9%),
not toward the documented 10,688-iteration failure trajectory. Combined with
the nnz reduction, `iterations * nnz` falls **23.36%** and measured wall falls
**27.60%**. The objective and reduced-model equality/bound residuals pass the
`2e-5` gates.

This clears the stated >=15% gate and commissions extending linprogx's native
aggregation to pds scale. `pds_20` is a necessary caution rather than a kill:
its iterations rise 13.9% and erase the nnz work-proxy gain, but its measured
wall still falls 18.1% because the much smaller matrix is cheaper per
iteration. The native project needs separate `pds_10` and `pds_20` trajectory,
proxy, wall, and certificate gates; it must not assume aggregation is
iteration-neutral across the family.

### Export and eq-box conversion

The highspy 1.14.0 export used `presolve_rule_off=126912`: every toggleable bit
from 6 through 16 was disabled except rule 12, `Aggregator`. Untoggleable basic
rules remained active as required.

| Fixture | Raw model | Aggregator-only HiGHS export | HiGHS attributed removals |
| --- | ---: | ---: | --- |
| `pds_10` | 16,558 x 49,932 x 107,605 | 4,592 x 37,966 x 89,839 | Singleton row 580; Aggregator 11,386 |
| `pds_20` | 33,874 x 108,175 x 232,647 | 9,821 x 84,198 x 197,820 | Empty row 76; singleton row 851; Aggregator 23,126 |

Both exported models contain equalities only: 4,592 equality rows for
`pds_10`, 9,821 for `pds_20`, and zero ranged, one-sided, or free rows. Thus
the implemented ranged-row conversion added zero slack columns on these two
exports. The general conversion used by the probe is exact: for
`lower_i <= a_i x <= upper_i`, add a zero-cost column `s_i` and equality
`a_i x - s_i = 0` with `s_i in [lower_i, upper_i]`; infinite endpoints remain
infinite. Equality rows are retained directly without a fixed slack.

The solve used the exported column costs and bounds, plus HiGHS's exported
objective offset. Both PDHG legs used the existing
`SparseSolver(algorithm="pdhg", presolve=False, eps=2e-5,
max_iterations=50_000, check_interval=50_000)` so no linprogx re-presolve
perturbed either trajectory.

### Shape, iterations, wall, and work proxy

| Fixture | Model | Shape `m x n x nnz` | PDHG iterations | Wall | `iterations * nnz` | Proxy gain vs baseline | Wall gain vs baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pds_10` | linprogx baseline presolved | 14,438 x 47,812 x 103,230 | 8,576 | 1.984s | 885,300,480 | — | — |
| `pds_10` | Aggregator-only | 4,592 x 37,966 x 89,839 | **7,552** | **1.436s** | 678,464,128 | **23.36%** | **27.60%** |
| `pds_20` | linprogx baseline presolved | 30,202 x 104,579 x 225,131 | 21,696 | 11.299s | 4,884,442,176 | — | — |
| `pds_20` | Aggregator-only | 9,821 x 84,198 x 197,820 | **24,704** | **9.255s** | 4,886,945,280 | **-0.05%** | **18.09%** |

The `pds_10` iteration count itself improves by 1,024. This is not merely a
projection from fewer nonzeros. On `pds_20`, iterations increase by 3,008;
the 12.1% nnz reduction almost exactly cancels that regression in the simple
proxy, while reduced row/column dimensions and locality still produce the
measured wall win.

### Objective and reduced-model residual gates

The oracle is HiGHS dual simplex on the original raw model with presolve off.
For the Aggregator-only legs, objective is the linprogx solution's reduced
cost plus `HighsLp.offset_`, compared directly with HiGHS's original-model
optimum as requested.

| Fixture | Model | Status | Objective incl. offset | Oracle objective | Absolute delta | Relative delta | Equality residual | Bound residual |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pds_10` | baseline | optimal | 26,727,095,083.27369 | 26,727,094,976 | 107.27369 | 4.014e-9 | 1.323e-5 | 2.217e-12 |
| `pds_10` | Aggregator-only | optimal | 26,727,094,962.55593 | 26,727,094,976 | 13.44407 | 5.030e-10 | 2.501e-6 | 2.103e-12 |
| `pds_20` | baseline | optimal | 23,821,658,161.13005 | 23,821,658,640 | 478.86995 | 2.010e-8 | 1.806e-5 | 2.137e-11 |
| `pds_20` | Aggregator-only | optimal | 23,821,658,684.21210 | 23,821,658,640 | 44.21210 | 1.856e-9 | 1.782e-5 | 2.842e-14 |

All four runs certify under the scale-normalized objective gate and the
`2e-5` equality/bound gates. The Aggregator-only objectives are closer to the
oracle than the baseline PDHG objectives on both fixtures.

### Final pds path

`pds_10` does have a presolve path: **large-scale equality aggregation before
PDHG**. It clears the work-proxy, measured-wall, objective, and residual gates,
and its PDHG iterations improve rather than blow up. This is now the first
implementation priority for the remaining `pds_10` loss.

The dual-simplex route remains a separate, much larger program: linprogx's
current DS hit 100,000 pivots and 91.248s without solving, versus HiGHS at
11,472 pivots and 0.343s on linprogx's presolved input. A
bound-flipping-ratio-test/pricing/basis-update-class DS unit remains relevant
to the solver's longer-term algorithm portfolio, but it is no longer the only
open `pds_10` path. Raw PDHG kernel bandwidth remains near its documented
floor; aggregation wins by removing work and, on `pds_10`, improving the
trajectory.
