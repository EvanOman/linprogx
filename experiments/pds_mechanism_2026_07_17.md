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
