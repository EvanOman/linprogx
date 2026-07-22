# Reopened frontier falsifier campaign (2026-07-22)

## Verdict

**NO FUNDED SUCCESSOR.** A final Overmind campaign reopened the strongest
algorithmic and exact-dataflow gaps left by the post-dossier census. It found
two real mechanisms—a zero-cost Phase-1 crash basis and a stable column-owner
dual-simplex pipeline—but neither clears its complete-cost and sentinel gates.
Independent role-swapping, decomposition, sifting, and full-KKT routes also
fail their opportunity screens.

No experimental C or Python solver change entered production. Protocol v3 was
therefore not rerun, and the certified board remains **23W-0P-1L**, with
greenbea at `1.2150867` and requiring a `17.7013%` whole-wall reduction.

## Independent algorithm frontier

The orthogonal solver census found no funded alternate route:

- Exact dualization reduces to the already-killed sparse revised-primal path.
  Its intentionally favorable floor is `0.537669s`, above the board target.
- Exact component/thin-border decomposition can detach at most `7.99%` of
  greenbea's columns while the border owns `54.81%` of its nonzeros.
- Omniscient active-bound sifting retains `95.53%` of the columns.
- The tested full-KKT block paths remain stopped by their endpoint and
  globalization evidence.
- Randomized sampled pricing is formally distinct, but it would need to retain
  at most `64.17%` of the attacked work with exact authority; no measured
  greenbea signal supports that cut.

These are funding screens, not impossibility claims for every future solver.

## Zero-cost Phase-1 crash: geometrically useful, economically dead

A degree-minimal rank-repaired crash basis selected zero-cost structural
columns before logical columns. Its exact zero-cost submatrix rank was 1,519.
The best repaired basis contained 1,457 structural and 68 logical columns and
reduced the homogeneous dual-sign score from `3110.463` to `394.073`
(`-87.3%`). It had about 6,600 basis nonzeros and only nine remaining sign
violations.

The endpoint is genuinely useful. Direct continuation from that basis was
deterministic and certificate-clean in three runs, took 4,073 pivots, and had
best solver wall `0.384285s`; maximum original equality and bound residuals
were `1.52e-8` and `1.15e-11`. But the dense Python rank repair took `2.463s`,
far beyond the complete wall budget. Homogeneous cleanup instead reached its
512-pivot cap and was killed.

The native rank-repair successor also failed:

- The raw 1,519-structural/6-logical seed performed 31 repairs, hit the current
  cap, fell back to the logical basis, and needed 7,603 pivots; best charged
  wall was `0.722815s`.
- Raising the repair cap to 128 avoided fallback after 48 repairs, but the
  resulting path took 9,381 pivots, returned deterministic
  `dual_infeasible`, and failed the certificate gate at `0.982--1.123s`.
- The environment-off control remained exact and authoritative.

The construction family is therefore killed in the tested dense-repair,
homogeneous-cleanup, and native rank-repair forms. No crash code was merged.

Retained artifacts:

- `/tmp/zero-cost-crash-s0.json`, SHA-256
  `264c6fee067f1f0f4bbd3707abeaebc5e6d66e34cb3bcc5f9f5730fef4c7836c`.
- `/tmp/zero-cost-raw-native-s0-BNgi6D/results.json`, SHA-256
  `29131753cc1398a3868e67f598bba6e0c779b66361991fac7f372848431269f0`.
- Isolated cap experiment: `/tmp/linprogx-native-repair-cap-s0-2BZ1fg`.

## Stable column-owner pipeline: exact, safe, insufficient

The exact-dataflow census identified four stable contiguous column owners.
Each owner computes the same per-column floating-point additions in original
CSR order, and a canonical merge preserves the existing Harris choice. An
environment-gated C prototype proved the mechanism:

- all 4,399 greenbea pivots matched exactly;
- status, objective, primal/dual vectors, basis, bounds, pivot sequence, and
  internal candidate/theta/entering/reduced-cost trace hashes matched;
- the final lifecycle-safe pool serialized dispatch, used targeted generations
  and condition variables, reset safely after `fork`, respected CPU affinity,
  and passed concurrent PDHG/dual-simplex and fork-child tests; and
- `just ci` in the isolated worktree passed 525 tests with 7 skipped and
  89.16% coverage.

This safety work was necessary. Review rejected the first pool because a
process-global dispatcher raced across overlapping solves and could hang in a
fork child. Review also exposed persistent-worker contamination, a historical
capacity/wake bug, affinity-blind thread selection, and the fact that each
owner rescanned and filtered every active CSR entry.

Fresh-process median-of-nine results for the repaired prototype were:

| measurement | baseline | candidate | ratio |
|---|---:|---:|---:|
| greenbea wall | 377.574ms | 350.676ms | 0.928762 |
| attacked slice | 171.385ms | 146.746ms | 0.856237 |
| cre_a sentinel | — | — | 1.0022 |
| woodw sentinel | — | — | **1.1220** |
| 80bau3b sentinel | — | — | 0.8919 |

The registered greenbea gate was `<=0.822987` and every sentinel gate was
`<=1.01`. The safe base therefore fails both the board and woodw gates and is
not fundable. Its terminal artifact is
`/tmp/column_owner_s0_terminal.json`, SHA-256
`7e27bceba84826f39f84fd32fcacd61e7573d2ad2bed65301c9e6008784fec9e`.

## Fused-dispatch successor: killed by measured dispatch cost

The proposed fused job would combine pivot-row scatter, Harris phase 1,
internal synchronization, and Harris phase 2, leaving reduced-cost work in a
second dispatch. Its paper projection required dispatch median at least
`3.331129us` to fund the missing wall cut.

The lifecycle-safe pool's fresh pure-dispatch median was only `0.9612us`.
Deleting all 8,798 removable dispatches saves `8.4566ms`, versus the
`39.9378ms` still required. The successor was killed before code. Evidence:
`/tmp/column_owner_fused_s0_kill.json`, SHA-256
`418c5a02e4556d1809f34047ba29020ee15d6e8c7a73655c80e733c566c96771`.

## Prepartitioned row-owner successor: killed by favorable bound

The last missing version of the mechanism removes four-way rescanning. A
setup-time stable partition can be represented exactly by four row-offset
arrays and either original-entry positions or copied owner-major column/value
subsequences. Stable traversal preserves each column's original addition
order.

For greenbea, the direct copied representation costs `328,120B` (`320.43KiB`):
`48,832B` of offsets, `93,096B` of columns, and `186,192B` of values. Building
it requires two passes, 46,548 entry classifications, 6,104 offset cells, and
23,274 copied pairs. The favorable projection charges **zero** incremental
setup time.

From 275 retained pivots, the current filtering model has 5,557,731 units of
barrier-critical work. Direct copied subsequences reduce that to 4,372,124,
or `0.786674` of current. The owner loads remain imbalanced at
`3,621,335 / 3,977,216 / 3,306,895 / 1,020,526`; the barrier-critical share is
`1.4664x` ideal balance.

Applying that favorable factor only to the measured `79.972564ms` pivot-row
phase saves `17.060205ms`:

| quantity | result |
|---|---:|
| current safe candidate | 350.676056ms |
| required candidate | 310.738262ms |
| favorable prepartition projection | 333.615851ms |
| projected whole-wall ratio | **0.883578** |
| remaining shortfall | 22.877588ms |

Even this favorable model is well above the `0.822987` gate. An impossible
bound that deletes all mandatory owned column/value loads only reaches
`0.817893`, with 1.923ms headroom. The sequential owner-boundary checks would
need to cost `10.745x` an owned/shared memory operation to hit the board; all
1,525 rows are sorted and have only 523 owner-boundary transitions, making
that assumption untenable. Woodw would independently require its pivot phase
to fall to `64.43%` of current, with no retained evidence for such a cut.

Result: `KILL_PRECODE` for both position-indirect and copied-direct stable row
prepartitioning.

## Reproduction boundary and production state

Experimental implementation lived only in
`/tmp/linprogx-column-owner-s0-zIr4EV`; the zero-cost experiments likewise
used isolated worktrees. Workers had no network access, no external solver
source was read, and no package action occurred. The authoritative production
tree remained at `d119ba4c7c1987137320d3fda7a53b00ab7bb132` throughout the
campaign.

The reopened frontier is empty under the current four-vCPU, fixed-`eps`,
global-policy, certificate-backed v3 protocol. Reopening now requires a new
mechanism with a measured opportunity beyond these exact owner and crash
families—not another spelling of replicated owner filtering, dispatch fusion,
stable row prepartitioning, or zero-cost rank repair.
