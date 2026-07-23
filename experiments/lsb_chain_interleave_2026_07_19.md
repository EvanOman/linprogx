# LS-B — dependency-chain interleaving without reordering (2026-07-19)

## Verdict: KILLED

LS-B found abundant *logical* independence but no wall-time win.  Across the
dominant FTRAN/BTRAN scatter streams, the same-target collision rate was
**0.000% at W=2, W=4, and W=8** on both the cold and retained B* trajectories.
Nevertheless, both allowed prototypes missed the performance gates.  The
stronger full-list gather/compute/ordered-commit prototype regressed the paired
median BTRAN+FTRAN slice by **5.51% cold** and **4.16% from B***, and regressed
end-to-end wall by **2.39% cold** and **1.38% from B***.

The arithmetic result is stronger than the fallback gate: knob-on and knob-off
runs have identical pivots, reduced objectives, complete reduced-x bytes, final
basis bytes, and bound-status bytes on both trajectories.  Knob-off hashes also
match the retained pre-change K4 scalar artifacts.  Thus the achieved identity
level is **byte-identical**, not merely trajectory-identical.

The binding kill rule was: stop if W=4 collisions exceed about 30%, or if two
prototype attempts miss the S3 gates.  S1 passed overwhelmingly, but both
prototypes missed; no third implementation was attempted.

## Scope and method

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`, presolved to 1,525 rows x 3,868
  columns x 23,274 nonzeros.
- Solver: dual simplex, Dantzig leaving, EXPAND enabled, `bfrt=0`, tolerance
  `1e-8`; certificate epsilon fixed at `2e-5`.
- Trajectories: native cold crash and the retained local P3 constructive B*
  basis from `/tmp/phase1-predictions/results.json`.  No auxiliary or external
  solver ran.
- Structural constraint: the existing FT logical traversal order was not
  changed and no level construction or preprocessing was added.
- Build: every C edit was followed by the dossier's editable reinstall with
  `UV_OFFLINE=1`, making network access mechanically unavailable.
- Timing: nine repetitions of all four trajectory/arm combinations, start order
  rotated by repetition, foreground, pinned to idle CPU 2.  The primary effect
  is the median of the nine within-repetition ratios.
- Attribution: existing `LINPROGX_DS_SOLVE_SLICE=1` nested FTRAN/BTRAN timers.

An initial median-of-nine run pinned to CPU 4 was discarded before analysis
because an unrelated integration benchmark was found saturating that core.
No process was stopped; the unchanged schedule was rerun on idle CPU 2.  Only
the uncontaminated rerun appears below.

## S1 — dependency micro-census

`LINPROGX_DS_DEP_CENSUS=1` records each actual scatter target emitted by the
dense FT solve bodies and their sparse startup counterparts.  History resets at
each inner sparse update list.  An update is a collision at W if its target
appeared among the preceding W-1 emitted targets in that same list.  This is the
local dependence that bounds K-way interleaving; unrelated list boundaries are
not counted as opportunities.

| trajectory | direction | emitted updates | W=2 collisions | W=4 collisions | W=8 collisions |
|---|---|---:|---:|---:|---:|
| cold | FTRAN | 11,907,180 | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| cold | BTRAN | 5,597,514 | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| B* | FTRAN | 9,127,427 | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |
| B* | BTRAN | 6,378,686 | 0 (0.000%) | 0 (0.000%) | 0 (0.000%) |

Totals are 17,504,694 cold scatter updates and 15,506,113 B* scatter updates.
The W=4 collision rate is zero, far below the approximately 30% early-kill
line, so LS-B proceeded.

The zero is consistent with the representation invariants: factor columns,
factor rows, individual spike columns, and individual eta supports contain
unique indices.  It does **not** say the entire triangular solve is independent:
outer column/row traversal still carries triangular dependencies, while the
eta and L-transpose dot products still carry one accumulator dependency.

## S2 — two prototypes

Both mechanisms are global and enabled only by
`LINPROGX_DS_PIPESOLVE=1`.  The unset path retains the historical scalar loops.

### Attempt 1: K=2/K=4 block interleaving

The first version loaded two or four targets and products into independent
locals, then committed the updates in original order.  Runtime duplicate-target
checks fell back to the exact scalar sequence.  A rotated three-repetition
screen showed no promising arm:

| trajectory | arm | median solve | change vs off | median wall | change vs off |
|---|---|---:|---:|---:|---:|
| cold | off | 198.022 ms | — | 555.342 ms | — |
| cold | K=2 | 198.338 ms | 0.16% slower | 556.898 ms | 0.28% slower |
| cold | K=4 | 199.438 ms | 0.72% slower | 559.693 ms | 0.78% slower |
| B* | off | 178.410 ms | — | 521.880 ms | — |
| B* | K=2 | 179.856 ms | 0.81% slower | 522.104 ms | 0.04% slower |
| B* | K=4 | 181.052 ms | 1.48% slower | 527.738 ms | 1.12% slower |

This counted as prototype miss 1.  The screen was sufficient to reject both K
widths but was not used for the final gate.

### Attempt 2: full-list gather/compute, then ordered scatter

The stronger version stages every result from one structurally unique sparse
list into existing LU scratch (`ws_w`, with `gp_stack` for filtered target
indices), then commits the list in its original traversal order.  It covers:

- FTRAN's L scatter;
- FTRAN's live static-U or spike-column scatter;
- BTRAN's live static-U row and spike-row scatter; and
- BTRAN's transposed eta scatter.

Each list commits before the next list begins.  Products and subtractions remain
one operation each, with no reassociation.  There is no allocation, persistent
schedule, level construction, or amortization bet.  This maximally separates
the gathered x loads from x stores, but it pays an extra write/read through the
scratch vector.

## S3 — alternating median-of-nine

The table reports medians of the nine runs; gate decisions use the paired
median reduction shown in the last two columns.

| trajectory | arm | FTRAN | BTRAN | BTRAN+FTRAN | wall | paired solve reduction | paired wall reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| cold | off | 137.928 ms | 68.406 ms | 206.359 ms | 574.528 ms | — | — |
| cold | pipe | 141.970 ms | 73.552 ms | 215.961 ms | 585.998 ms | **-5.51%** | **-2.39%** |
| B* | off | 119.519 ms | 68.015 ms | 186.944 ms | 549.374 ms | — | — |
| B* | pipe | 121.056 ms | 74.019 ms | 195.075 ms | 558.406 ms | **-4.16%** | **-1.38%** |

Negative reduction means regression.  BTRAN is hurt most from scratch staging:
the ratio of medians is +7.52% cold and +8.83% from B*.  FTRAN regresses +2.93%
cold and +1.29% from B*.

### Gate ledger

| gate | required | measured | result |
|---|---:|---:|---|
| BTRAN+FTRAN slice, cold | >=20% faster | 5.51% slower | FAIL |
| BTRAN+FTRAN slice, B* | >=20% faster | 4.16% slower | FAIL |
| end-to-end, cold | >=8% faster | 2.39% slower | FAIL |
| fixture regression | no fixture >1% | greenbea cold +2.39%; B* +1.38% | FAIL |
| identity | byte-identical target | byte-identical both trajectories | PASS |
| certificate | optimal, residuals <=2e-5 | all 36 timed runs pass | PASS |
| pytest | green | 522 passed, 7 skipped | PASS |

The broader fixture battery was not expanded after greenbea itself failed the
1% regression gate and the second-prototype kill condition fired.

## Identity and correctness

| trajectory | pivots | reduced objective | original objective | equality residual | bound violation |
|---|---:|---:|---:|---:|---:|
| cold | 4,399 | -72,557,668.26492292 | -72,555,248.12984590 | 1.77e-7 | 3.86e-12 |
| B* | 3,334 | -72,557,668.26492676 | -72,555,248.12984978 | 4.77e-7 | 2.88e-12 |

For every run in each trajectory, pipe/off hashes form the same singleton for:

- the complete reduced-x double buffer;
- final basis int32 buffer; and
- final bound-status int8 buffer.

Those three knob-off hashes also equal the pre-change scalar hashes in
`/tmp/k4_ratio_probe_2026_07_19.json`.  The active path therefore attains exact
byte identity as well as trajectory identity.

## Why independence did not pay

S1 falsifies “not enough independent targets,” but independence is only a
necessary condition.  The scalar baseline's distinct addresses already permit
the Zen 2 out-of-order core to overlap multiple cache-resident misses after
address generation.  Attempt 1 adds block-control and duplicate checks without
creating a new machine-level capability.  Attempt 2 more forcefully removes
store/load proximity, but adds a scratch write and read per update plus target
index staging on filtered lists.  That traffic and the second commit loop cost
more than the avoided forwarding hazard.  Neither attack helps the remaining
serial dot accumulators or the outer triangular dependency.

Thus K1's low IPC is real, but the reported store-to-load-looking sequence is
not a 20%-removable software serialization at this granularity.  The processor
was already extracting the useful target-level parallelism.

## S4 — flip arithmetic stacked with K4

The K4 read-only report measures a **7.21% cold end-to-end reduction**.  Applying
that and LS-B's paired **2.394% regression** multiplicatively to the dossier's
90.5 us/pivot cold reference gives:

```text
K4 + LS-B = 90.5 * (1 - 0.0721) * (1 + 0.023944)
          = 85.99 us/pivot
target    = 54.00 us/pivot
gap       = 31.99 us/pivot
```

After K4, LS-B would have needed another **35.70% reduction** to reach 54
us/pivot.  It instead regresses 2.39%.  From the measured stacked point, a
further **37.20% reduction** is still required.  LS-B therefore does not improve
K4's flip arithmetic and cannot win the design competition on measured wall.

## Validation and artifacts

- Full tests: **522 passed, 7 skipped** in 65.58 s.
- Coverage gate: **89.16%**, above the 85% floor; 522 passed, 7 skipped.
- Ruff lint and format check: passed.
- `ty check`: passed.
- Bandit medium-and-higher scan: passed.
- `pip-audit`: not run because its advisory lookup is network-capable and the
  dossier forbids all network access.

Artifacts:

- Driver: `experiments/lsb_chain_interleave_probe.py`
- Raw median-of-nine results: `/tmp/lsb_chain_interleave_2026_07_19.json`
- Census and gated implementation: `src/linprogx/_csparse.c`
- Feature gate: `LINPROGX_DS_PIPESOLVE=1`
- Census gate: `LINPROGX_DS_DEP_CENSUS=1`

No network access, competing-solver source inspection, per-problem tuning, Git
operation, structural reordering, or background benchmark was used.

**Final verdict: KILLED.**
