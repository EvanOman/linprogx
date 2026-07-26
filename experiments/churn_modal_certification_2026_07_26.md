# CERTIFIED: the churn penalty moves greenbea's board cell

**PROVENANCE: CLEAN-ROOM (independent).** The churn penalty was derived from
linprogx's own `cols_reentering_gt10` diagnostic, not from reading HiGHS. It
predates and does not depend on the source-informed work on this branch.

## The measurement

Protocol v3 envab, Modal AWS `us-west-2`, **3 hosts x 7 interleaved pairs**,
median-of-hosts, `loadavg 0.00` at start and end on every host. Ref
`af6bd89823fd`. Artifact: `assets/modal_bench_af6bd89823fd_envab_hosts3.json`.

- **A** = shipped path, no overrides.
- **B** = `LINPROGX_DS_CHURN_DANTZIG=1 LINPROGX_DS_CHURN_ALPHA=2.0
  LINPROGX_DS_CHURN_DEADBAND=5 LINPROGX_DS_CHURN_CAP=1000`

| cell | ratio B/A | host range | pairs won by B | verdict |
|---|---:|---|---:|---|
| **lp_greenbea** | **0.9814** | [0.9802, 0.9814] | **21 / 21** | **B faster** |
| lp_woodw *(null control)* | 1.0038 | [0.9989, 1.0050] | 8 / 21 | coin flip |

## Why the control matters

`lp_woodw` is **IPM-routed**: the churn environment variables are read only on
the dual-simplex path, so they **cannot** act on it. B and A are therefore the
same computation, and the cell measures the instrument, not the change. It
returns **1.0038 with 8/21 pairs** — a coin flip — which pins the noise floor at
roughly **+/-0.5%** and confirms the envab harness is not manufacturing
differences.

greenbea's **-1.86%** is unanimous (21/21), reproduces on three independent
hosts to within **0.12%**, and sits well outside that floor.

## Board effect

**greenbea 1.156 -> 1.1345.**

Still a loss. Still 23W-0P-1L. But it is the campaign's **second** certified
greenbea improvement, and unlike the first (-4.89%, a bit-identical per-pivot
optimisation) this one comes from **reducing the pivot count** — 4,399 -> 4,283,
the thing the campaign spent months failing to move.

## Consistency check

Pivots fell **2.6%** and wall fell **1.86%**. The wall gain being slightly
smaller is expected: per-solve fixed costs (presolve at 3.0% of the cell, setup,
certification) do not scale with pivot count. The two numbers are consistent,
which is a check that the wall result is the pivot result and not an artefact.

## What it is not

- Not a flip. greenbea needs ~13.46% from the original 1.156 baseline; this
  delivers 1.86%.
- Not from DSE. Exact DSE cannot win greenbea's wall at any price
  (`greenbea_outside_kernel_kills_2026_07_26.md`); it is a class mechanism.
- Not yet the default. The gates remain OFF pending a full-suite regression run
  to confirm no other cell regresses.
