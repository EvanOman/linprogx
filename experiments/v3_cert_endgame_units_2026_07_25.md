# v3 certification — endgame units (2026-07-25)

**Certified on-host effect: −4.89%** (v3 median-of-hosts, Modal AWS us-west-2,
3 hosts × 7 interleaved pairs). Bit-identical. Projected greenbea cell
**1.215 → 1.1556**. **Board remains 23W-0P-1L** — the cell improves but does not
flip (17.7013% required, 4.89% delivered).

Source snapshot: `d50b01adb29419d8834c5b5c849d892930524c12`.

## Units certified

1. Harris cheapest-filter-first early-outs (`ds_harris_pass1_avx2`)
2. Narrow CSR index cache (pivot-row scatter)

Both bit-identical: 4,399 pivots, objective `-72555248.12984592`, residual
1.769e-07, unchanged on every run.

## Scope: exactly one board cell

`experiments/route_census.py` resolved the public backend of all 24 cells.
**Only greenbea routes to the dual simplex** (`native-c-sparse-dual-simplex`);
every other cell is `native-c-sparse-ipm` or `native-c-sparse-pdhg`. The units
live inside `CSRMatrix_solve_eq_box_dual_simplex`, so the other 23 cells cannot
be affected — they never execute the changed code. **woodw is IPM**, which also
explains the previously undiagnosed four-column-owner woodw regression as
IPM/`LINPROGX_CHOL_SCHED` thread-pool contention rather than DS arithmetic.

## Why the paired cert could not certify this, and the envab could

A v3 **paired** run (linprogx vs HiGHS) was run first and is **uninterpretable
for a 4.9% effect**:

| cell | host 0 | host 1 | host 2 | median | record |
|---|---:|---:|---:|---:|---:|
| greenbea (touched) | 1.4698 | 1.4677 | 1.1619 | 1.4677 | 1.215 |
| **woodw (untouched control)** | 0.9696 | 0.9163 | 0.7865 | 0.9163 | 0.789 |

woodw's code did not change, yet its cell moved **+16%**. HiGHS's own greenbea
wall varied **0.2738 → 0.4221 s (54%)** across hosts. Since HiGHS is fixed, that
spread is pure host heterogeneity, and it shows the *ratio itself* is
host-dependent — consistent with the ledger's note that "the SIMD gains amplify
on bandwidth-tight hosts". **The board was NOT updated to 1.468**: attributing a
host draw to code when the untouched control moved with it would be exactly the
error the campaign's own doctrine forbids.

The correct instrument is **envab**: an on-host A/B of the knob, which cancels
host effects by construction because both arms run on the same container.

## envab result (units OFF vs units ON)

`--env-a "LINPROGX_DS_HARRIS_FASTPATH=0,LINPROGX_DS_IDX32=0"`, `--env-b ""`.

| host | units OFF | units ON | ratio B/A | pairs won |
|---|---:|---:|---:|---:|
| 0 | 0.33803 s | 0.32149 s | 0.9511 | 6/7 |
| 1 | 0.33811 s | 0.32158 s | 0.9511 | 7/7 |
| 2 | 0.57266 s | 0.53679 s | 0.9374 | 6/7 |

**v3 median-of-hosts: 0.9511 ⇒ −4.89%.** 19/21 pairs won. Host spread 1.4%.
Verdict `lxB_faster` on every host.

## The transfer factor: hypothesis FALSIFIED

The adversary lane argued that every local-vs-gate comparison in this campaign
assumes a local→on-host transfer of 1.0, and that the in-class precedent is the
shipped SIMD unit at **2.4873** (−11.30% local → 1.690/1.215 = −28.107% on-host).
Under that factor the rejected four-column-owner prototype would land at
greenbea 0.99979 — parity — which would have been a major reopening.

**Measured here: local −4.55% → on-host −4.89%, transfer factor 1.075.**

The 2.4873 precedent does **not** generalize to this mechanism class. The
finding was a good hypothesis, correctly identified as decisive, and the
measurement settled it against itself. Consequences:

- Local A/B numbers on this class can be read at face value (transfer ≈ 1).
- The four-column-owner resurrection is **not** supported: at transfer ≈ 1 it
  projects to ~1.128, still a loss, and its woodw blocker was in any case a
  different route.
- Lane A's *other* findings (woodw ratio-vs-cell category error, the 17.31%
  "ceiling" not being a ceiling on 4 vCPUs, the never-assembled owner stack)
  are untouched by this and remain open.

## Board

Board of record **23W-0P-1L, unchanged**. greenbea cell projected
`1.215 × 0.9511 = 1.1556`. Stating the cell as *measured* rather than projected
requires a fresh paired cert on a clean host draw; this run''s draw had a
control moving 16% and is not usable for that purpose.

Artifacts: `experiments/modal_bench_d50b01adb294_envab_hosts3.json`,
`experiments/modal_bench_d50b01adb294_paired_hosts3.json`,
`experiments/route_census.json`.
