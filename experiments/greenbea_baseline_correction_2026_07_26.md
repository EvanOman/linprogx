# CORRECTION: greenbea's board ratio is ~1.527, and per-pivot cost is at parity

Two load-bearing numbers this campaign has been reasoning with are wrong. Both
were corrected by one direct paired measurement on clean hosts.

Artifact: `assets/modal_bench_af6bd89823fd_paired_hosts3.json`
(protocol v3 paired, Modal AWS `us-west-2`, 3 hosts x 7 interleaved pairs,
`loadavg 0.00` at start and end on all three, ref `af6bd89` = churn OFF).

## Correction 1 — the board ratio does not reproduce at 1.156

```
ratio_median_of_hosts = 1.5266   hosts [1.4897, 1.5428]
linprogx pairs won    = 0 / 21   verdict: highs_faster
```

The ledger's greenbea record ran **1.215 -> ~1.156**. A direct paired
re-measurement at the same code state gives **1.527**. The record does not
reproduce.

This is not a regression — it is the failure mode the campaign's own measurement
doctrine already documented: *"a v3 paired cert cannot resolve a sub-10% code
effect on the Modal host population... a paired run showed greenbea at 1.468
against a record of 1.215 purely from the host draw."* The absolute paired ratio
is **host-population dependent**; 1.215 and 1.156 were favourable draws.

**What this does NOT invalidate:** the `envab` results. An envab A/B runs both
arms on the *same* container, so host effects cancel by construction. The churn
certification (**0.9783**, 21/21 pairs, plus 0.9814 in a second run) is a valid
*relative* measurement. It is the **baseline it was applied to** that was stale.

**Corrected board position: greenbea ~1.527 -> ~1.494 with churn.**

## Correction 2 — linprogx is NOT 1.73x better per pivot; it is at parity

Both solvers on the same host, using deterministic load-invariant counts
(linprogx 4,399 pivots, HiGHS 2,836 simplex iterations):

| host | lx s | hx s | ratio | lx us/pivot | hx us/iter | lx per-pivot advantage |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.5516 | 0.3703 | 1.490 | 125.4 | 130.6 | **1.041x** |
| 1 | 0.6020 | 0.3902 | 1.543 | 136.9 | 137.6 | **1.005x** |
| 2 | 0.4088 | 0.2678 | 1.527 | 92.9 | 94.4 | **1.016x** |

The campaign's "**linprogx's per-pivot cost is 1.73x better** (85.7us vs
148.4us)" compared **linprogx measured on a quiet local box** against **HiGHS
measured in a different context**. On the same host, at the same moment, the two
are within **4%**.

## Why this matters more than either number

```
pivot-count ratio   4,399 / 2,836 = 1.551
measured wall ratio               = 1.527
```

**These agree.** greenbea's board loss is, to within measurement error,
*entirely* its pivot-count deficit. Per-pivot cost contributes essentially
nothing.

Consequences:

1. **The target is -35.5% pivots** (4,399 -> 2,836), not the -13.46% derived
   from the 1.156 baseline. Churn's -2.6% is real but is 7% of the distance.
2. The strategic claim that "closing the pivot gap is worth -35.5% against a
   13.46% bar -- a 2.6x margin" is **void**. There is no margin: closing the
   pivot gap is worth exactly the bar, because they are the same quantity.
3. Per-pivot work and pivot-count work pay at the **same rate** (both are ~99.5%
   of the cell). Neither is privileged; pick by expected size.
4. Any future greenbea claim must be an **envab** result against a
   **freshly-measured** paired baseline. Never compose a relative envab gain
   onto a paired ratio recorded from an older host draw -- that is precisely the
   error this document corrects.
