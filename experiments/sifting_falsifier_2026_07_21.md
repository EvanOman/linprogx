# Active-bound sifting falsifier: greenbea

Date: 2026-07-21  
Verdict: **KILL** — do not add a production restricted-master path.

This experiment tested whether the 39th-settled active-set observation could
be converted into a sound global active-bound sifting scheme. It could not.
The instrumentation and shadow kernels are environment-gated and never alter
the production pricing, ratio-test, reduced-cost update, basis, or certificate
path.

## Gates

The candidate had to satisfy every gate:

- no omitted column may beat or replace the production entrant;
- global scans may not occur more often than every 8 pivots;
- the median legally wakeable live set must be at most 60% of columns;
- fully charged projected whole-wall improvement must be at least 20%;
- if scan and maintenance overhead exceeds 5% of wall time, attacked work
  removal must be at least 51.4% (the removal needed to clear the 20% whole-wall
  gate when the attacked share is about 49.4%);
- status, objective, residuals, solution, basis, bound statuses, pivot count,
  and pivot hash must be unchanged.

The historical context also recorded a 35% probe gate and board ratio 1.2150867
(17.7013% whole-wall gap), but the more permissive 20% gate is decisive here.

## S0: read-only trajectory oracle

Fixture: `/tmp/lpsuite/lp_greenbea.mat`, presolved to 1,525 rows, 3,868
columns, and 23,274 nonzeros. The binary `LPXSIFT1` trace is streamed to disk
with bounded O(n) working memory. At each of 4,399 committed pivots it records
the pre-pivot row support, legal ratio candidates, sign/basis ambiguity set,
column-state transitions, entrant/leaver, and whether refactorization followed.

The trace was 71,240,064 bytes. With tracing disabled, no trace file or trace
buffers are created. Baseline and traced solves were identical:

- status `optimal`, 4,399 pivots, 33 refactorizations;
- original objective -72,555,248.1298459;
- maximum original equality residual 1.7688915798785843e-7;
- maximum original bound violation 3.857486786583777e-12;
- pivot hash 1054624160779546655;
- identical solution, basis, and bound-status SHA-256 hashes.

The focused identity test also runs an S2 shadow solve and requires exact
identity across the same outputs.

## S1: offline epoch/eviction replay

Policies were the cross product of epochs `{8,16,32,64}` and boxed-column
eviction ages `{1,2,4}`. All 3,611 one-sided or free columns are mandatory.
Basic and sign-ambiguous columns are mandatory at each pivot. Only one of the
257 finite boxed columns with a strict, safe reduced-cost sign can become
dormant. A dormant boxed column whose sign becomes ambiguous is immediately
woken and charged as a bound-flip/restoration repair. A dormant production
entrant is charged as an emergency full scan.

An omniscient lower bound, allowed to know every future entrant in an epoch,
still had median live sets of 3,695 columns (95.5274%) for epochs 8, 16, and 32,
and 3,696 (95.5533%) for epoch 64. This alone fails the 60% gate.

No legal replay was sound without emergency repair:

| Epoch | Age | Median live | Missed entrants | Sign-risk repairs | Emergency scan pivots | Weighted attacked work removed |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 96.2771% | 20 | 70 | 88 | 3.0592% |
| 8 | 2 | 96.4840% | 19 | 68 | 85 | 2.8212% |
| 8 | 4 | 96.7425% | 18 | 61 | 77 | 2.5214% |
| 16 | 1 | 96.2513% | 24 | 94 | 114 | 3.0824% |
| 16 | 2 | 96.5098% | 24 | 86 | 106 | 2.8029% |
| 16 | 4 | 96.7942% | 20 | 80 | 97 | 2.4566% |
| 32 | 1 | 96.3289% | 27 | 94 | 118 | 3.0139% |
| 32 | 2 | 96.6132% | 26 | 86 | 109 | 2.6759% |
| 32 | 4 | 97.0527% | 24 | 75 | 96 | 2.2411% |
| 64 | 1 | 96.3547% | 25 | 103 | 122 | 2.9531% |
| 64 | 2 | 96.8201% | 23 | 96 | 113 | 2.5707% |
| 64 | 4 | 97.2079% | 22 | 83 | 99 | 2.1107% |

The least-bad attacked-work estimate was epoch 16, age 1. Its 114 repair
pivots had a median gap of 24 pivots, but they are additional to the scheduled
scan every 16 pivots; therefore the effective restoration cadence sometimes
exceeds the allowed one scan per 8 pivots. More importantly, its 24 missed
production entrants directly falsify safe restriction.

The work estimate weights exact trace counts by the measured attacked buckets:

```text
removed_attacked =
    (0.248 * removed_pivot_row
   + 0.149 * removed_ratio
   + 0.097 * removed_rcost) / 0.494
```

For the selected policy the individual removals were only 0.9895%, 6.0210%,
and 3.9197%, respectively.

## S2: duplicate-kernel and restoration timing

Five shadow runs timed epoch 16, age 1. Every run retained the baseline status,
objective, residuals, 4,399-pivot trajectory, and output hashes. The shadow
path performs duplicate work only: exact full reduced-cost scans use
`y = B^-T c_B` and `r_j = c_j - a_j^T y`; full and active CSC pricing,
ratio, and reduced-cost kernels alternate execution order to reduce cache-order
bias. The maximum exact-scan discrepancy from maintained reduced costs was
6.4462714561841494e-6, below the fixed 2e-5 experiment epsilon.

Median raw totals across the five runs (microseconds):

| Component | Full | Active |
|---|---:|---:|
| CSC pricing | 169,135.84 | 163,481.78 |
| ratio candidates | 87,742.33 | 84,911.03 |
| reduced-cost update | 48,096.73 | 53,564.93 |

The active kernel processed 9,916,457 of 10,551,803 eligible column-visits
(94.0%) and 70,032,575 of 71,218,728 nonzero-visits (98.3%). Median exact-scan
time was 33,331.78 us and median mask maintenance was 117,492.56 us. There were
275 scheduled scans, 4,399 valid live samples, mean live fraction 96.2660%, and
24 dormant production entrants in every run.

The fully charged projection uses the uninstrumented baseline phase buckets:

```text
projected_attacked =
    baseline_pivot_row * (active_price / full_price)
  + baseline_ratio    * (active_ratio / full_ratio)
  + baseline_rcost    * (active_rcost / full_rcost)
  + exact_scan_time
  + maintenance_time

whole_wall_improvement =
    (baseline_attacked - projected_attacked) / baseline_wall
```

At the median it projects 422,955.83 us versus 273,941.65 us of baseline
attacked work: **-54.3963% attacked-work removal** and **-23.8931% whole-wall
improvement** (a slowdown). Scan plus maintenance alone consumes 24.5952% of
the 0.623669 s baseline wall time, far above the 5% overhead gate, while the
active layout removes only 1.6656% of nonzero-visits rather than the required
51.4% attacked work.

## Certificate and bound safety

This result does not claim that cached reduced costs certify a restricted
master. A production design would need an exact full restoration scan before
an optimal/infeasible conclusion. Omitted finite boxed columns whose exact
reduced-cost sign changes must be woken and charged for the corresponding
bound flip. One-sided and free columns cannot be made dormant; columns carrying
the solver's artificial-bound fallback are likewise mandatory. The replay and
shadow policy enforce those rules. Even with them, the scheme misses entrants,
retains nearly every column, and has decisively negative charged economics.

## Decision

Stop at S2. Do not implement production active-bound sifting for greenbea.
The idea fails every economic gate and the primary soundness gate. The useful
artifact is the env-gated trace/shadow falsifier, not a restricted solver path.
Machine-readable evidence is in `/tmp/sifting-falsifier/results.json`; the
earlier malformed run was preserved as
`/tmp/sifting-falsifier/results_pre_buffer_fix.json` for audit only.
