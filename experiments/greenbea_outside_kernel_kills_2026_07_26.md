# Two greenbea hypotheses outside the pivot loop, both killed in one session

Both were cheap to test, both were plausible, both are now closed with numbers.
Recording them so nobody funds them again.

## KILL 1 — thread oversubscription on constrained hosts

**Hypothesis.** The ledger records a paradox: on this 12-core box linprogx
solves greenbea in the same league as HiGHS, yet the board (Modal AWS us-west-2,
**4-vCPU** containers) records a 1.156x loss, with the ratio varying **1.16-1.47
across three hosts of the same class** and HiGHS's own wall varying **54%**.
That is a contention signature, not an algorithmic one.

The mechanism looked real in the source: linprogx sizes its thread pool from
`sysconf(_SC_NPROCESSORS_ONLN)` (`_csparse.c:7274`, `:1694`), which reports the
**host's** online CPUs, **not the cgroup CPU quota** a container is throttled
to. On a 4-vCPU container on a large host that would oversubscribe badly, and
HiGHS would not, being single-threaded here.

**Measured (7 interleaved reps, greenbea, public route):**

```
linprogx CPU/wall = 0.74
HiGHS    CPU/wall = 0.76
```

**FALSIFIED.** A threaded solver shows CPU/wall > 1. Both are ~0.75, i.e. both
are single-threaded and the box is loaded. **greenbea's route never enters the
thread pool at all** — those `pool_ensure` sites are on the PDHG and IPM paths,
and greenbea routes to the dual simplex. There is no oversubscription to fix.

The host-dependence of the board ratio remains unexplained, but it is **not**
threading.

## KILL 2 — route, presolve and Python-glue overhead

**Hypothesis.** Every attack of this campaign has targeted the pivot loop, but
the board measures the **public route**. If a meaningful fraction of the cell
were spent in presolve, certification, or Python marshalling, that would be
completely unattacked ground — and greenbea needs only 13.46%.

**Measured (median of 5, CPU ms, load-invariant):**

| stage | CPU ms | share |
|---|---:|---:|
| presolve | 15.0 | **3.0%** |
| dual simplex (4,399 pivots) | 499.1 | **99.5%** |
| everything else (route / cert / glue) | -12.5 | ~0% (noise) |
| **public route total** | **501.6** | 100% |
| *(problem construction, outside the route)* | *5.3* | — |

**FALSIFIED.** **99.5% of the board cell is the pivot loop.** There is no
overhead to reclaim. The negative "everything else" is timing noise between
separately-measured stages, which independently bounds any glue cost at
essentially zero.

## What this leaves

greenbea's cell is the pivot loop and nothing else. Every remaining path is
therefore **fewer pivots** or **cheaper pivots**:

- **Fewer pivots.** Dantzig + churn is the only mechanism that has ever reduced
  greenbea's count: **4,399 -> 4,283 (-2.6%)**, certified. Against a 13.46%
  bar that is real but not sufficient.
- **Cheaper pivots.** This is where the campaign's one certified ship came from
  (-4.89%, Harris dead-division early-outs + narrow CSR index cache). The
  per-pivot bucket census leaves **55.59 us/pivot (39%) unattributed** in
  "everything else" on the Dantzig path — the largest single unexamined block
  in greenbea's cell.
- **NOT exact DSE.** Arithmetic kill: Dantzig's greenbea cell is
  `4,399 x 143.8 us = 633 ms`, so DSE+churn's 4,342 pivots would need
  `<= 145.7 us/pivot`. DSE's buckets **excluding `pricing_update` entirely**
  already sum to **204.75 us**. Even with that kernel driven to zero, DSE
  loses greenbea on wall. **DSE is a board-v2 class mechanism, not a greenbea
  mechanism** — a distinction that must not get blurred when the class result
  (1.843x -> 1.115x) is reported.
