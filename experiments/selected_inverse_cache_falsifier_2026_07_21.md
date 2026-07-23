# Exact selected-inverse row-cache falsifier — 2026-07-21

## Verdict: KILLED

Caching selected rows of \(B^{-1}\) cannot meet the campaign's 20% charged
whole-wall probe gate on greenbea. The conclusion does not depend on an
implementation-quality estimate:

- the current median BTRAN phase is only **99.6 ms of 588.6 ms**, or **16.92%**
  of wall;
- therefore deleting every BTRAN at zero cost would miss both the **20% probe
  gate** and the board's **17.7013% required improvement**;
- the leaving-position trace has 1,068 compulsory first touches, so even a
  clairvoyant unlimited cache can save at most 3,331 of 4,399 pivot BTRANs;
  this zero-maintenance ceiling is **75.42 ms, or 12.81% of wall**;
- an optimistic charged model including row maintenance and refactor rebuilds
  peaks at only **1.91% of wall**.

There is also an independent numerical-authority failure. The existing
one-step rank-one lookahead probe uses the same exact-real inverse-update
identity and changes 15 Harris entering choices in floating point. A cached
row cannot be consumed as authoritative without an exact correction or fresh
BTRAN, which spends the work the cache was intended to save.

No solver implementation is warranted.

## Exact update and Forrest–Tomlin compatibility

Let basis column \(r\) be replaced by entering column \(a_q\), and let

\[
d = B^{-1}a_q.
\]

The updated basis is

\[
B' = B E,
\qquad
E = I + (d-e_r)e_r^T.
\]

Because \(e_r^T(d-e_r)=d_r-1\), Sherman–Morrison gives

\[
E^{-1}
= I - \frac{(d-e_r)e_r^T}{d_r},
\qquad
B'^{-1}=E^{-1}B^{-1}.
\]

For inverse row \(w_i^T=e_i^TB^{-1}\), the exact-real update is therefore

\[
\boxed{
w_i'^T
=w_i^T-\frac{d_i-\delta_{ir}}{d_r}w_r^T
}.
\]

In particular,

\[
w_r'^T=w_r^T/d_r.
\]

This identity is compatible with Forrest–Tomlin because it depends only on
the basis update, not on the factor representation. The current pivot already
has \(d\) from FTRAN and \(w_r\), also called \(\rho\), from BTRAN. On a cache
miss the fresh \(w_r\) can be inserted after scaling; on a hit the cached row
can supply \(\rho\).

The notation hides the maintenance cost. For \(K\) cached rows, a dense
rank-one update performs approximately \(2Km\) floating-point operations per
pivot and streams at least \(16Km\) bytes just to read and write the row
matrix. Here \(m=1,525\). The old \(w_r\) must remain unchanged until every
row is updated, so an implementation must either update its aliased cache row
last or preserve a scratch copy. Exact-zero entries of \(d\) can skip
individual rows, but the decisive zero-maintenance ceiling below already
grants all maintenance for free.

## Leaving-position locality

The existing exact diagnostic trace
`/tmp/sifting-falsifier/trajectory.bin` contains 4,399 records and marks 33
refactorizations. Its leaving-position census is:

| measure | value |
|---|---:|
| basis rows | 1,525 |
| pivots | 4,399 |
| unique leaving positions | 1,068 |
| repeat events | 3,331 |
| immediate repeats | 251 |
| median reuse gap | 110 pivots |
| p75 reuse gap | 413 pivots |
| p90 reuse gap | 1,097 pivots |
| p95 reuse gap | 1,675 pivots |
| maximum reuse gap | 4,030 pivots |

The most frequently selected position occurs 63 times, but frequency alone
does not eliminate acquisition or eviction misses. The following table
replays ordinary LRU and also an impossible future-aware Belady policy. “Keep”
retains cache identities across refactorizations; “clear” discards them after
each marked refactorization.

| rows K | LRU keep | LRU clear | offline-optimal keep | offline-optimal clear |
|---:|---:|---:|---:|---:|
| 1 | 251 | 248 | 251 | 248 |
| 2 | 328 | 324 | 592 | 570 |
| 4 | 459 | 449 | 911 | 855 |
| 8 | 646 | 620 | 1,244 | 1,106 |
| 16 | 890 | 836 | 1,622 | 1,324 |
| 32 | 1,204 | 1,086 | 2,035 | 1,398 |
| 64 | 1,512 | 1,276 | 2,434 | 1,398 |
| 128 | 2,032 | 1,391 | 2,836 | 1,398 |
| 256 | 2,560 | 1,398 | 3,152 | 1,398 |
| 512 | 3,008 | 1,398 | 3,331 | 1,398 |
| 1,024 | 3,330 | 1,398 | 3,331 | 1,398 |
| 1,525 | 3,331 | 1,398 | 3,331 | 1,398 |

Clearing at refactors caps the clairvoyant hit count at 1,398, or 31.78% of
pivots. Retaining rows removes that reset penalty but requires numerical
reconciliation with each freshly rebuilt authoritative factor.

## Whole-wall ceilings

The contemporaneous campaign median supplied for this falsifier was:

```text
whole wall       588.6 ms
BTRAN phase       99.6 ms
pivots             4,399
BTRAN / pivot     22.642 us
refactorizations      33
```

Three increasingly realistic ceilings follow.

### 1. Delete every BTRAN

```text
99.6 / 588.6 = 16.9215% whole wall
```

This is already below the 17.7013% board requirement and the 20% falsifier
gate. It assumes impossible zero-cost hits on all pivots.

### 2. Charge compulsory cache misses only

Even an unlimited cache cannot hit a leaving position before its first touch:

```text
maximum hits         = 4,399 - 1,068 = 3,331
maximum saving       = 3,331 * 22.642 us = 75.419 ms
whole-wall saving    = 75.419 / 588.6 = 12.813%
```

If numerical authority requires clearing after refactorization, the
future-aware maximum is:

```text
maximum hits         = 1,398
maximum saving       = 31.653 ms
whole-wall saving    = 5.378%
```

Both figures still grant free lookup, insertion, row updates, row storage,
and refactor handling.

### 3. Charge row streaming and refactor rebuilds

An in-memory, single-thread SciPy `dger` probe reached a best observed
effective row read/write bandwidth of 52.7 GB/s. Rather than charge the
measured call times, the projection gives every cache size that best bandwidth
continuously and charges only

\[
t_{maintenance}\ge\frac{16Km\cdot4399}{52.7\times10^9}.
\]

This is deliberately optimistic: it omits coefficient construction, lookup,
insertion, scratch preservation, cache conflicts, and all non-row traffic.
Retained caches are charged 33 rebuilds at only the average 22.642 us BTRAN
cost per row. Cleared caches pay no rebuild but use the refactor-reset hit
trace. Hits are the future-aware offline optimum, not LRU.

| K | maintenance lower bound | rebuild charge | keep/rebuild net wall | clear net wall |
|---:|---:|---:|---:|---:|
| 1 | 2.037 ms | 0.747 ms | 0.493% | 0.608% |
| 2 | 4.073 ms | 1.494 ms | 1.331% | 1.501% |
| 4 | 8.147 ms | 2.989 ms | **1.612%** | **1.905%** |
| 8 | 16.294 ms | 5.977 ms | 1.002% | 1.486% |
| 16 | 32.588 ms | 11.955 ms | -1.328% | -0.443% |
| 32 | 65.175 ms | 23.909 ms | -7.307% | -5.695% |
| 64 | 130.351 ms | 47.819 ms | -20.907% | -16.768% |
| 128 | 260.701 ms | 95.638 ms | -49.631% | -38.914% |
| 256 | 521.403 ms | 191.275 ms | -108.955% | -83.206% |

The best charged projection is 1.905% whole-wall improvement at K=4 under
the refactor-clear policy. The best retain/rebuild projection is 1.612% at
the same size. Both use clairvoyant replacement and a favorable bandwidth
floor, so they are ceilings rather than implementation forecasts.

## Numerical authority objection

The algebra above is exact over the reals, but repeated IEEE-754 rank-one
updates do not reproduce a fresh Forrest–Tomlin BTRAN's arithmetic path. This
is not speculative. The existing exact next-pivot lookahead falsifier reports:

| measure | result |
|---|---:|
| predictions / leaving-position matches | 4,398 / 4,398 |
| rho mismatches above 1e-9 | 60 |
| rho mismatches away from refactors | 58 |
| maximum rho error | 7.495e-6 |
| pivot-row mismatches above 1e-8 | 25 |
| Harris entering-choice mismatches | **15** |

That probe's output remained behavior-identical only because the shadow row
was not consumed. The mismatches occur predominantly away from
refactorization, so rebuilding cached rows only at the 33 refactors is not an
adequate authority gate.

An exact correction for approximate cached row \(\hat w_i^T\) starts from

\[
r_i^T=e_i^T-\hat w_i^TB
\]

and solves

\[
B^T\delta_i=r_i,
\qquad
w_i=\hat w_i+\delta_i.
\]

That is itself a BTRAN, in addition to forming the residual. Alternatively,
a fresh BTRAN can verify and replace every cache hit. Either approach removes
the intended saving. A proven interval/error-bound acceptance test could in
principle avoid some corrections, but no such authority proof exists, and
even free perfect correction cannot exceed the 12.813% compulsory-miss
ceiling.

## Reproducible read-only command log

The investigation reused existing worktrees and `/tmp` artifacts. The core
trace parser and cache replay were run in memory with:

```bash
/home/evan/dev/linprogx-sifting/.venv/bin/python - <<'PY'
import collections
import struct
from pathlib import Path

payload = Path('/tmp/sifting-falsifier/trajectory.bin').read_bytes()
n, m, tol = struct.unpack_from('<iid', payload, 8)
offset = 24
positions = []
refactors = []
while offset < len(payload):
    fields = struct.unpack_from('<9i', payload, offset)
    offset += 36
    iteration, flags, leaving, _, _, support_n, candidates_n, ambiguous_n, transitions_n = fields
    positions.append(leaving)
    if flags & 1:
        refactors.append(len(positions) - 1)
    offset += 4 * (support_n + candidates_n + ambiguous_n + 2 * transitions_n)
assert offset == len(payload)

def lru(size, clear=False):
    cache = collections.OrderedDict()
    hits = 0
    refactor_set = set(refactors)
    for pivot, leaving in enumerate(positions):
        if leaving in cache:
            hits += 1
            cache.move_to_end(leaving)
        else:
            if len(cache) >= size:
                cache.popitem(last=False)
            cache[leaving] = None
        if clear and pivot in refactor_set:
            cache.clear()
    return hits

def belady(size, clear=False):
    future = collections.defaultdict(collections.deque)
    for pivot, leaving in enumerate(positions):
        future[leaving].append(pivot)
    cache = set()
    hits = 0
    refactor_set = set(refactors)
    for pivot, leaving in enumerate(positions):
        future[leaving].popleft()
        if leaving in cache:
            hits += 1
        else:
            if len(cache) >= size:
                victim = max(
                    cache,
                    key=lambda row: future[row][0] if future[row] else len(positions) + 1,
                )
                cache.remove(victim)
            cache.add(leaving)
        if clear and pivot in refactor_set:
            cache.clear()
    return hits

for size in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1525):
    print(size, lru(size), lru(size, True), belady(size), belady(size, True))
PY
```

The optimistic row-stream calibration was:

```bash
OPENBLAS_NUM_THREADS=1 /home/evan/dev/linprogx-sifting/.venv/bin/python - <<'PY'
import time

import numpy as np
from scipy.linalg.blas import dger

m = 1525
rng = np.random.default_rng(1)
for k, repetitions in ((1, 4000), (2, 4000), (4, 4000), (8, 3000),
                       (16, 2500), (32, 1800), (64, 1200), (128, 800),
                       (256, 500), (512, 300), (1024, 180)):
    rows = np.asfortranarray(rng.normal(scale=1e-8, size=(k, m)))
    coefficients = rng.normal(scale=1e-8, size=k)
    rho = rng.normal(scale=1e-8, size=m)
    samples = []
    for _ in range(9):
        started = time.perf_counter_ns()
        for _ in range(repetitions):
            dger(-1.0, coefficients, rho, a=rows, overwrite_a=1)
        samples.append((time.perf_counter_ns() - started) / repetitions / 1000)
    usec = float(np.median(samples))
    effective_gbps = 16 * k * m / (usec * 1e-6) / 1e9
    print(k, usec, effective_gbps)
PY
```

The charged projection was reproduced with:

```bash
/home/evan/dev/linprogx-sifting/.venv/bin/python - <<'PY'
pivots = 4399
m = 1525
wall_ms = 588.6
btran_ms = 99.6
btran_us = btran_ms * 1000 / pivots
refactors = 33
bandwidth = 52.7e9
hits = {
    1: (251, 248), 2: (592, 570), 4: (911, 855), 8: (1244, 1106),
    16: (1622, 1324), 32: (2035, 1398), 64: (2434, 1398),
    128: (2836, 1398), 256: (3152, 1398), 512: (3331, 1398),
}
for k, (keep_hits, clear_hits) in hits.items():
    maintenance_ms = 16 * k * m * pivots / bandwidth * 1000
    rebuild_ms = refactors * k * btran_us / 1000
    keep = (keep_hits * btran_us / 1000 - maintenance_ms - rebuild_ms) / wall_ms
    clear = (clear_hits * btran_us / 1000 - maintenance_ms) / wall_ms
    print(k, maintenance_ms, rebuild_ms, 100 * keep, 100 * clear)
PY
```

Supporting artifact/source inspection used only `cat`, `sed`, `find`, `rg`,
and Python JSON reads against:

- `/home/evan/dev/linprogx-sifting/AGENTS.md`
- `/home/evan/dev/linprogx-sifting/experiments/sifting_falsifier.py`
- `/tmp/sifting-falsifier/results.json`
- `/tmp/sifting-falsifier/trajectory.bin`
- `/home/evan/dev/linprogx-lookahead/AGENTS.md`
- `/home/evan/dev/linprogx-lookahead/experiments/lookahead_btran_falsifier_2026_07_21.md`
- `/home/evan/dev/linprogx-lookahead/experiments/lookahead_btran_probe.py`
- `/tmp/lookahead_btran_greenbea_2026_07_21.json`

One stale guessed JSON path,
`/tmp/lookahead-btran-falsifier/results.json`, returned `FileNotFoundError`; it
did not mutate state and was followed by the documented artifact path above.

## Git, filesystem, and network audit

- **Git:** no Git command was run.
- **Network:** no network-capable command, package installation, fixture
  download, web request, or external solver request was run.
- **Filesystem:** the solver source and all existing experiment and `/tmp`
  artifacts were read-only. The BLAS benchmark allocated arrays only in
  process memory. This Markdown report is the sole file created for the
  selected-inverse task.

**Final verdict: KILLED.** The mechanism is capped below the target before
maintenance, is only about a 2% optimistic charged opportunity, and lacks a
valid floating-point authority path that could consume cached rows without
repaying BTRAN cost.
