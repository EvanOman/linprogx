# HiGHS vs linprogx presolve diff — greenbea (+woodw, 80bau3b), 2026-07-17

Measurement-only probe. No solver source changed; throwaway scripts only
(`experiments/greenbea_highs_presolve_probe.py`). HiGHS behavior read from its
own runtime logs and documented options — no HiGHS source was read.

## Tools, versions, options used

- `highspy==1.14.0` (HiGHS 1.14.0, git 7df0786), installed into the dev venv
  via `uv pip install "highspy==1.14.0"` (14-day-old 1.15.1 was blocked by the
  local index; 1.14.0 is 101 days old and resolved cleanly). scipy 1.17.1's
  bundled HiGHS no longer exposes `scipy.optimize._highs`, so highspy is the
  route to the per-rule report.
- Fixtures: `/tmp/lpsuite/lp_{greenbea,woodw,80bau3b}.mat`, loaded with
  `scipy.io.loadmat` exactly as `experiments/suite_bench.py` does. Built as an
  equality LP in HiGHS (`row_lower_ == row_upper_ == b`, col bounds from
  lo/hi, `kHighsInf` for infinities).
- HiGHS options set (probed via `getOptionValue`, not guessed):
  `presolve_rule_logging=True` (emits the per-rule reduction table),
  `log_dev_level=2`, `output_flag=True`, `log_to_console=False`,
  `log_file=<path>`, `highs_debug_level=0`. Ran `Highs.presolve()` only, then
  read shape from `getPresolvedLp()`.
- Ablation via the documented `presolve_rule_off` bitmask. HiGHS's own log
  prints the legend: Forcing row=bit6(64), Forcing col=bit7(128), **Free col
  substitution=bit8(256)**, Doubleton equation=bit9(512), Dependent
  equations=bit10(1024), Dependent free cols=bit11(2048), **Aggregator=bit12
  (4096)**, Parallel rows and columns=bit13(8192), Sparsify=bit14, Probing=15,
  Enumeration=16. Empty/singleton/fixed/dominated/forcing-row are not
  individually toggleable, so those are attributed from the per-rule table.
- linprogx side: `linprogx.presolve.presolve_matrix()` on the same CSR, with
  `float("inf")` bounds exactly as `sparse.py` passes them. This reproduced the
  census greenbea shape **1525×3868, removed 867r/1730c** with identical rule
  counts, confirming the harness is faithful.

## HiGHS per-rule reduction tables (rows / cols / calls)

### greenbea — raw 2392×5598×31070 → presolved **951×3158×23609**

| HiGHS rule | rows | cols | calls |
| --- | ---: | ---: | ---: |
| Empty row | 3 | 0 | 3 |
| Singleton row | 78 | 76 | 78 |
| Empty column | 0 | 6 | 6 |
| Fixed column | 0 | 236 | 236 |
| Dominated col | 0 | 11 | 11 |
| Forcing row | 190 | 815 | 112 |
| **Free col substitution** | **24** | **24** | 24 |
| Doubleton equation | 577 | 577 | 577 |
| Dependent equations | 0 | 0 | 1 |
| **Aggregator** | **551** | **551** | 6 |
| Parallel rows and columns | 18 | 144 | 2 |
| **Total** | **1441** | **2440** | |

### woodw — raw 1098×8418×37487 → presolved **557×4095×15823**

| HiGHS rule | rows | cols | calls |
| --- | ---: | ---: | ---: |
| Dominated col | 0 | 1118 | 1118 |
| Forcing row | 386 | 3050 | 386 |
| **Free col substitution** | **4** | **4** | 4 |
| Doubleton equation | 5 | 5 | 5 |
| Dependent equations | 0 | 0 | 1 |
| **Aggregator** | **146** | **146** | 2 |
| Parallel rows and columns | 0 | 0 | 1 |
| **Total** | **541** | **4323** | |

### 80bau3b — raw 2262×12061×23264 → presolved **1537×9876×20012**

| HiGHS rule | rows | cols | calls |
| --- | ---: | ---: | ---: |
| Singleton row | 28 | 28 | 28 |
| Empty column | 0 | 87 | 87 |
| Fixed column | 0 | 520 | 520 |
| Dominated col | 0 | 532 | 532 |
| Forcing row | 9 | 55 | 9 |
| **Free col substitution** | **61** | **61** | 61 |
| Doubleton equation | 212 | 212 | 212 |
| Dependent equations | 0 | 0 | 1 |
| **Aggregator** | **415** | **415** | 6 |
| Parallel rows and columns | 0 | 275 | 3 |
| **Total** | **725** | **2185** | |

## linprogx per-rule reduction (presolve_matrix, same instances)

Row-removal attribution sums exactly to `removed_rows`; the col counts below are
the reduction_counts keys.

### greenbea — → **1525×3868×23274** (removed 867r / 1730c) — matches census exactly

| linprogx rule | rows removed | reduction_counts |
| --- | ---: | --- |
| forcing_rows | 351 | forcing_rows=351, forcing_columns=1102 |
| doubletons | 438 | doubletons=438 |
| singleton_rows | 47 | singleton_rows=47 |
| column_singletons | 23 | column_singletons=23 |
| empty_rows | 8 | empty_rows=8 |
| fixed_columns | 0 | fixed_columns=113 |
| duplicate_columns | 0 | duplicate_columns=7 |
| **aggregator / free-col-sub** | **0** | **absent — no such rule** |
| **Total rows** | **867** | |

### woodw — → **707×5363×19807** (removed 391r / 3055c)

`forcing_rows=386, forcing_columns=3050, column_singletons=4, doubletons=1`.
No aggregator/free-col-sub. (Note: linprogx removes NO dominated columns; HiGHS
removes 1118 here — see candidate 3.)

### 80bau3b — → **1992×11155×21798** (removed 270r / 906c, deep fixpoint)

`fixed_columns=531, empty_columns=76, doubletons=192, column_singletons=45,
duplicate_columns=9, forcing_rows=8, forcing_columns=28, singleton_rows=25`.
No aggregator/free-col-sub. (This is `presolve_matrix`'s full fixpoint; the
public solver currently ships the shallower 2079×11878 per the census gate.
Either way HiGHS's 1537 rows is unreached.)

## The diff, with row/col attribution

The row-count deficit is the campaign-relevant metric (it drives dual-simplex
pivots: 2836 HiGHS vs 4399 linprogx on greenbea). Ablation isolates it cleanly.

### greenbea ablation (HiGHS, `presolve_rule_off`)

| HiGHS config | rows | cols | nnz | Δrows | Δcols |
| --- | ---: | ---: | ---: | ---: | ---: |
| all rules on (baseline) | 951 | 3158 | 23609 | 0 | 0 |
| **Aggregator OFF** | **1498** | 3705 | 22810 | **+547** | +547 |
| Free-col-sub OFF | 951 | 3158 | 23609 | 0 | 0 |
| **Aggregator + Free-col-sub OFF** | **1521** | 3729 | 22958 | **+570** | +571 |
| Doubleton OFF | 983 | 3191 | 23404 | +32 | +33 |
| Parallel OFF | 1043 | 3377 | 25055 | +92 | +219 |
| Aggr+FreeCol+Doubleton OFF | 1961 | 4175 | 23886 | +1010 | +1017 |

**linprogx ships 1525 rows. HiGHS with Aggregator+Free-col-sub disabled = 1521
rows.** The entire ~574-row gap is the equality-aggregation family.

Reading the row diff by HiGHS category vs linprogx (order-dependent, but the
signs are unambiguous): Aggregator **+551**, Doubleton **+139** (HiGHS 577 vs
our 438), Free-col-sub **+24**, Singleton row +31, Parallel +18. linprogx
actually removes **more** forcing rows (351 vs HiGHS 190) — because, lacking an
aggregator, rows HiGHS would have substituted away remain to be caught as
forcing. Free-col-sub shows 0 marginal until the aggregator is also off because
the aggregator subsumes free-column substitutions; together they own the gap.

### woodw / 80bau3b — same cause, confirmed by ablation

| Instance | HiGHS baseline rows | Aggr OFF | Aggr+FreeCol OFF | linprogx rows |
| --- | ---: | ---: | ---: | ---: |
| woodw | 557 | 703 (+146) | **707 (+150)** | **707** (exact) |
| 80bau3b | 1537 | 1960 (+423) | **1997 (+460)** | **1992** (≈) |

On woodw the match is exact (707 = 707); on 80bau3b within 5 rows. On all three
instances HiGHS's row advantage over linprogx is **entirely** the Aggregator +
Free-col-substitution family and nothing else.

### Column side (secondary)

Columns also mostly follow the aggregator (1 col removed per aggregation). The
extra col-only levers linprogx lacks:
- **Dominated column** (dual-domination): greenbea 11, woodw **1118**, 80bau3b
  **532** cols; removes 0 rows. Large on woodw/80bau3b width but no pivot-count
  benefit.
- **Parallel rows and columns**: greenbea 18r/144c (ablation +92r/+219c),
  80bau3b 0r/275c. linprogx's `duplicate_columns` catches only 7 on greenbea —
  HiGHS's parallel detection (proportional rows/cols, not just exact dup) is
  strictly stronger.

## RANKED shortlist of presolve rules linprogx lacks

### 1. Equality-row aggregation (Aggregator + free-column substitution) — THE cause

**What HiGHS does.** Pick an equality row and a column with a usable pivot
coefficient in it, express that column from the row, substitute it into every
other row it appears in (accepting bounded fill), then delete the row and the
column. "Free col substitution" is the special case where the pivot column has
no finite bound. This *generalizes linprogx's doubleton* (the k=2 special case)
to equality rows of any support.

**Measured attribution.** Aggregator alone: greenbea +547 rows, woodw +146,
80bau3b +423. Aggregator+free-col-sub: +570 / +150 / +460 — closing the row gap
to 1521 vs linprogx 1525 (greenbea), 707 vs 707 (woodw), 1997 vs 1992
(80bau3b). This is the whole HiGHS presolve row advantage on the set.

**Projected greenbea shape if adopted.** From linprogx 1525×3868×23274 toward
HiGHS **951×3158×23609**. Note HiGHS ends with 1.4% *more* nnz — aggregation
trades a little fill for 574 fewer rows; that is the deal that cuts DS pivots
4399→2836. Fewer rows is the lever, not fewer nnz.

**Implementation surface.**
- `src/linprogx/presolve.py`: `_presolve_eq_box_python` already builds
  `_Doubleton` (k=2) and `_ColumnSingleton` (col in exactly one row) records and
  replays them in `postsolve_x`. The aggregator is the k>2 generalization of the
  doubleton loop: add an `_Aggregation` record `(eliminated_col, pivot_coef,
  rhs, terms)` — structurally identical to the existing `_ColumnSingleton`
  record, whose postsolve/`_remap_record` logic already substitutes a column
  from a list of `terms`. Reuse the existing `max_fill` guard (default 5;
  greenbea already hits it on 4 doubletons) to bound fill.
- `src/linprogx/_csparse.c`: the fast path (`_c_presolve_v2` / candidate scan)
  must gain the same rule to matter at scale; the Python path is enough for the
  falsification prototype.
- `_empty_reduction_counts()`: add an `"aggregator"` key.
- HIGH-RISK per AGENTS.md ("Presolve reductions and reconstruction of primal or
  dual solutions") — requires characterization tests + an external-oracle check
  before behavior change.

**Falsification probe.** Prototype aggregation in the Python presolve behind an
env flag (`LINPROGX_PRESOLVE_AGGREGATOR=1`), reconstruct original space through
the existing postsolve, feed the unchanged Dantzig/DS solver. Record: reduced
(m,n,nnz), pivot count, every DS phase, objective, and original-space equality
+ bound residual, plus paired public wall on greenbea/woodw/80bau3b.

**Kill criterion.** Kill if greenbea presolved rows do not fall below 1000
(from 1525), if DS pivots do not fall ≥20% (below 3520 from 4399), if public
wall does not fall ≥25%, if fill pushes nnz materially past HiGHS's +1.4% such
that per-pivot cost eats the pivot-count win, or if postsolve misses the
existing 2e-5 external-oracle objective/residual/status gate.

### 2. Dominated-column elimination (dual domination)

**What HiGHS does.** Uses dual feasibility / column-domination reasoning to fix
a column at a bound when another column dominates it. Removes columns only
(0 rows).

**Measured attribution.** greenbea 11 cols, **woodw 1118 cols**, **80bau3b 532
cols**. linprogx removes 0 dominated columns anywhere. No row benefit, so no
direct DS pivot-count help, but it cuts factorization width and matvec cost —
relevant to the woodw/80bau3b IPM route and PDHG-style steps.

**Projected shape.** greenbea negligible; woodw would shed up to ~1118 of its
5363 residual cols; 80bau3b up to ~532 of 11155.

**Implementation surface.** New rule in `presolve.py` + `_csparse.c`; needs a
dual-feasibility test over column cost/bounds. Distinct machinery from
aggregation. `reduction_counts` key `"dominated_columns"`.

**Falsification probe.** No-solve reducer that reports resulting (m,n,nnz) and
estimated factor/matvec work on woodw and 80bau3b before any solver change.

**Kill criterion.** Kill unless the shape-only probe shows ≥10% column (or
factor-work) reduction on woodw or 80bau3b; on greenbea it is already
negligible (11 cols) and must not be justified there.

### 3. Parallel/proportional row-and-column detection

**What HiGHS does.** Merges parallel (proportional, not just identical) rows and
columns. linprogx's `duplicate_columns` catches only exact duplicates (7 on
greenbea).

**Measured attribution.** greenbea 18 rows / 144 cols (ablation +92r/+219c with
cascade), 80bau3b 0r/275c, woodw 0. Secondary lever; a real but small row
contribution on greenbea on top of the aggregator.

**Projected shape.** After the aggregator lands, parallel detection removes an
extra ~92 rows / ~219 cols on greenbea toward HiGHS's 951×3158.

**Implementation surface.** Generalize the existing duplicate-column path in
`presolve.py`/`_csparse.c` from exact-match to proportional (normalized
coefficient hashing with a scale factor), and extend to rows. Overlaps the
already-killed "generalized parallel-column merge" (census candidate 6) — but
that census probe measured a *standalone* 3.3% ceiling; here it is a cheap
*follow-on* to the aggregator, not a standalone win.

**Falsification probe.** No-solve reducer reporting (m,n,nnz) delta on greenbea
*after* the aggregator prototype is applied (measure the marginal, not
standalone).

**Kill criterion.** Kill unless it removes ≥50 additional rows on greenbea on
top of the aggregator; do not pursue standalone (census already killed that at
3.3%).

## Recommendation

Pursue candidate 1 only, first. The ablation is unambiguous: equality-row
aggregation (Aggregator + free-column substitution) is the *entire* source of
HiGHS's presolve row advantage on greenbea, woodw, and 80bau3b — disabling
just those two HiGHS rules reproduces linprogx's row count to within 0–5 rows on
all three. It is also the natural k>2 generalization of machinery linprogx
already ships (doubleton + column-singleton records, `max_fill` guard,
postsolve replay), so the implementation surface is an extension, not a new
subsystem. Candidates 2 and 3 are follow-ons that widen columns and shave a few
more rows but do not touch the primary pivot-count deficit.

## Dead ends / caveats

- HiGHS 1.14.0 vs scipy 1.17.1's bundled HiGHS: exact per-rule counts could
  shift by version, but the presolved *shapes* match the census (951×3158×23609
  greenbea, 557×4095 woodw, 1537×9876 80bau3b), so the attribution is on the
  same target the campaign benchmarks against.
- `presolve_rule_off` cannot toggle empty/singleton/fixed/dominated/forcing-row
  individually (only rules 6–16), so those are attributed from the per-rule
  table, not ablation. The aggregator (12) and free-col-sub (8) — the ones that
  matter — are both toggleable, so the key result is ablation-backed.
- highspy 1.15.x (14 days old) was blocked by the local index's release-age
  policy; used 1.14.0 (101 days). Not a bypass.
