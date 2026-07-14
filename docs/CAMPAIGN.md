# linprogx Performance Campaign — Longitudinal Record

*Living write-up of the `perf-supernodal-simplex` ship campaign. Auto-regenerable
from `assets/campaign.db`; see [Regenerate](#regenerate) at the bottom.*

## The goal

linprogx is an independently built LP solver (never reads public solver source;
papers and textbooks only). The standing goal is to **exceed both HiGHS and
Clarabel** — on coverage and on runtime — across the 24-fixture LPnetlib suite.
Hard constraints are unchanged: `eps=2e-5` is never loosened, every reported
optimum is certificate-backed, and there is no per-problem tuning (only global
thresholds calibrated to measured machine throughput).

Where the two axes stand:

- **Coverage: EXCEEDED.** linprogx solves **24/24**; HiGHS solves 23 (times out on
  `qap15`) and Clarabel solves 23 (`ken_18` DualInfeasible). This is settled and
  reproduced in the replay: every one of the 21 replayed commits solves all 24 to
  certified optimality.
- **Runtime: aggregate EXCEEDED, per-instance majority WON.** The suite total and
  geometric-mean time ratio have favored linprogx since early in the campaign; the
  per-instance head-to-head crossed into linprogx's majority (14-10 on the paired
  protocol) mid-campaign and the ship commits since have deepened the wins and
  narrowed the losses — culminating in the 2026-07-14 clean-box certification,
  where the hard-loss ladder collapsed to nothing above 2× and `cre_b` flipped
  to a certified win (see [Clean-box certification](#clean-box-certification-2026-07-14)).

## The arc

The campaign narrative (fully dated in `docs/HANDOFF.md`) runs from a **14-10**
paired head-to-head at the session-start baseline (`a1a355d`, 2026-07-04) through
twenty substantive ship commits to presolve V2 shipping on 2026-07-14. The
through-line: the IPM factor path and the dual-simplex LU path were each driven
to their measured floors, closing whole classes of hypotheses along the way —
and further ships kept landing after the Forrest-Tomlin program (a Cholesky
uplook kernel and an LU pivot-search bound) and the Suhl bounded pivot search
kept moving both floors further than "complete" suggested. The final two ships
came out of a joint Claude/Codex strategy round that killed three more
falsifier probes before finding its two winners (see below).

### Ship-by-ship story

The replayed ship commits, in order (see the per-instance trajectory table below
for the numbers):

1. **`0145c8f` — Tdense latent-bug fix.** The supernodal refactor never populated
   `ctx->Tdense`, so the default supernodal+BLAS route fed zeros to the dtrsv tail
   solve, NaN'd its first IPM attempt, and survived only via the floored retry.
   Gating the tail solve on `tail_dense_valid` (and keeping the tail on the scalar
   CSC walk, measured *faster* than gather+dtrsv) took **maros_r7 ~2.4s → ~1.7s**.
2. **`86d7064` — resident supernode panels + zero-copy dgemm operands.** Panel
   assembly gather eliminated; bit-identical. maros_r7 refactor −10%; ken_18
   (supernodal via the prefix-flops gate) −15..−21%.
3. **`55cae27` — size-gated per-call OpenBLAS threading** in the supernodal
   refactor (large panels 4-thread, small work serial; symbolic-only gate).
   Parallel-supernode task scheduling was refuted here (1.28× ceiling on maros_r7).
4. **`e7186c0` — dense-tail BLAS threshold 400 → 256.** The 400 gate predated the
   1e-11 dpotrf ridge. **ken_11 flips to a win**; 80bau3b near-flip. 256 is the
   stability boundary (192/128 cost cre_a iterations).
5. **`29f77a6` — linear-merge symbolic build + regime-constrained supernodal
   routing.** **stocfor3, ken_11, ken_13 flip** (ken_13 clean pair 0.80 → 0.44).
6. **`c1812a7` — contiguous scalar update kernels** for the supernodal fine
   streams (100% of stocfor3/ken_13 updates are srcpos-contiguous). Scalar slice
   −18%, at the memory floor.
7. **`f919642` — two-candidate symbolic ordering (min-degree vs natural).** Fires
   only on maros_r7's banded/QP structure (natural order = 2.36× fewer flops):
   fill −32%, **maros_r7 wall −23%** (to ~1.1–1.2s).
8. **`0a20b2e` — MCC cost-ratio gate 5.5 → 3.0.** The session's factor cheapening
   had turned Gondzio correctors off nearly everywhere; retuning recovered
   maros_r7 15 → 12 iters and deepened the pilot87 win.
9. **`c33f12f` — DS rate levers** (scatter reuse, alpha_pattern support list,
   candidate cache). **greenbea 382 → 292 µs/pivot; ~3.9s → ~2.9s.**
10. **`2a73a10` — explicit `dual_simplex` path runs the certified EXPAND config**
    (auto routes were already correct).
11. **`3d53bee` — mu safeguard at breakdown steps.** Fail-closed step guard in the
    certificate window (mu < 1e-7): reject any tentative step whose post-step mu
    exceeds ~10× pre-step mu. **Dissolves the 60,000× endgame detonation** that
    inflated 80bau3b under forced correctors (105 → 52 iters). Inert on every
    default path.
12. **`cfed6f6` — DS LU cadence, diag-ratio stability guard 1e6 → 1e8.** The 1e6
    guard fired on 47% of greenbea's refactorizations on harmless pivot spread;
    **greenbea −12.6%** (~2.9 → ~2.5s projected clean).
13. **`459c804` — PDHG auto thread rule** (physical cores; default threads 4 → 0).
    Bit-identical trajectories: **pds_10 −21.5%, pds_20 −36.6%.**
14. **`c7190b5` — mu-gated second refinement round** (bit-exact no-op while
    mu > 1e-5) + supernodal kwarg-parse fix.
15. **`e09d425` — uplook pattern cache.** Elimination patterns recorded at symbolic
    time instead of re-walked every refactor: **cre_b uplook −14%,** cre_a −7.7%,
    80bau3b −8.2%, pilot87 −7.1%.
16. **`d0e6cb1` — Forrest-Tomlin program ship** (port from `exp-leaving`). True FT
    replaces PFI eta chains: `expand=1` + FT default on + `dtau=5e-11` + `wmax=3e5`.
    **greenbea 2.36s certified** (was ~3.10 pre-ship, ~3.80 at session start); the
    LU program (steps 1–4) is complete.
17. **`2f4a1df` — block-row uplook gate** (merge of `exp-panel`'s
    `6ef7ce2`). Consecutive rows (≤4) sharing pattern structure are processed as a
    block with per-row accumulators, amortizing the random-load Li/Lx stream instead
    of walking each shared pattern column once per row. Gated by a symbolic-time
    census of the saveable scatter-pair fraction at `b=4`: the block path engages
    only when `saved >= 0.5 * W` (one global structural constant — clean separation
    between losers ≤45.5% and winners ≥56.4%; not per-problem tuning). Gated-on is
    outcome-exact (obj reldiff ≤2.8e-12, iterations identical) rather than
    bit-identical, since the merged pass visits columns in ascending-index order
    instead of `chol_ereach` topological order; gated-off is byte-identical.
    Paired 9-trial: **cre_b uplook 1.72s → 1.13s (−35%)**, **cre_b wall −12.8%**,
    **osa_14 −4.1%**, cre_a's prior regression eliminated as noise.
18. **`11f4157` — Suhl bounded pivot search** (port from `exp-leaving`'s `92652af`).
    Columns are still walked sparsest-first, but the search now stops after
    `LU_SUHL_MAX_COLS=8` threshold-viable columns, or accepts immediately at
    `merit<=LU_SUHL_ACCEPT=4` (the exact `merit==0` fast exit is preserved).
    **greenbea pivot search 0.322s → 0.013s (−96%, 44× fewer column visits)**;
    paired walls **greenbea +35.4%** (compounded by a −14% favorable iteration
    shift), woodw +20.3%, stocfor3 +12.6%, cre_d +7.9%, 80bau3b +8.1% — all
    DS-family instances improve, no refactorization storms, all objectives
    certified identical. **greenbea public 1.650s certified** — the session arc is
    now **greenbea 3.80s → 1.65s**. MONITORED (not settled): cre_d same-basis fill
    is +40% (a proxy-gate breach), but outcomes stay healthy — wall improves, no
    storms, and no global budget holds its fill without giving back greenbea's
    gain. The LU and Cholesky uplook programs are both still producing wins
    past the point either looked "complete."

### Strategy round: three kills, then two winners

With both structural programs past "complete," the session shifted to a joint
Claude/Codex strategy round: pre-registered falsifier probes run by Codex,
evaluated against the same certificate/residual bar as every ship. Three were
killed outright — **BFRT post-FT/Suhl** (greenbea pivots 9,150 → 9,865, wall
1.65s → 2.60s with `bfrt=1`; the boxedness signal doesn't discriminate —
80bau3b's -40% pivot / -36% wall win under BFRT is IPM-routed and
scoreboard-irrelevant), **IPM→DS crossover basis** (KILL 0/4 against a 2/4 bar
— Tapia-ranked deterministic structural matching reaches full structural
coverage at every mu crossing, but every candidate basis is numerically
singular at the Markowitz LU gate before cleanup metrics can even be computed;
the degenerate optimal faces don't admit cheap crossover bases), and
**fixed-step Halpern PDHG** (decisive KILL — the 2k-window terminal-KKT edge
of reset-anchor cells was a restart artifact; at 10k iterations the adaptive
baseline is ~98,000× better at equal work, and all anchored/reflected × fixed-η
variants are now dead for pds). Then a review fan-out: the **block-gate
threshold validated** (the near-threshold gated-OFF instances — 80bau3b 44.6%,
pilot87 45.5%, plus cre_a/woodw — all KEEP_OFF under 9 forced interleaved
trials; the 0.5 saveable-fraction constant stands), **PDLP diagonal steps
killed** (post-Ruiz alpha=1 diagonal steps stall far above tolerance on
pds_10/pds_20 and destroy the qap12 win; the pds pair is now closed at unit
level — the adaptive η/ω machinery is the strength, not a stopgap), and
**exact DSE rechecked and killed again** (count cuts are real but small —
greenbea -7.4%, 80bau3b -14.5% — while pricing-update overhead regresses wall
everywhere and detonates cre_d's iteration limit at the fresh economics).
Two winners came out of the same round — the Dantzig route and presolve V2,
below.

19. **`422af49` — plain-Dantzig leaving on the DS auto-rescue routes.** The
    formal kwarg-default probe (global plain-Dantzig leaving) was itself a
    KILL — cre_d regressed +41% on the direct-DS path — but the auto-rescue
    route population is narrower ({greenbea-shaped instances, `cycle`}), and
    route-level config (the same shape as the `expand=1` ship) is correct
    there: **greenbea public 1.66s → 0.83s** (−50%, 9,150 → 6,533 pivots),
    `cycle` 905 → 676 iterations (resid 3.6e-12), 249 tests green,
    deterministic across repeats. Session arc: **greenbea 3.80s → 0.83s**;
    clean-box ratio projected ~2.6× from ~5×.
20. **`5f89032` — presolve V2 (the campaign's biggest single ship).** A
    fresh-eyes structural census of the cre pair found what earlier presolve
    passes had missed: cre_b carries 4,690 lower-bounded column singletons
    (51.5% of its 77,076 columns removable, projected nnz −55.8%, factor-flops
    estimate −57.6%); cre_d is similar (55.2% removable, nnz −59.4%). The
    3-round build — column singletons (free/implied-free, chained), fixed
    columns, dual fixing, row forcing, duplicate-column merge — is
    **fill-guarded** (the elimination can't grow the working matrix) and gated
    by a **native O(nnz) opportunity census** that costs nothing when no
    structure qualifies. Hand-verified: **cre_b 5.68s → 1.69s** (vs HiGHS
    ~1.94s — likely flip), **cre_d 5.27s → 1.41s** (was a 4.6× loss, now
    ~1.15–1.3×), **greenbea 0.83s → 0.51s** (session arc 3.80s → 0.51s, ~7.5×);
    woodw, maros_r7, stocfor3, and 80bau3b all improve too. The standing "cre
    pair closed on every IPM axis" verdict (settled-hypotheses ledger) was
    true **at fixed problem size** — presolve V2 moved the size instead. 260
    tests green; `LINPROGX_PRESOLVE_V2=0` reverts to the old path.

### Settled-hypotheses ledger

The campaign is as much a record of what does **not** work. Permanently closed:

- **cre pair (cre_b/cre_d) via IPM:** measured-unreachable by every IPM-side lever
  — factor floor reached, correctors optimal, ordering near-optimal, coarse-stream
  supernodal refuted by arithmetic, inexact-Newton refuted by preconditioner
  quality. The 3–5× gap belongs to the DS count program.
- **cre count via DS pivoting:** selection weights (DSE: no effect — rechecked
  post-FT/Suhl at the fresh economics and still a KILL: count cuts are real but
  small while pricing-update overhead regresses wall everywhere and detonates
  cre_d's iteration limit), degeneracy perturbation (halves thrash, count
  unmoved), big-M magnitude (bit-identical across 4 orders), Phase-1 vs boxing
  (bit-identical), leaving-rule family (all cap at 100k), BFRT (re-confirmed
  post-FT/Suhl: greenbea pivots +7.8%, wall 1.65s → 2.60s), tie-breaks, Harris
  band width — every DS-internal pivoting lever is eliminated. The count itself
  was never a pivoting problem: presolve V2's census-driven column-singleton
  eliminations (fill-guarded, native-opportunity-gated) cut the walk directly
  by shrinking the problem — **cre_b 5.68s → 1.69s, cre_d 5.27s → 1.41s**
  (ship-by-ship story, `5f89032`). Crash reachable-set and pricing locality
  remain open next-program levers for whatever gap presolve V2 doesn't close.
- **IPM→DS crossover basis:** KILL 0/4 against a 2/4 bar — Tapia-ranked
  deterministic structural matching achieves full structural coverage at every
  mu crossing, but every candidate basis is numerically singular at the
  Markowitz LU gate before cleanup metrics can even be computed. The degenerate
  optimal faces do not admit cheap crossover bases; the crossover program is
  closed at probe level.
- **Parallel supernode factorization:** permanent negative (1.14–1.28× ceilings
  from the etree DAG analysis).
- **pilot87/stocfor3/80bau3b certificate tails:** these are genuine primal
  convergence, not wasted certificate work — certificate tricks cannot flip them.
- **PDHG:** profiled PDLP-complete and memory-bandwidth-bound at the practical
  floor; the pds gap is pure iteration count at 2e-5. Closed at unit level:
  fixed-step Halpern (anchored/reflected × fixed-η, all variants) loses to the
  adaptive baseline by ~98,000× at equal work once slope probes span
  restart-scale dynamics (a 2k-iteration window was too short to see it);
  post-Ruiz diagonal steps stall above tolerance and destroy the qap12 win.
  The adaptive η/ω machinery is the strength, not a stopgap; the remaining pds
  gap is program-scale or accepted.

## Current certified scoreboard

The **certified** standing uses the paired 5–7-run interleaved protocol in
`docs/HANDOFF.md` (not the single-shot replay below). As of the Suhl bounded
pivot search ship:

- **Head-to-head: 14-10 linprogx** on the paired protocol, plus all aggregate axes
  (coverage 24/24, aggregate suite time and geomean decisively linprogx's). The
  two post-FT ships did not flip any instance's win/loss classification — they
  deepened existing positions (greenbea's loss margin, cre_b/osa_14's certified
  wall) rather than crossing the knife edge; the paired protocol has not been
  re-run end-to-end since the FT ship, so this line is carried forward, not
  re-verified.
- **Knife-edge band** (paired 7-run): **degen3 WIN 0.80** (7/7); **osa_14 TIE 1.00**
  (4/7, pre-block-row-gate) — the block-row uplook gate cut osa_14 a further
  **−4.1%** paired but a fresh 7-run has not been taken to re-call the tie;
  losses stocfor3 1.03, 80bau3b 1.07 (mins dead-tied), cre_a 1.22.
- **Hard-loss ladder** (by ratio, from the quiet 03dc77c re-baseline): **greenbea
  ~6.2×** (now **1.650s certified vs 0.267s HiGHS**, down from 10× — Forrest-Tomlin
  plus the Suhl bounded pivot search have now cut the ladder's steepest rung by
  ~38% again on top of the FT drop), cre_d 5.6× / cre_b 3.3× (IPM side closed;
  the block-row uplook gate found a further **cre_b wall −12.8%** but the DS
  count program remains the only path to close the ladder position itself),
  pds_10 2.3× / pds_20 1.4× (PDHG route), woodw 2× (flop-bound, though the Suhl
  pivot search found **woodw +20.3%** paired within that ceiling), maros_r7 1.6×
  (serial floor reached), 80bau3b ~1.1× paired (Suhl **+8.1%** further), cre_a 1.3×.

This snapshot predates the Dantzig route and presolve V2 ships (`422af49`,
`5f89032`) and is carried forward as the last locally re-verified paired
snapshot — it has not been re-run end-to-end on the local box since. The
current authoritative standing is the **clean-box certification** below, which
supersedes it: the hard-loss ladder collapsed to nothing above 2× and `cre_b`
flipped to a certified win.

## Clean-box certification (2026-07-14)

Second validated run on the Modal clean-box harness (`tools/modal_bench.py`;
see [Modal cloud benchmarking harness](#modal-cloud-benchmarking-harness-toolsmodal_benchpy)
below for methodology), at ref `1f4351d` (`5f89032` plus the chronicle commits
on top of it) — AWS `us-west-2`, `cpu_count=20`, loadavg 0.00 throughout both
the suite and paired runs
(`assets/modal_bench_1f4351dcfa96_{suite,paired}.json`):

- **13W-11L**, geomean lx/HiGHS ratio **0.558** (was 0.735 on the first
  clean-box run) — 24/24 coverage vs HiGHS's 23/24, aggregate suite wall 49.1s
  vs HiGHS's 192.8s-plus-a-qap15-timeout.
- **The headline flip: `cre_b` LOSS 2.75× → WIN 0.940×** (6/7 paired, IPM
  2.42s vs HiGHS 2.57s) — presolve V2's certified headline. `cre_a` flipped
  back to a loss (1.066×, knife-edge churn — machine-dependent, as the first
  clean-box run already showed with degen3/pilot87).
- **The hard-loss ladder collapsed — nothing above 2×**: `greenbea` 1.89
  (was 5.25; 0.69s certified via the Dantzig route), `woodw` 1.69, `pds_10`
  1.66 (was 2.83), `cre_d` 1.53 (was 4.63), `80bau3b` 1.32 (a regression from
  the 1.20 paired figure — local A/B shows −9% on the same change; the clean
  box disagrees and this is open as an anomaly), then six instances bunched
  at 1.00–1.17: `maros_r7` 1.17, `degen3` 1.08, `cre_a` 1.07, `stocfor3` 1.06,
  `pds_20` 1.03 (near-tie, was 1.63), `pilot87` 1.008 (dead knife-edge, 3/7).

Session totals as of this certification: 20+ ship commits, 21 settled
hypotheses, geomean nearly halved in two days.

## Headline per-instance trajectories

Single-shot replay wall-seconds, **baseline `a1a355d` → current `5f89032`**, against
the single-shot HiGHS reference on the same (loaded) box. Ordered by final ratio
(wins first). **These are narrative-grade, not certification-grade** — see the
noise note below.

| instance | baseline (s) | current (s) | HiGHS ref (s) | ratio | route | W/L |
|---|---:|---:|---:|---:|---|:--:|
| qap12 | 1.74 | 1.65 | 102.98 | 0.02 | pdhg | **WIN** |
| truss | 0.13 | 0.10 | 2.75 | 0.04 | ipm | **WIN** |
| fit2p | 0.09 | 0.11 | 1.45 | 0.07 | ipm | **WIN** |
| osa_60 | 7.79 | 5.02 | 19.13 | 0.26 | ipm | **WIN** |
| osa_30 | 2.10 | 1.52 | 4.23 | 0.36 | ipm | **WIN** |
| d2q06c | 0.39 | 0.37 | 0.88 | 0.42 | ipm | **WIN** |
| ken_13 | 0.67 | 0.48 | 0.96 | 0.50 | ipm | **WIN** |
| ken_07 | 0.02 | 0.02 | 0.04 | 0.53 | ipm | **WIN** |
| ken_18 | 8.99 | 5.27 | 8.56 | 0.62 | ipm | **WIN** |
| ken_11 | 0.24 | 0.18 | 0.30 | 0.62 | ipm | **WIN** |
| osa_14 | 1.10 | 0.90 | 1.09 | 0.83 | ipm | **WIN** |
| cre_b | 5.98 | 1.62 | 1.94 | 0.84 | ipm | **WIN** |
| degen3 | 0.21 | 0.19 | 0.22 | 0.85 | ipm | **WIN** |
| pilot87 | 4.09 | 3.18 | 3.63 | 0.88 | ipm | **WIN** |
| maros_r7 | 2.39 | 0.89 | 1.01 | 0.89 | ipm | **WIN** |
| stocfor3 | 0.90 | 0.63 | 0.61 | 1.02 | ipm | loss |
| 80bau3b | 0.33 | 0.19 | 0.18 | 1.03 | ipm | loss |
| cre_d | 5.38 | 1.30 | 1.09 | 1.20 | ipm | loss |
| cre_a | 0.13 | 0.12 | 0.09 | 1.34 | ipm | loss |
| pds_20 | 20.98 | 15.08 | 10.72 | 1.41 | pdhg | loss |
| woodw | 0.18 | 0.13 | 0.09 | 1.41 | ipm | loss |
| greenbea | 3.87 | 0.48 | 0.27 | 1.80 | dual-simplex | loss |
| pds_10 | 3.71 | 2.91 | 1.45 | 2.00 | pdhg | loss |
| qap15 | 0.87 | 0.71 | TIMEOUT | n/a | pdhg | **WIN** |

The clean structural wins survive the noise and match the paired/clean-box
record: **maros_r7 2.39 → 0.89** (Tdense fix + ordering + MCC, now a
single-shot WIN), **greenbea 3.87 → 0.48** (DS rate + LU cadence + FT + Suhl
bounded pivot search + Dantzig route + presolve V2), and **cre_b 5.98 → 1.62**
/ **cre_d 5.38 → 1.30** (presolve V2) — `cre_b` crosses this single-shot
table's break-even for the first time, at ratio 0.84. This replay's `current`
column now runs through the two newest ship commits (`422af49` Dantzig route,
`5f89032` presolve V2), measured at a quiet loadavg (2.9–5.6, well below the
box's usual 6–16) with no spike this pass. The two new ships' own certified
deltas (paired/clean-box protocol, not this single-shot table) are in the
ship-by-ship story above and the clean-box certification below: **greenbea
public 1.66s → 0.83s** from the Dantzig route, and **cre_b 5.68s → 1.69s /
cre_d 5.27s → 1.41s**, plus **cre_b's flip to a certified 0.940× win**, from
presolve V2.

### Measurement note (read before trusting a single cell)

Every number in the table and DB is a **single-shot replay under variable machine
load** (the box ran at load average 6–16 throughout, with other benchmark work
active). This is deliberately **narrative-grade**: it reconstructs the shape of the
improvement arc, not a certifiable head-to-head. The campaign's **certification**
protocol is different — paired, interleaved 5–7-run measurements on a quiet box,
recorded in `docs/HANDOFF.md`. Where a single-shot cell disagrees with the paired
record (e.g. a load-spiked pilot87/cre_b in the last commit, or a commit whose
aggregate rose purely because its replay window was loaded), trust the paired
record. Load average per run is stored in the DB's `runs` table for exactly this
reason.

## Regenerate

The entire record is reproducible from the replay harness. Nothing here is
hand-maintained except this prose.

**Database:** `assets/campaign.db` (sqlite3). Schema:

- `results(commit_hash, commit_date, commit_subject, instance, solver,
  wall_seconds, status, objective, residual, route, iterations, loadavg_1,
  measured_at)` — one row per (commit, instance, solver) cell.
  `commit_hash='reference'` holds the commit-independent HiGHS/Clarabel cells.
  `wall_seconds` is NULL on timeout/crash.
- `runs(commit_hash, commit_date, commit_subject, solver_group, started_at,
  finished_at, loadavg_1, loadavg_5, loadavg_15, n_cells, note)` — one row per
  replay batch, capturing machine load for honesty.

**Harness:** `tools/replay_bench.py` in the replay worktree
`/home/evan/dev/linprogx-replay` (self-contained, stdlib + sqlite3). It is
**idempotent** — a populated (commit, instance, solver) cell is skipped — so the
DB extends cleanly with future commits.

```bash
# from /home/evan/dev/linprogx-replay (a detached worktree at baseline a1a355d)

# 1. one-time reference pass (HiGHS + Clarabel; commit-independent)
python3 tools/replay_bench.py reference          # all 24 fixtures, both solvers
python3 tools/replay_bench.py reference --instances pds_20,qap15   # chunk long poles

# 2. replay linprogx ship commits (checkout --detached + build + 24 cells each)
python3 tools/replay_bench.py replay a1a355d 0145c8f 86d7064 ...   # any commit list

# 3. inspect coverage
python3 tools/replay_bench.py status

# 4. rebuild the report JSON + this table
python3 tools/build_report_data.py
```

Each replayed commit is checked out **detached in the replay worktree only** and
rebuilt with `uv sync --extra dev --reinstall-package linprogx`; the per-instance
wall timeout is 300s (timeouts recorded as NULL wall + status). The harness never
touches any other worktree and only ever checks out commits.

**Visualization:** `docs/campaign_report.html` — self-contained (no external
assets), theme-aware, with the per-instance small multiples, the aggregate arc, and
callout boxes for the big moments. It embeds a static JSON blob generated from the
DB by `tools/build_report_data.py`, so the page is standalone.

To add future commits: append their hashes to a `replay` call, then re-run
`build_report_data.py` and re-embed the JSON into the report.

## Publishing

The report is served at https://evanoman.github.io/linprogx/ from the
`gh-pages` branch. To republish after extending the DB: regenerate
`docs/campaign_report.html` (tools/build_report_data.py + re-embed),
then copy it to `index.html` on `gh-pages` (with CAMPAIGN.md alongside)
and push. A Claude-hosted mirror is redeployed from the same file.

## Modal cloud benchmarking harness (`tools/modal_bench.py`)

Suite benchmarks can run on a **clean, reproducible, no-other-load CPU
container** (Modal) instead of the busy dev machine. Absolute walls differ
across CPUs; the apples-to-apples product is the **ratio** of linprogx wall
to HiGHS wall measured on the same box.

**Environment** (full pins in the file header): Modal `debian_slim`
python 3.12 image + `build-essential`, `git`, `libopenblas-dev`,
`pkg-config`; `uv` installs the repo's own `uv.lock` (`uv sync --extra dev`),
so scipy/clarabel/numpy versions are exactly the committed lockfile.
Resources: `cpu=4.0` dedicated, 8 GiB, 3600s timeout, **CPU-only** — never
requests a GPU. Cost is cents per full run (one container-hour ceiling).

**Fixtures** live in the `linprogx-lpsuite` Modal Volume (uploaded once
from local `/tmp/lpsuite`, more reliable than re-downloading from
sparse.tamu.edu per run). **Source** comes either from a `git clone` of the
public repo (for pushed refs) or — the default — a `git archive HEAD`
snapshot tarball uploaded to the `linprogx-src` Volume keyed by sha, which
lets the harness benchmark local commits that aren't on GitHub yet.

```bash
# one-time setup (idempotent)
uvx modal run tools/modal_bench.py --action upload-fixtures
uvx modal run tools/modal_bench.py --action upload-src   # archives worktree HEAD

# smoke test
uvx modal run tools/modal_bench.py --action bench --mode paired \
    --ref <sha> --instances lp_woodw --pairs 3

# full 24-instance single-shot suite (rows match experiments/suite_bench.py shape,
# so results can feed assets/campaign.db)
uvx modal run tools/modal_bench.py --action bench --mode suite --ref <sha>

# certified knife-edge paired verdicts (interleaved lx/HiGHS, median/min/wins)
uvx modal run tools/modal_bench.py --action bench --mode paired --ref <sha> \
    --instances lp_degen3,lp_osa_14,lp_stocfor3,lp_80bau3b,lp_cre_a,lp_greenbea,lp_cre_b \
    --pairs 7
```

Output JSON (`{ref, machine_info, load_checks, rows|paired}`) is printed to
stdout and saved to `/tmp/modal_bench_<ref>_<mode>.json`. `machine_info`
records the CPU model/count, memory, Modal region/cloud, and start/end
loadavg so every run carries its own load-honesty check. Per-cell subprocess
timeout is 200s.

First validated run (2026-07-13, ref `7e9947a`; suite on GCP
`europe-west1`, paired on Azure `westus3`, loadavg 0.00 throughout):
24/24 linprogx coverage, geomean lx/HiGHS ratio **0.735** (local quiet
reference ~0.78); most per-instance ratios agreed with local within
~±30%. Material shifts: `greenbea` 4.8x vs 10.1x local (clean box much
kinder — local load was inflating the loss), `fit2p` 0.14 vs 0.07 (lx
advantage halved but still ~7x faster), `ken_07` 0.39 vs 0.64 (sub-100ms
walls, noise-prone). Paired knife-edge verdicts flipped two losses into
wins on the clean box: `osa_14` 0.96 (WIN 6/7) and `cre_a` 0.97 (WIN 5/7);
`degen3`/`stocfor3` sit at 1.06, `80bau3b` 1.20, `cre_b` 2.75,
`greenbea` 5.25. Total cost of the full validation (uploads + smoke +
24-instance suite + 7x7 paired): ~22 container-minutes on 4 CPU / 8 GiB,
roughly $0.25-0.40.

A second validated run followed the Dantzig route and presolve V2 ships — see
[Clean-box certification (2026-07-14)](#clean-box-certification-2026-07-14)
above for the full numbers (geomean 0.735 → 0.558, `cre_b` flips to a
certified win, hard-loss ladder collapses to nothing above 2×).
