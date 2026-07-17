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
  reproduced in the replay: every one of the 23 replayed commits solves all 24 to
  certified optimality.
- **Runtime: aggregate EXCEEDED, per-instance majority WON.** The suite total and
  geometric-mean time ratio have favored linprogx since early in the campaign; the
  paired head-to-head is now **16W-2P-6L**, including the `qap15` coverage win,
  on the 2026-07-16 AWS `us-west-2` protocol-v3 board. The hard-loss ladder has
  collapsed below 1.7×, with two genuine coin flips at parity (see
  [Current certified scoreboard](#current-certified-scoreboard)).

## The arc

The campaign narrative (fully dated in `docs/HANDOFF.md`) runs from a **14-10**
paired head-to-head at the session-start baseline (`a1a355d`, 2026-07-04) through
twenty substantive ship commits to presolve V2 shipping on 2026-07-14, followed
by the setup fast path, native presolve port, and protocol-v3 certification wave
on 2026-07-15 and 2026-07-16. The through-line:
the IPM factor path, dual-simplex LU path, presolve layer, and measurement
protocol were each tightened under paired certification, closing whole classes
of hypotheses along the way. The final pre-V2 ships came out of a joint
Claude/Codex strategy round that killed three more falsifier probes before
finding its two winners (Dantzig rescue-route leaving and presolve V2); the
post-V2 wave then focused on setup cost, native presolve porting, and
host-pinned certification.

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
21. **`26a9359` — Cholesky setup fast path.** A bucketed min-degree queue,
    exact preallocation, and fused compaction preserve the ordering and factor
    structure exactly. Setup fell 46% on `cre_d`, 31% on `maros_r7`, 11% on
    `degen3`, and 7% on `cre_a`.
22. **`82cd31d` — native presolve V2 hot path.** The native port removed the
    Python-side reduction bottleneck. Its first same-host paired certification
    put `maros_r7` at 0.733 (9/9), `stocfor3` at 0.854 (9/9), and `cre_a` at
    0.896 (6/9), while `woodw` moved from a 1.60 loss to 1.022 parity.

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
- **AMD-style approximate degree:** two outcome-gated attempts failed to move
  `cre_a`. The corrected Amestoy-Davis-Duff bound improved the `degen3` and
  `cre_d` ordering slices by 21% and 24%, but cut `cre_a` only 9.6% while adding
  7.1% factor flops. The target instance was the one the approximation hurt.
- **Ruiz pass-count reduction:** every board IPM instance uses all 10 passes.
  A 0.05 early-exit tolerance cut `cre_a` 3.7%, changed objectives, and regressed
  `cre_d` 3.4%; 0.01 did not help (1.018). Equilibration remains numerics-active.
- **Certificate-evaluation windowing:** the measured ceiling on `cre_a` was
  0.2–1.2 ms against a roughly 2.7 ms (3%) bar. The windowed attempt measured
  1.0034, so the lever closed on ceiling.
- **Dual-simplex dense-U FTRAN:** the candidate showed three distinct bandwidth
  regimes: 16% faster under three-worker local contention, 1.8% faster on the
  Modal host class (`greenbea` 0.982, 18/21), and no gain on a quiet box.
  `woodw` measured 0.999, `stocfor3` 1.002, and `80bau3b` 0.989 with a
  0.946–1.127 host spread. The on-host result missed the 5% bar and closed the
  dense-U path.

### Protocol v3 makes host luck visible

Protocol v3 runs the paired benchmark concurrently on three AWS `us-west-2`
hosts, seven interleaved pairs per host, and scores the median host ratio. The
artifact keeps each host's spread and all 21 pair outcomes. A region pin alone
was insufficient because Modal can place two `us-west-2` runs on different host
generations; bandwidth-sensitive verdicts were still moving with the hardware
lottery.

The first v3 certification settled the bandwidth-sensitive set. `pilot87`
became a certified win at **0.826**, with a host range of 0.813–0.939 and
**21/21** pair wins. `pds_20` held its win at **0.824** (20/21). The four losses
in that wave were stable on every host: `greenbea` 1.695, `osa_14` 1.424,
`osa_60` 1.290, and `pds_10` 1.258.

The knife-edge certification then repriced the old pin4 parity band. `woodw`
moved from 0.996 parity to **1.201** (3/21), and `80bau3b` moved from 1.010
parity to **1.198** (2/21). Their pin4 results were host luck. `cre_a` at
**1.002** and `stocfor3` at **0.999**, both with 12/21 wins, are the two true
coin flips. The resulting board of record is **16W-2P-6L**.

## Current certified scoreboard

The **certified** standing uses the protocol-v3 median-of-hosts method in
`docs/HANDOFF.md` and the Modal artifacts replayed into `assets/campaign.db`;
the single-shot replay table below is narrative-grade only. The board of record
combines the stable prior cells with the two 2026-07-16 AWS `us-west-2` v3
certifications (three hosts × seven pairs):

- **V3 board of record: 16W-2P-6L**, including `qap15` as a coverage win.
- **Confirmed wins:** `pilot87` 0.826 (21/21) and `pds_20` 0.824 (20/21).
- **True coin flips:** `cre_a` 1.002 (12/21) and `stocfor3` 0.999 (12/21).
- **Losses:** `greenbea` 1.69, `osa_14` 1.42, `osa_60` 1.29, `pds_10` 1.26,
  `woodw` 1.20, and `80bau3b` 1.20.

The v3 board supersedes the single-host pin4 board. The important certification
waypoints since presolve V2 shipped:

- **Clean-box certification at `1f4351d` (2026-07-14, AWS `us-west-2`,
  `assets/modal_bench_1f4351dcfa96_{suite,paired}.json`):** 13W-11L; geomean
  0.558; aggregate 49.1s vs HiGHS 192.8s plus the `qap15` timeout; coverage
  24/24 vs 23/24. `cre_b` flipped from LOSS 2.75× to WIN 0.940 (6/7), and the
  loss ladder collapsed to nothing above 2×.
- **Host-conditional margins (2026-07-16,
  `assets/knife_chunk{A,B}.json`):** the knife-edge verdicts depend on host
  class. On Azure-Asia, `pilot87`, `degen3`, `stocfor3`, and `pds_20` became
  clear losses, while `cre_a` remained the true knife-edge at 1.036 median and
  0.969 min-vs-min. This established the doctrine that certifications must pin
  cloud and region.
- **Setup fast-path ship and pinned-region certification (2026-07-16, AWS
  `us-west-2`, `assets/modal_bench_99ce9c9fd693_{suite,paired}.json`):**
  bucketed min-degree queue, exact preallocation, and fused compaction improved
  setup while preserving exact output. The pinned-region certification reached
  15W-9L, flipping `pilot87` and `pds_20`, with `degen3`, `cre_a`, and
  `stocfor3` at parity.
- **Post-native-port paired certification (2026-07-16,
  `assets/modal_bench_82cd31d060d2_paired.json`):** `maros_r7` flipped to WIN
  9/9 (0.733), `stocfor3` to WIN 9/9 (0.854), and `cre_a` to WIN 6/9 (0.896).
  `woodw` moved to parity at 1.022, while `cre_d` and `80bau3b` narrowed but
  remained losses on that run.
- **Pin4 board at the `957347b`-era build (2026-07-16, AWS `us-west-2`,
  `assets/pin4_chunk{1,2}.json`):** 14W-5L-4P plus `qap15` coverage, for 15
  wins. The OSA swing was
  investigated and not attributed to a code regression; Modal still does not
  expose instance-type pinning. This board remains a waypoint, but its
  `woodw`/`80bau3b` parity calls did not survive multi-host measurement.
- **Protocol-v3 first certification (`c344177`, 2026-07-16, AWS `us-west-2`,
  `assets/modal_bench_c34417761bb6_paired_hosts3.json`):** `pilot87` won at
  0.826 (21/21), `pds_20` at 0.824 (20/21), and the four bandwidth-sensitive
  losses remained losses on all three hosts.
- **V3 knife-edge certification (`b656ef3`, 2026-07-16, AWS `us-west-2`,
  `assets/modal_bench_b656ef3f8915_paired_hosts3.json`):** `woodw` and
  `80bau3b` repriced to roughly 1.20 losses; `cre_a` and `stocfor3` settled as
  12/21 coin flips. This produced the 16W-2P-6L board of record.
- **Dense-U on-host envab (`bda0579`, 2026-07-16, AWS `us-west-2`,
  `assets/modal_bench_bda057900a4d_envab_hosts3.json`):** `greenbea` improved
  only 1.8% on the scoring host class, below the 5% ship bar. The artifact is
  stored as envab metadata and A/B evidence, not as linprogx/HiGHS paired rows.

## Headline per-instance trajectories

Single-shot replay wall-seconds, **baseline `a1a355d` → current `82cd31d`**, against
the single-shot HiGHS reference on the same (loaded) box. Ordered by final ratio
(wins first). **These are narrative-grade, not certification-grade** — see the
noise note below.

| instance | baseline (s) | current (s) | HiGHS ref (s) | ratio | route | W/L |
|---|---:|---:|---:|---:|---|:--:|
| qap12 | 1.74 | 1.53 | 102.98 | 0.01 | pdhg | **WIN** |
| truss | 0.13 | 0.10 | 2.75 | 0.04 | ipm | **WIN** |
| fit2p | 0.09 | 0.09 | 1.45 | 0.06 | ipm | **WIN** |
| d2q06c | 0.39 | 0.29 | 0.88 | 0.33 | ipm | **WIN** |
| ken_13 | 0.67 | 0.49 | 0.96 | 0.51 | ipm | **WIN** |
| ken_07 | 0.02 | 0.02 | 0.04 | 0.53 | ipm | **WIN** |
| cre_b | 5.98 | 1.05 | 1.94 | 0.54 | ipm | **WIN** |
| ken_11 | 0.24 | 0.17 | 0.30 | 0.57 | ipm | **WIN** |
| ken_18 | 8.99 | 5.47 | 8.56 | 0.64 | ipm | **WIN** |
| maros_r7 | 2.39 | 0.64 | 1.01 | 0.64 | ipm | **WIN** |
| degen3 | 0.21 | 0.17 | 0.22 | 0.75 | ipm | **WIN** |
| osa_30 | 2.10 | 3.85 | 4.23 | 0.91 | ipm | **WIN** |
| pilot87 | 4.09 | 3.31 | 3.63 | 0.91 | ipm | **WIN** |
| 80bau3b | 0.33 | 0.17 | 0.18 | 0.93 | ipm | **WIN** |
| cre_d | 5.38 | 1.01 | 1.09 | 0.93 | ipm | **WIN** |
| osa_60 | 7.79 | 18.58 | 19.13 | 0.97 | ipm | **WIN** |
| stocfor3 | 0.90 | 0.60 | 0.61 | 0.98 | ipm | **WIN** |
| cre_a | 0.13 | 0.10 | 0.09 | 1.08 | ipm | loss |
| pds_20 | 20.98 | 12.57 | 10.72 | 1.17 | pdhg | loss |
| woodw | 0.18 | 0.11 | 0.09 | 1.21 | ipm | loss |
| osa_14 | 1.10 | 1.53 | 1.09 | 1.40 | ipm | loss |
| pds_10 | 3.71 | 2.20 | 1.45 | 1.52 | pdhg | loss |
| greenbea | 3.87 | 0.43 | 0.27 | 1.63 | dual-simplex | loss |
| qap15 | 0.87 | 0.67 | TIMEOUT | n/a | pdhg | **WIN** |

The clean structural wins survive the noise: **maros_r7 2.39 → 0.64**,
**greenbea 3.87 → 0.43**, and **cre_b 5.98 → 1.05 / cre_d 5.38 → 1.01**.
The `current` column now includes the backfilled `26a9359` setup-fast-path and
`82cd31d` native-presolve rows, bringing the trajectory to 23 commits. These
single-shot ratios still do not set the board; the v3 results above do.

### Measurement note (read before trusting a single cell)

Every number in the table and DB is a **single-shot replay under variable machine
load** (the box ran at load average 6–16 throughout, with other benchmark work
active). This is deliberately **narrative-grade**: it reconstructs the shape of the
improvement arc, not a certifiable head-to-head. The campaign's **certification**
protocol is different: three hosts, seven interleaved pairs per host, scored by
the median host ratio and recorded in `docs/HANDOFF.md`. Where a single-shot cell disagrees with the paired
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
