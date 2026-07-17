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
  paired head-to-head is now **20W-1P-3L**, including the `qap15` coverage win,
  on the 2026-07-17 AWS `us-west-2` protocol-v3 aggregation-era board. A loss census
  plus two presolve ships (H0, H1) took the board to 20W-0P-4L, and a native
  equality-row aggregation ship then flipped `80bau3b` to a win while reclassifying
  `cre_a` to an honest coin flip — dropping the loss column to three and opening a
  one-cell parity column. The three remaining losses are greenbea 1.74, pds_10
  1.26–1.57 (host-dependent), and woodw 1.20; the single parity cell is cre_a
  (see [Current certified scoreboard](#current-certified-scoreboard)).

## The arc

The campaign narrative (fully dated in `docs/HANDOFF.md`) runs from a **14-10**
paired head-to-head at the session-start baseline (`a1a355d`, 2026-07-04) through
twenty substantive ship commits to presolve V2 shipping on 2026-07-14, followed
by the setup fast path, native presolve port, and protocol-v3 certification wave
on 2026-07-15 and 2026-07-16, a loss-census-driven presolve wave
(H0's O(nnz) row-build fix and H1's fixpoint re-stage) that flipped four cells to
a **20W-0P-4L** board on 2026-07-17, and finally a native equality-row aggregation
ship the same day that flipped `80bau3b` to a win and reclassified `cre_a` to a
coin flip, settling the **20W-1P-3L** aggregation-era board. The through-line:
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

### The census wave: four flips to 20W-0P-4L

With the pivoting and IPM-factor levers exhausted, the endgame turned to a
**loss census** — an eight-cell phase attribution of every HiGHS-vs-linprogx
loss, decomposing each into its presolve / factor / iteration / route slices and
ranking six falsifiable hypotheses by projected work saved. The census did not
propose new algorithms; it read where the wall actually went. Its buried lede:
current presolve on the two `osa` instances yielded **zero reductions** yet cost
58% and 82% of the public wall. That is not a tuning knob — it is a bug smell.

**H0 — the quadratic presolve row-build war story.** The zero-yield osa presolve
overhead was an accidental **O(degree²)** loop. The classic presolve row build
called a row-set helper (`ps_row_set` → linear `ps_row_find`) once per nonzero,
so each dense border row re-scanned its own growing set. On osa's border rows
(degree 38k and 173k) that detonated. The signature was unmistakably quadratic:
from the two degrees the predicted cost ratio was **20.45**, and the measured
ratio was **20.9** — a quadratic fingerprint matched to two significant figures.
A generation-stamp dedup makes the build **O(nnz)**: presolve wall `osa_14`
1050 ms → 9 ms (−98.3%), `osa_60` 21,916 ms → 66 ms (−99.5%), with
**bit-identical reduced problems on all 24 fixtures** (fingerprinted, and
independently re-verified). It is strictly better than a skip-gate because it
also cheapens presolve wherever it fires. Shipped as `d727389`.

**H1 — the fixpoint re-stage and its acceptance-gate lesson.** The census also
found that re-running the semantic V2 gate after the classic cascade reaches a
**second fixpoint** with measured gains. H1 composes that: a classic pass making
≥2% progress triggers a second V2 fixpoint on the rebuilt reduced problem;
second reductions under 2% of the reduced shape are **discarded** (byte-identical
off-path). Iterations drop deterministically — `cre_a` 36 → 34, `80bau3b`
62 → 47 — and `stocfor3` holds its iteration count at −12% nnz. The lesson is in
the gate itself: naive re-staging **regressed** `pds_10` −41% (PDHG 8576 → 10688
iters) and `d2q06c` −34% via conditioning perturbation from tiny reductions —
**the acceptance gate is load-bearing, not hygiene**. An in-C single-call variant
reached an inferior order-dependent fixpoint (`cre_a` 36 → 38) and was reverted:
compose-rebuild-then-rerun is the correct shape. 57 new characterization tests.
Shipped as `928399c`.

**The four flips.** The H0+H1 certification wave (v3, three us-west-2 hosts ×
seven pairs, at `928399c`,
`assets/modal_bench_928399cf5fea_paired_hosts3.json`) flipped four cells off the
prior board:

- `osa_60` **1.29 → 0.280** [0.253, 0.283], 21/21 — the quadratic-build fix
  makes linprogx **3.6× faster than HiGHS**.
- `osa_14` **1.42 → 0.912** [0.819, 1.018], 17/21.
- `cre_a` **1.002 → 0.939** [0.917, 0.948], 18/21 — H1's 36 → 34 iterations
  lands the old coin flip.
- `stocfor3` **0.999 → 0.962** [0.935, 0.973], 17/21.

`80bau3b` narrowed but did not flip: **1.062** [0.951, 1.063], 7/21 — H1's +26%
local gain shrank to ~11% on-host because the bandwidth-heavy refactor slice
damps presolve gains there. The sentinels stayed clean: `greenbea` 1.69
unchanged, and `pds_10` printed 1.569 vs a prior 1.258 but with **8576
iterations in every pair of both waves** and flat HiGHS walls — a pure
host-hardware swing on the PDHG side, not a regression. The parity column is now
empty: **20W-0P-4L**.

**Two architectural kills scoped the endgame.** The census's remaining big-loss
hypotheses both died at the same boundary. `pds_10`'s 38,852 degree-2 unit
columns contain **zero free columns** (31,999 are `[0,inf)` arcs, 6,853
capacitated); exact contraction in the eq-box form is limited to the 1,342 with
a provably redundant bound (−3.2% work vs a 25% gate). The census's projected
−30.5% was HiGHS's *realized* shape, reachable only because a contracted
capacitated arc becomes a **ranged row** — a constraint form the SparseSolver /
PDHG architecture cannot express. `greenbea` hits the same wall from the other
side: it is already at a **bound-propagation fixpoint**, so eliminating all 338
bounded singletons yields only 10 redundant rows (against a ~574–1000 bar), zero
tightenings, zero fixings, and the eq-box relabel makes Dantzig *worse*
(+73% pivots). HiGHS's 574-row greenbea reduction therefore does **not** come
from bounded-singleton elimination; the cause is unknown and under measurement
(a presolve-log rule-count diff). Ranged-row support end-to-end is the single
structural unit behind both remaining big losses — an architecture project, not
a presolve probe, if the ceiling is judged worth it.

### The aggregation arc: one rule family, a refuted thesis, and one real win

The census had named greenbea's 574-row HiGHS reduction as an unknown cause. A
HiGHS 1.14.0 `presolve_rule_off` **ablation** found it: **one rule family** —
the Aggregator (rule 12) plus free-column substitution (rule 8), i.e. general
equality-row aggregation, the *k>2 generalization of our doubleton* — accounts
for the entire presolve deficit on **three of the four remaining losses**.
Disabling it lands HiGHS on our shapes within 0–5 rows: greenbea 951 → 1521 (vs
our 1525), woodw 557 → 707 (vs our 707 **exactly**), 80bau3b 1537 → 1997 (vs our
1992). We already remove *more* forcing rows than HiGHS (351 vs 190); what we
lack is the substitution that consumes rows first.

**The transferability lesson (shape parity is not pivot parity).** Building the
aggregation was straightforward and correct — it hit every ablation shape target
with oracle equivalence everywhere (greenbea 936 rows < HiGHS 951; 80bau3b 1569 ~
1537; woodw 0 aggregations, its singletons genuinely not implied-free). But the
performance thesis was **refuted for our solver**: our Dantzig dual simplex does
*more* pivots on the aggregated greenbea. The fill-guard frontier peaked at −7%
pivots (FILL=15 at 1234 rows) and hit +24% at the 936-row target; **no setting
achieved rows<1000 AND pivots<3520**. HiGHS's 2,836-pivot behavior on that shape
belongs to *its* pricing, not to the shape — a projection from another solver's
realized behavior is not transferable. greenbea's presolve frontier is therefore
**closed**; its remaining gap is pricing-side (HiGHS-class dual steepest edge,
published literature, paper-only). The live residue was **80bau3b**: there the
aggregation is fill-*negative* (nnz 21798 → 21511) with IPM iters 47 → 43, and
the cell needed only ~6% — blocked purely by the Python pass cost.

**The native port economics.** The pure-Python pass could not net the win: even
driven from 465ms to 42ms (11×), its floor was ~6ms build + ~30ms scan against a
~4ms budget. The C port closed it: general equality-row aggregation as
`PS_REC_AGGREGATION` (tag 5), **bit-identical on all 24 fixtures**, **2.23ms
accept / sub-2ms rejects** (Python was **465ms**), RSS-flat over 1200 iterations.
Shipped default-**on** at `54e9232`, **double-gated**:

- **Fill-non-positive** (structural): the aggregation is accepted only where it
  cannot grow the working matrix — on the board that is 80bau3b, d2q06c, ken_07.
- **IPM-route-only** (the status-semantics save): aggregated shapes raise DS
  pivots (per the transferability lesson) *and* can push PDHG past convergence.
  The worker found the `cycle` benchmark going **optimal → iteration_limit** when
  fill-gate-accepted — a status regression. The route gate fixes it, and the
  `cycle` test now runs **unpinned** against the shipped default.

Local A/B: **80bau3b −6.8%** (IPM 47 → 44), **d2q06c −19.7%**, **ken_07 −7.7%**;
`cre_a` pays a ~3% reject scan (no cheap discriminator exists — fill-trajectory
minima overlap between accepts and rejects). `just ci` fully green (coverage
88.87%); fresh PYSEC advisories were remediated en route (pillow 12.3.0,
setuptools 83.0.0, advancing the exclude-newer pin 06-20 → 07-10; both releases
13–16 days old, the machine's 7-day gate still applies) at `8697483`.

**The certification (`70203c4`, v3, three us-west-2 hosts × seven pairs).**

- `80bau3b` **1.062 → 0.881** [0.840, 0.948], 20/21 — the native aggregation flip.
- `d2q06c` **0.371** (21/21) and `ken_07` **0.410** (21/21), both deepened.
- `greenbea` sentinel **1.741**: iters 4399 identical, our wall +1%, HiGHS −2.5%
  — host drift, clean.
- `cre_a` **1.021** [1.010, 1.052], 7/21 — but decomposed against the prior 0.939
  wave via **bit-identical iterations** (34 both waves): our side +4ms (~the
  2.75ms reject scan), HiGHS side −6ms (host luck). The scan's ~2% is real on a
  cell whose true margin is ±3% around parity, so `cre_a` is **honestly a coin
  flip** (0.939 and 1.021 across the two waves) and is **scored as parity**, not
  as the census-wave win.

The flip and the reclassification net to **20W-1P-3L**: the loss column drops to
three (greenbea, pds_10, woodw) and a one-cell parity column (cre_a) opens.

## Current certified scoreboard

The **certified** standing uses the protocol-v3 median-of-hosts method in
`docs/HANDOFF.md` and the Modal artifacts replayed into `assets/campaign.db`;
the single-shot replay table below is narrative-grade only. The board of record
combines the stable prior cells with the 2026-07-17 AWS `us-west-2`
aggregation-era certification (three hosts × seven pairs), which re-certifies the
five instances the native equality-row aggregation touched, layered over the
census-wave and prior v3 certifications:

- **Aggregation-era board of record: 20W-1P-3L**, including `qap15` as a coverage win.
- **The aggregation flip:** `80bau3b` 1.062 → **0.881** [0.840, 0.948] (20/21),
  the native equality-row aggregation win — with `d2q06c` **0.371** (21/21) and
  `ken_07` **0.410** (21/21) deepened on the same wave.
- **Parity (1):** `cre_a` is honestly a coin flip — **0.939** and **1.021**
  across the two waves. Decomposed by bit-identical iterations (34 both waves):
  our +4ms (~the 2.75ms reject scan) vs HiGHS's −6ms of host luck, on a cell
  whose true margin is ±3% around parity. Scored as parity, not the census win.
- **Carried-over flips and wins:** `osa_60` **0.280** (21/21), `osa_14`
  **0.912** (17/21), and `stocfor3` **0.962** (17/21) from the census wave;
  `pilot87` 0.826 (21/21) and `pds_20` 0.824 (20/21) from the first v3 wave;
  plus the stable structural-win core (qap12, ken_18, maros_r7, cre_b, cre_d, …).
- **Losses (3):** `greenbea` 1.74 (sentinel: 4399 iters identical, our wall +1%,
  HiGHS −2.5% host drift), `pds_10` 1.26–1.57 (host-dependent PDHG swing;
  iterations flat across pairs), and `woodw` 1.20.
- **Backfill pending:** the per-commit replay trajectory in `assets/campaign.db`
  does not yet include rows for the ship commits `d727389` (H0), `928399c` (H1),
  or `54e9232` (native aggregation); the board is certified from the Modal
  artifacts, and the single-shot trajectory table below still ends at `82cd31d`.

The aggregation-era board supersedes the 20W-0P-4L census-wave board, which
superseded the 16W-2P-6L v3 board, which in turn superseded the single-host pin4
board. The important certification waypoints since presolve V2 shipped:

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
- **H0+H1 census-wave certification (`928399c`, 2026-07-17, AWS `us-west-2`,
  `assets/modal_bench_928399cf5fea_paired_hosts3.json`):** the four flips —
  `osa_60` 0.280 (21/21), `osa_14` 0.912 (17/21), `cre_a` 0.939 (18/21),
  `stocfor3` 0.962 (17/21) — took the board to **20W-0P-4L**. `80bau3b` narrowed
  to 1.062 (7/21) without flipping; `greenbea` 1.69 and `pds_10` (host-dependent
  PDHG swing) held as the remaining structural losses alongside `woodw` 1.20.
- **Native aggregation certification (`70203c4`, 2026-07-17, AWS `us-west-2`,
  `assets/modal_bench_70203c413cea_paired_hosts3.json`):** `80bau3b` flipped
  **1.062 → 0.881** [0.840, 0.948] (20/21) on the native equality-row
  aggregation ship, with `d2q06c` 0.371 and `ken_07` 0.410 (both 21/21)
  deepened. `cre_a` printed 1.021 [1.010, 1.052] (7/21) and — decomposed against
  the 0.939 wave via 34 bit-identical iterations both times — was reclassified to
  an honest coin flip and scored as parity. `greenbea` sentinel 1.741 held clean
  (iters identical, host drift only). The board settled at **20W-1P-3L**.

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
