# Session Handoff — linprogx campaign orchestration (2026-07-21)

You are the new overmind. This document + the dated research ledger
(docs/HANDOFF.md — read every entry from "PRESOLVE V2 SHIPPED" onward;
the 2026-07-17..21 entries are the current era) are your complete state.
Fable plans/reviews/synthesizes; workers implement. Everything below is
battle-tested; deviate knowingly or not at all.

## 1. Mission and board

- GOAL (Evan's standing /goal): beat HiGHS head-to-head on ALL 24
  LPnetlib instances. Constraints (inviolable): never read any solver's
  source code (papers/textbooks fine); no per-problem tuning (global
  mechanisms/thresholds only); never loosen eps=2e-5; certificate-backed
  optimality only; honest reporting.
- BOARD OF RECORD: **23W-0P-1L** (protocol v3: 3 hosts x 7 interleaved
  pairs, median-of-hosts, Modal AWS us-west-2; cumulative host-
  conditional accounting per the pilot87 precedent). Sole loss:
  **greenbea 1.215** [1.208,1.235] — from 14x at campaign origin, 1.69
  at this run's start.
- Scoring doctrine: paired interleaved only; v3 median-of-hosts is the
  scoreboard; single-shot/single-host is narrative-grade; knife-edge
  cells use cumulative cross-wave accounting (majority + median<1).
- Work happens in /home/evan/dev/linprogx-perf-worktree, branch
  perf-supernodal-simplex; backup tag session-backup-20260702 tracks
  HEAD (re-point after every commit). The main checkout
  /home/evan/dev/linprogx belongs to another session — NEVER touch it.

## 2. Reference index (read on demand, not all up front)

Ledger + protocol:
- docs/HANDOFF.md — THE research ledger: ~50 dated verdicts, every
  ship/kill/doctrine. The single source of truth.
- docs/CAMPAIGN.md + docs/campaign_report.html — public chronicle
  (gh-pages branch mirrors it; Claude artifact
  https://claude.ai/code/artifact/a1eba80f-2b5d-426f-8527-a2d7f4545d3e —
  republish with the Artifact tool passing url=).
- tools/modal_bench.py — bench harness: --mode paired|envab, --hosts N
  (protocol v3), upload-src (keyed by exact sha; REQUIRED after every
  commit before benching), pinned cloud=aws region=us-west-2.
- tools/replay_bench.py (artifact ingestion; ARTIFACT_DATES/LABELS maps)
  + tools/build_report_data.py (report regeneration) — the chronicle
  pipeline. assets/campaign.db is gitignored, rebuilt by ingest.

greenbea science (chronological spine; all under experiments/):
- greenbea_dossier_2026_07_18.md — evidence pack + closed axes (note
  the CORRECTED 61.6% active figure).
- greenbea_ideas_{gpt5,codex-contrarian,claude-opus,glm-5.2}_2026_07_18
  .md + greenbea_research_plan_2026_07_18.md — the 4-model ideation
  fan-out + synthesis.
- probe_{activeset,tomography,blockds,precision,locality,schur}_*.md —
  the probe-wave verdicts (tomography = the dual-Phase-1
  identification, four corroborating lines).
- dual_phase1_derivation_2026_07_18.md + phase1_predictions_2026_07_18
  .md — the orchestrator's independent Fenchel derivation (P1 confirmed
  EXACTLY: HiGHS DuPh1 is b-invariant; B* dual-feasibility proven by
  direct linear algebra; 3,334 native pivots) and the conservation law
  (pivots x us/pivot ~0.38-0.40s across every start ever built).
- kernel_campaign_dossier_2026_07_19.md + kernel_campaign_angles_*.md +
  the k1..k12/lsa/lsb/int reports — the 12-angle fan-out: K1 IPC census
  (solves at IPC 0.30-0.60), the hardware-floor kills (k3 dense sweeps
  75-87x slower; k12 flags 0%; lsb interleaving — 0.000% collisions yet
  slower, memory disambiguation; lsa level scheduling — perfect DAG,
  overhead swamps at 1.5k rows), and int_kernel_combined (THE SHIPPED
  SIMD UNIT: K4 branchless Harris + K2-safe AVX2 scan; -11.3% local;
  certified 2026-07-20: greenbea 1.69->1.215, cre_a CERTIFIED 0.912
  19/21 five-wave-cumulative, woodw 0.789).
- creative_attack_dossier_2026_07_21.md + creative_attack_angles_*.md —
  the CURRENT C-wave (section 5).
- Supporting: greenbea_ipm_stall_2026_07_18.md (dual-certificate stall
  anatomy + NaN guard origin), greenbea_warmstart_2026_07_18.md
  (crossover kills; its cheap partial-IPM timings were an ARTIFACT —
  see probe_activeset), greenbea_pivot_gap_2026_07_17.md (the decisive
  crosses + basis transfer), k7 (2,399-pivot native-basis discovery),
  k9 (density shaping LIVE, pipeline-blocked), k8 (boxed-fixing
  vindicated).

Other board cells (all won; context if ever re-certifying):
- pds arc: pds_mechanism_2026_07_17.md (HiGHS presolve makes HiGHS
  3.16x SLOWER on pds; its DS is the weapon), rr_falsifier_2026_07_17
  .md (series-chain premise vacuous), the clean-room netagg (multi-row
  implied-bound intersection — after the NETAGG INTEGRITY INCIDENT and
  quarantine; the clean room BEAT the tainted unit) +
  pds_parallel_cols_2026_07_18.md (proof-carrying endpoint dominance).
  pds_10 certified 0.893/cumulative 0.939; pds_20 0.499.
- ipm_slice_census_2026_07_17.md + the a2 ship (single-thread
  cache-sized dense-tail dpotrf; LINPROGX_CHOL_SCHED) — woodw's flip.
- loss_census_2026_07_16.md — the census that started the H0/H1 wave
  (quadratic presolve row-build fix -> osa_60 0.280; fixpoint re-stage
  -> cre_a/stocfor3 flips).

## 3. Orchestration protocol (hard-won; do not relearn)

Backends (~/.claude/skills/overmind/backends/*.md; run
~/.claude/skills/overmind/bin/usage-check.sh BEFORE any fan-out):
- codex: codex-worker.sh run -C <worktree> --label X - (brief on stdin
  with '-' for run AND cont; "$(cat -)" in zsh pipelines HANGS
  silently). gpt-5.5 high-reasoning is the working profile. Evan wants
  gpt-5.6 ("always use 5.6") but it API-400s on this plan ("not
  supported when using Codex with a ChatGPT account") — last probed
  2026-07-21; retry periodically, surface, never silently fall back
  (memory: codex-model-choice.md).
- claude: Agent tool, ALWAYS explicit non-Fable model (opus for hard
  work). NEVER SendMessage-resume a subagent from a Fable session
  (resume drops the model pin -> bills Fable). Fresh spawns only.
- opencode/GLM-5.2: opencode-worker.sh with WORKER_DIR env; metered
  dollars; dispatch parallel workers as SEPARATE harness Bash calls
  (nested &/wait children hang); workers can zombie at "build" for
  24h+ — hunt with `ps -eo pid,etime,args | grep 'opencode run'`, kill
  by pid. GLM/probe worktree venvs: `uv sync` REMOVES ad-hoc installs
  (highspy!) — pre-install yourself and forbid plain uv sync in briefs.
- Fable is NEVER a worker (billing hard constraint).

Worker briefs: GOAL/CONTEXT/CONSTRAINTS/DONE WHEN/VERIFY; falsifier-
first with explicit kill criteria; "no git ops"; rebuild before test:
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation,
then UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e .
--no-build-isolation after C changes (sync alone does NOT rebuild the
extension). Characterization-first on AGENTS.md high-risk areas.

INTEGRITY PROTOCOL (born 2026-07-17: a gate-passing unit was QUARANTINED
for downloading HiGHS source — NETAGG INTEGRITY INCIDENT in the ledger):
every brief prohibits network outright; AUDIT every worker log before
trusting a ship: LOG=$(codex-worker.sh log <session>); grep -cE 'curl -|
wget |git clone|https://github' "$LOG" — repo files containing 'curl'
(download_lpsuite.sh, modal_bench.py) give false positives; inspect
hits in context. Network use on a design-informable unit = discard +
clean-room re-derivation.

Ship discipline: review the DIFF (not the summary); rerun VERIFY
yourself; `just ci` for substantial changes (its pip-audit needs
network — workers must skip it, you run it; the repo pins exclude-newer
in uv.toml — advance only within the machine's 7-day release-age rule);
commit staged paths only (never add -A); tag -f session-backup-20260702;
dated ledger entry (ship or kill — every verdict); upload-src; v3 cert;
chronicle on board moves (worker pattern -> verify embed yourself ->
commit -> gh-pages via a SCRATCH worktree (never checkout in the perf
worktree; scratchpad path works) -> Artifact republish with url= ->
nanobot notify Evan).

Git/exec quirks: this checkout maps LF->CRLF on touch — NEVER git stash
(stash/pop conflicts on churned files; use a scratch worktree at HEAD
to test pre-existing failures). `cmd | tail` swallows exit codes — a
red pytest once slipped into a commit; check pytest's own status.
Probe worktrees: one per worker (git worktree add
/home/evan/dev/linprogx-<slug> -b work/<slug>), remove + branch -D
after banking. Modal runs cost ~$0.20-1/container; never include
lp_qap15 in PAIRED certs (HiGHS times out 300s/cell — 105 wasted
minutes; it is a coverage-only cell).

Measurement doctrines: LOADED-BOX (ship-gate A/Bs need a quiet box — a
16% "gain" once evaporated when the box went quiet); PROJECTION (never
size a unit from another solver's realized behavior — shape parity is
not pivot parity, proven bidirectionally); disputes "regression vs host
lottery" are settled by iteration/pivot counts (bit-identical iters +
flat HiGHS walls = host); alternating A/B median-of-9 is the local
standard; cumulative cross-wave accounting for knife-edges.

## 4. greenbea: complete science (the only open cell)

Local ~0.37s (post-SIMD) vs HiGHS 0.24s; on-host 1.215. Flip needs
~-18%. PROVEN WALLS: the solves (37% of wall) are memory-latency-bound
at IPC 0.30-0.60 — immune to dense sweeps of sparse storage,
flags/vector width, software pipelining (disambiguation-bound), and
level scheduling (overhead swamps at 1.5k rows). Pricing rules (5),
starting bases (foreign + native crossovers), presolve depth (all
families), route changes (IPM stalls on a pinned dual certificate;
PDHG uncompetitive; dense tableau and sparse revised-primal routes killed),
and perturbation (C4) all
carry dated kills. The pipeline route (2,399-pivot native B* + K9
shaping) is dead at BOTH ends: exact auxiliary costs 0.157-0.215s
(intrinsically ~2,000 simplex pivots), approximate auxiliary (PDHG,
C5) yields worthless bases. Pre-C-wave reopening condition: a
different factorization DATA STRUCTURE or hardware regime.

## 5. THE C-WAVE: CLOSED (2026-07-21) — six mandates, six kills

Six creative mandates (creative_attack_angles_2026_07_21.md), codex
gpt-5.5-high, No worktrees remain. Final disposition:
- C1 data-structure replacement: KILLED — best alt format 10.7% SLOWER than CSC in fair microbench; the representation was never the problem. Banked.
- C2 BTRAN||FTRAN overlap: KILLED — the dossier independence claim was WRONG (BTRAN's pivot row determines FTRAN's RHS; legal same-pivot overlap = 0%). A successor audit later derived exact-real next-pivot lookahead; section 6 records why it is still unusable. Banked.
- C3 scaling families: KILLED — every alternative regresses (Pow2 +31% + cert-break; best clean +80%); 10-pass Ruiz is the optimum. Banked.
- C4 perturb+recover: KILLED — trajectory not stall-bound (4,400+2 vs
  4,399 pivots; large eta breaks certificates). Banked.
- C5 PDHG-approximated auxiliary: KILLED — approximate support gives
  4,341-pivot bases; pipeline closed at both ends. Banked.
- C6 primal-route probe: KILLED — the tested dense tableau primal times out at 300s (>973x the flip target). This kills that route, not every possible sparse revised-primal implementation. Banked.
ALL SIX KILLED. The wave-closure ledger entry ("C-WAVE CLOSED") is
recorded; greenbea's file closes at 1.215; the board RESTS AT
23W-0P-1L. Remaining successor duties: (1) optional chronicle round
for the C-wave closure (the pattern in section 3's ship discipline;
board unchanged so it is a science-story round like round 7/8);
(2) nothing else is owed — no re-certs, no open probes, no
uncommitted state. New work on greenbea requires genuinely new
science (see section 4's reopening condition); everything softer
has a dated kill.

## 6. SUCCESSOR REOPENING AUDIT: CLOSED (2026-07-21)

Fresh-eyes review found three claims worth falsifying rather than inheriting:

- Active-bound sifting was explicitly unprobed. A 4,399-pivot trace and 12
  sound epoch/age replays show that 3,611 of 3,868 columns are mandatory;
  even an omniscient policy keeps 95.5% live. Every real policy misses
  entrants, and charged shadow timing projects a 23.9% slowdown. KILLED.
- C2's same-pivot dependency does not forbid exact-real next-pivot BTRAN
  lookahead. The derived rank-one transform predicts all 4,398 leaving rows
  and survives all 33 refactors, but its floating row changes 15 Harris
  entering choices (maximum error 7.495e-6). KILLED at exactness.
- Selected rows of B^-1 can be updated by an exact-real rank-one recurrence,
  but erasing every BTRAN is only 16.92% of wall, below the 17.7013% board
  gap. Compulsory misses cap free caching at 12.81%; fully charged optimistic
  caching is 1.91%, and the same floating-authority failure applies. KILLED.

Reports: `experiments/sifting_falsifier_2026_07_21.md`,
`lookahead_btran_falsifier_2026_07_21.md`, and
`selected_inverse_cache_falsifier_2026_07_21.md`. No production solver code
changed, so the v3 board remains 23W-0P-1L without recertification. The
reopening condition survives in narrower form: a mechanism must change more
than the current BTRAN slice or reduce the 4,399-pivot trajectory while
preserving the fixed certificate and global-policy constraints.

## 7. SECOND SUCCESSOR AUDIT: CLOSED (2026-07-22)

Three further globally applicable fallback claims were tested without changing
the production solver:

- Sparse revised primal is not an unfunded implementation gap. The strongest
  public trajectory found was 7,427 pivots versus cold dual's 4,399. Applying
  only linprogx's mandatory current sparse-kernel costs gives 0.643792s; an
  intentionally over-generous early-sparsity projection is 0.537669s, just
  4.06% below its 0.560440s control and above the 0.461235s board target. Even
  an impossible free partial-pricing projection reaches only 15.16%. KILLED.
- Integrating exact homogeneous Phase 1 with Phase 2 does not remove its cost:
  the cold path already reaches an auxiliary-optimal basis by pivot 2,060. The
  best checkpoint is `2050 + 10 + 2183 = 4243` pivots (3.55% fewer); the best
  optimistic fused-core opportunity is 4.13%. Exact block alternation bottoms
  out at 4,328 pivots and is slower when charged. KILLED.
- Exact simultaneous changed-endpoint rank-k exchange was tested at fresh warm
  checkpoints 512/1536/3072/4096. A native trace established exact basis
  import, allowlisted status normalization, deterministic iteration-0 state,
  and exact k=1 replay. Both the no-flip and exact flip-aware face oracles were
  tested for k=2/3/4; all twelve flip-aware endpoints made zero flips. The only
  surviving bases honestly total 4,404, 4,397, and 4,396 pivots versus the
  3,519 funding ceiling, and all fail the public bootstrap state gate. This
  closes the tested fixed-policy face exchange, not every conceivable rank-k
  algorithm. KILLED.

Reports: `experiments/sparse_primal_falsifier_2026_07_22.md`,
`experiments/phase_transition_falsifier_2026_07_22.md`, and
`experiments/rankk_exchange_falsifier_2026_07_22.md`. Diagnostic rank-k C and
probe code remained isolated and was not merged. With no production change,
there is no v3 recertification: the board remains 23W-0P-1L. Reopening now
requires a genuinely different mechanism, not scalar-endpoint shadow panels,
sparse primal role-swapping, Phase-1 transition bookkeeping, or the tested
fixed-policy simultaneous face exchange.

## 8. REMAINING FRONTIER CENSUS: CLOSED (2026-07-22)

A final clean-room census found no funded CPU mechanism under the current v3
protocol, without changing production code:

- Sequential dual suboptimization is distinct from the old scalar shadow
  panels and the fixed-policy rank-k endpoint audit, but a public-API screen on
  the exact reduced greenbea model took 4,669 pivots versus serial dual's
  3,309. Four-thread median wall was 0.659879s versus 0.380202s. This is
  corroboration, not a wall projection onto linprogx; together with the prior
  panels it supplies no funding invariant for a native build.
- The homogeneous auxiliary's only basis-dependent joint signal is its
  reduced-cost sign score. Cold reaches the auxiliary-optimal face at pivot
  2,060, after which a board flip still requires a 29.44% continuation cut and
  the 20% gate requires 34.01%. Integrated and alternating measurements reach
  only 3.55% and 1.61% total-pivot gains. KILLED without implementation.
- The four-vCPU solve slice cannot fund ordinary multicore level scheduling.
  Accelerator-resident simplex is ineligible because v3 allocates CPU only;
  changing both solvers' hardware would be a new campaign protocol.
- Certificate-corrected basis-solve replay used four actual bases and next
  right-hand sides. Exact BTRAN rows match 404/404 Harris decisions; fixed
  matched-Jacobi true-residual correction at degrees 1/2/3/4/8/16/32 matches
  0/404 at every degree, with BTRAN residuals stagnant at 0.742--0.998.
  Every sample therefore pays exact fallback and is a strict regression.
  This kills ordinary frozen-refactor LU Krylov and this recurrence through
  degree 32, not every imaginable sparse approximate inverse or recycled
  space.

Reports: `experiments/remaining_frontier_census_2026_07_22.md` and
`experiments/krylov_basis_solve_falsifier_2026_07_22.md`. The diagnostic is a
standalone Python probe; no C or production solver behavior changed. There is
therefore no v3 recertification, and the certified board remains 23W-0P-1L.
Reopening requires a concrete richer approximate-inverse/recycled construction
with complete funding and authority evidence, a new Phase-1 invariant with a
demonstrated 29--34% continuation opportunity, or an explicitly new hardware
protocol.

## 9. POST-FRONTIER FRESH-EYES AUDIT: CLOSED (2026-07-22)

The earlier statement “no funded CPU mechanism” remains a funding census, not
an impossibility theorem. A new Overmind wave tested the strongest concrete
constructions outside its measured scope:

- Fixed non-diagonal left transformation is exact and certificate-mappable,
  but `U=B_512^-1` makes `UA` 3.906x denser and later sampled LU fill rises
  78.48%, 134.26%, and 156.81%. Natural algebraic reformulations are KILLED.
- Oracle top-`K` whole-inverse sparsification reaches full sampled Harris
  authority only at `K=128`, whose favorable traffic is 93.280us/equivalent
  pair. The only raw-funded `K=16` point fails half the decisions; exact
  fallback leaves an 8.46% modeled wall gain. KILLED in scope.
- Full-KKT block PDAS was a real gap: it permits both primal and dual
  infeasibility and accepted a width-60 exchange, beyond the previous `k<=4`
  face audit. Adversarial review found a backward-step bug; after repair and a
  fresh authoritative run, the fixed greedy-prefix+Bland policy accepted
  widths 60/8/2/2, had median width 5 versus the gate of 18, and stopped with
  bound violation 62,263.4827. It is KILLED only for that exact global
  selection policy; improving forward-valid edges exist outside its prefixes.

Reports: `experiments/global_reformulation_falsifier_2026_07_22.md`,
`experiments/rich_inverse_falsifier_2026_07_22.md`, and
`experiments/block_pdas_falsifier_2026_07_22.md`. The latter two include
standalone reproducible probes. No production source changed, so there is no
v3 recertification and the board remains 23W-0P-1L. Reopening requires a
different predeclared full-KKT selection/globalization rule, a different
concrete approximate-inverse representation with complete authority and cost,
or a genuinely new factor/hardware protocol.

## 10. Standing state, memory, and Evan

- All campaign evidence through the post-frontier fresh-eyes audit is committed
  and tagged. Disposable diagnostic worktrees were removed after their reports
  were banked; none entered the production solver.
- Quotas (check fresh): codex snapshot was stale-96% but workers run
  fine (plan likely changed); claude 7d ~59%; GLM metered (~$1.4/$4.4
  per Mtok).
- Memory (auto-loaded from
  /home/evan/.claude/projects/-home-evan-dev-linprogx/memory/):
  linprogx-solver-goal.md (board+constraints — update on board moves,
  plus the MEMORY.md index line), subagent-model-policy.md,
  codex-worker-cont-stdin.md, codex-model-choice.md (gpt-5.6 status).
- Evan's operating style: commissions fan-outs ("fan out hypotheses in
  tension"); values honest kills as much as ships; wants Telegram
  updates (nanobot notify) at board moves/milestones; reads the
  gh-pages chronicle; asks direct questions expecting direct answers
  with the numbers. The campaign's real product is the ledger's ~50
  verdicts as much as the 23W board — keep both immaculate.
