# Session Handoff — linprogx campaign orchestration

Read this + the tail of docs/HANDOFF.md (the research ledger, entries are
dated; the 2026-07-17 entries — census wave + aggregation arc — are the
current era).

## State (2026-07-17, end of the census/aggregation session)
- Branch perf-supernodal-simplex in THIS worktree
  (/home/evan/dev/linprogx-perf-worktree). Backup tag
  session-backup-20260702 tracks HEAD. Main checkout /home/evan/dev/linprogx
  belongs to another session (web-demo) — never touch it.
- BOARD OF RECORD (protocol v3: 3 hosts x 7 pairs median-of-hosts, AWS
  us-west-2): **21W-2P-1L** (incl. qap15 coverage win).
  - Parity coin flips (host-lottery, no engineered lever outstanding):
    cre_a 0.995 and pds_10 0.985 [0.891,1.030] — periodic re-certs may
    flip either; do not build units for them.
  - Sole loss: greenbea ~1.7 — frontier TOTALLY closed (38 settled
    verdicts; see 35th/37th/38th). Needs a genuinely new idea class.
  - The 07-18 pds arc: clean-room netagg + parallel/dominance merging
    took pds_10 1.26-1.57 -> 0.985 and pds_20 0.824 -> 0.499.
- This session shipped and certified: protocol v3 + envab harness modes,
  DS solve-slice instrument, H0 quadratic presolve row-build fix
  (osa_60 0.280!, osa_14 0.912), H1 presolve fixpoint re-stage (cre_a,
  stocfor3 flips), native equality-row aggregation double-gated
  (80bau3b 0.881, d2q06c 0.371, ken_07 0.410). Ledger now holds 30+
  dated verdicts (through the aggregation cert).
- GOAL (user's /goal hook): beat HiGHS on ALL 24 LPnetlib instances.

## Immediate queue
1. Coin flips (cre_a 0.995, pds_10 0.985): occasionally re-certify
   (one v3 wave, both instances); a friendly host draw certifies
   either as a win. No engineering.
2. greenbea: THE RESEARCH CAMPAIGN IS CLOSED (39th-45th settled,
   2026-07-18). Terminal account: per-pivot work at a proven hard
   floor (rho density factorization-invariant; kernels 96%
   cache-line-reuse optimal; panels die in 1.28 pivots; fp32 gains
   0.98-1.18x); pivot gap identified as HiGHS's refined dual
   Phase-1 but replication 0-for-2 (11,377 vs their 1,655).
   Reopening requires a novel Phase-1 formulation or an unnamed
   idea class. Read experiments/greenbea_dossier + the probe
   reports before proposing ANYTHING.
3. Supernodal sparse front (refactor lever c): optional
   general-value unit; not board-critical.
4. Chronicle round 7 (research-campaign story) may be committed/
   published — verify gh-pages + artifact before re-publishing.

## Orchestration protocol (hard lessons, do not relearn)
- Overmind mode: Fable plans/reviews; workers implement. NEVER
  SendMessage-resume a Claude subagent from a Fable session. Fresh Agent
  spawns with explicit model (opus workers handled all of today's units).
  CODEX QUOTA: check usage-check.sh first — codex weekly hit 90% on
  2026-07-16 (resets 07-23).
- codex-worker.sh: briefs via stdin-file with '-' for BOTH run and cont
  ("$(cat -)" in zsh pipelines hangs silently).
- Worker briefs: GOAL/CONTEXT/CONSTRAINTS/DONE WHEN/VERIFY; falsifier-
  first with kill criteria; no git ops; rebuild before test
  (UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation,
  then uv pip install --reinstall -e . --no-build-isolation for C
  changes); characterization-first on high-risk areas.
- LOADED-BOX DOCTRINE (25th settled): ship-gate A/Bs need a quiet box.
- PROJECTION DOCTRINE (31st settled): never project pivot/iteration
  counts from another solver's realized behavior — shape parity is not
  pivot parity. Falsify on OUR solver before sizing a unit.
- Scoreboard: protocol-v3 median-of-hosts only (--hosts 3). envab mode
  for on-host env-knob A/B. upload-src after EVERY commit before
  benching. Single-shot/single-host is not scoreboard-grade.
- Ship discipline: orchestrator reviews the diff, reruns VERIFY (just ci
  for substantial changes — includes security; the repo pins
  exclude-newer in uv.toml, advanced only within the machine's 7-day
  release-age rule), commits, advances the backup tag, records the dated
  ledger verdict, chronicles after board moves.
- GIT QUIRK: this checkout maps LF->CRLF on touch — NEVER git stash here
  (a stash/pop cycle conflicts on churned files; use a scratch worktree
  at HEAD to test pre-existing failures instead).
- INTEGRITY PROTOCOL (2026-07-17 incident — see NETAGG INTEGRITY
  INCIDENT in the ledger): the codex sandbox can sometimes reach the
  network; a worker once downloaded HiGHS source and its gate-passing
  unit was quarantined for it. EVERY worker brief must explicitly
  prohibit fetching remote content (no curl/wget/new installs; solver
  source strictly forbidden — the campaign constraint). Before
  shipping any unit whose design could have been externally informed,
  AUDIT the worker's event log: codex-worker.sh log <session>, grep
  for curl/wget/github/clone. Network use on such a unit = discard.
