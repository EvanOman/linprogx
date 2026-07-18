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
  us-west-2): **21W-1P-2L** (incl. qap15 coverage win).
  - Parity: cre_a — TRUE coin flip (0.939/1.021/0.995 across waves).
  - Losses: greenbea ~1.7 (EVERY scoped family settled — 35th settled;
    needs a new idea class) and pds_10 1.26-1.57 (ranged-row
    architecture, 29th settled).
  - woodw flipped 0.962 21/21 (a2 refactor scheduling, b394c7e);
    pilot87 is a host-conditional win (cumulative 30/42, 0.927).
- This session shipped and certified: protocol v3 + envab harness modes,
  DS solve-slice instrument, H0 quadratic presolve row-build fix
  (osa_60 0.280!, osa_14 0.912), H1 presolve fixpoint re-stage (cre_a,
  stocfor3 flips), native equality-row aggregation double-gated
  (80bau3b 0.881, d2q06c 0.371, ken_07 0.410). Ledger now holds 30+
  dated verdicts (through the aggregation cert).
- GOAL (user's /goal hook): beat HiGHS on ALL 24 LPnetlib instances.

## Immediate queue
1. pds_10 (1.26-1.57): the ONLY loss with a scoped path — ranged-row/
   inequality support end-to-end (29th settled): presolve records +
   PDHG/IPM kernels + postsolve, ~25% ceiling (HiGHS's arc-contraction
   shape). An architecture project; commission deliberately.
2. greenbea (~1.7): NO live scoped lever (35th settled closed the
   pivot frontier: pivot parity and per-pivot parity trade against
   each other; presolve/leaving-rules/basis/ratio-test/crash/slice
   families ALL settled 20th-35th). Needs a new idea class. Do not
   re-probe settled families.
3. cre_a (0.995 coin flip, 13/21): no engineered lever left (34th
   settled). Optionally re-certify occasionally; it may certify on a
   friendly wave. The a2 ship did not move its path.
4. Supernodal sparse front (lever c of the refactor unit): needs a
   ~57% front-time cut for a further woodw-class gate; woodw already
   flipped, so commission only for pilot87-robustness/general value.
5. Chronicle round 5 (21W-1P-2L) may be committed/published — verify
   gh-pages and the Claude artifact match before re-publishing.

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
