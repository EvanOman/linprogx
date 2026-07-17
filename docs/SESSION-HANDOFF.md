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
  us-west-2): **20W-1P-3L** (incl. qap15 coverage win).
  - Parity: cre_a — TRUE coin flip (0.939 and 1.021 across waves,
    iterations bit-identical; the aggregation reject-scan costs it ~2%
    on a +-3% margin).
  - Losses: greenbea ~1.7, pds_10 1.26-1.57 (host-dependent), woodw 1.20.
- This session shipped and certified: protocol v3 + envab harness modes,
  DS solve-slice instrument, H0 quadratic presolve row-build fix
  (osa_60 0.280!, osa_14 0.912), H1 presolve fixpoint re-stage (cre_a,
  stocfor3 flips), native equality-row aggregation double-gated
  (80bau3b 0.881, d2q06c 0.371, ken_07 0.410). Ledger now holds 30+
  dated verdicts (through the aggregation cert).
- GOAL (user's /goal hook): beat HiGHS on ALL 24 LPnetlib instances.

## Immediate queue (all scoped, none dispatched)
1. cre_a reject-scan cost: the native aggregation scan costs ~2.75ms on
   cre_a (a reject) — target <0.5ms or a proven a-priori discriminator
   (fill-trajectory minima proven NON-separating; see the aggregation
   cert entry). Flipping this restores cre_a's win margin.
2. woodw (1.20): on-host IPM bandwidth family — local IPM is near HiGHS
   parity; the gap is the refactor slice's memory-bandwidth behavior on
   Modal hosts. Bandwidth-lean factorization work (blocking/layout) is
   the family. No specific unit scoped yet — needs its own census.
3. greenbea (~1.7): presolve frontier CLOSED (31st settled: aggregation
   hits HiGHS's shape but our DS does MORE pivots on it — shape parity
   is not pivot parity). The remaining gap is pricing-side: HiGHS does
   2,836 pivots to our 4,399 on comparable shapes. Dual steepest edge
   (Forrest-Goldfarb, published literature — papers OK, never solver
   source) is the candidate unit. High effort.
4. pds_10: requires ranged-row/inequality support end-to-end (29th
   settled) — an architecture project (presolve records + kernels +
   postsolve), ~25% ceiling. Scope only if judged worth it.
5. Chronicle: round 4 (aggregation era, 20W-1P-3L) may already be
   committed/published — verify gh-pages 750c859+ and the Claude
   artifact match the board before re-publishing.

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
