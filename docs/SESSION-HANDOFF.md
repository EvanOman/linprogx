# Session Handoff — linprogx campaign orchestration

Read this + the tail of docs/HANDOFF.md (the research ledger, entries are
dated; everything below ~"PRESOLVE V2 SHIPPED" is the current era — the
2026-07-17 entries are the census wave).

## State (2026-07-17)
- Branch perf-supernodal-simplex in THIS worktree
  (/home/evan/dev/linprogx-perf-worktree). Backup tag
  session-backup-20260702 tracks HEAD. Main checkout /home/evan/dev/linprogx
  belongs to another session (web-demo) — never touch it.
- BOARD OF RECORD (protocol v3: 3 hosts x 7 pairs median-of-hosts, AWS
  us-west-2, assets/modal_bench_928399cf5fea_paired_hosts3.json et al.):
  **20W-0P-4L** (incl. qap15 coverage win). Losses: greenbea 1.69,
  pds_10 1.26-1.57 (host-dependent), woodw 1.20, 80bau3b 1.06.
- The census wave (experiments/loss_census_2026_07_16.md) produced two
  certified ships on 2026-07-17: H0 quadratic presolve row-build fix
  (osa_60 0.280, osa_14 0.912) and H1 presolve fixpoint re-stage
  (cre_a 0.939, stocfor3 0.962). 28 settled hypotheses in the ledger.
- GOAL (user's /goal hook): beat HiGHS on ALL 24 LPnetlib instances.

## Immediate queue
1. H4 (greenbea ranged-row singleton elimination) and H5 (pds_10
   degree-2 network contraction) opus probes may be running or
   unresolved — check worktrees /home/evan/dev/linprogx-h4 and -h5 and
   the census doc sections 4/5 for specs+gates. greenbea needs ~41%
   (ceiling 35-45%); pds_10 needs ~21% net of PDHG iteration effects.
2. Chronicle round 3 after H4/H5 resolve: update
   tools/build_report_data.py CANONICAL_BOARD + docs/CAMPAIGN.md +
   report tiles to the 20W board (+ any H4/H5 outcome), ingest new
   artifacts (tools/replay_bench.py artifacts), rebuild
   (tools/build_report_data.py), push gh-pages (worktree checkout of
   gh-pages branch; cp campaign_report.html -> index.html), republish
   Claude artifact
   https://claude.ai/code/artifact/a1eba80f-2b5d-426f-8527-a2d7f4545d3e
   (Artifact tool with url param).
3. 80bau3b (1.062, needs ~6%): census levers spent; on-host refactor
   bandwidth damps presolve gains (local +26% -> on-host +11%).
   Bandwidth-lean IPM factorization work is the likely family.
4. woodw (1.20): no queued lever. Local IPM is near HiGHS parity; the
   on-host gap is bandwidth in the refactor slice. Same family as 3.

## Orchestration protocol (hard lessons, do not relearn)
- Overmind mode: Fable plans/reviews; workers implement. NEVER
  SendMessage-resume a Claude subagent from a Fable session (resume drops
  the model pin -> silently bills Fable). Fresh Agent spawns with explicit
  model. CODEX QUOTA: check ~/.claude/skills/overmind/bin/usage-check.sh
  first — codex weekly hit 90% on 2026-07-16 (resets 07-23); claude/opus
  workers are the fallback and handled H0/H1/H4/H5 fine.
- Codex workers: ~/.claude/skills/overmind/bin/codex-worker.sh run -C
  <worktree> --label X - (brief on stdin). For cont: ALSO use the
  stdin-file form (cont <session> - < brief.txt) — "$(cat -)" in a zsh
  pipeline hangs silently.
- Worker briefs: GOAL/CONTEXT/CONSTRAINTS/DONE WHEN/VERIFY; falsifier-
  first with kill criteria; "no commits"; rebuild before test
  (UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev --no-build-isolation,
  then uv pip install --reinstall -e . --no-build-isolation to force the
  C extension rebuild); characterization-first on high-risk areas.
- LOADED-BOX DOCTRINE (25th settled): A/B ship gates need a quiet box —
  alternating A/B is robust to load drift but NOT to load-dependent
  relative effects (a 16% "gain" evaporated once the box went quiet).
  Don't run ship-gate A/Bs while other workers bench.
- Paired interleaved protocol only; scoreboard verdicts are protocol-v3
  median-of-hosts (--hosts 3). Single-shot and single-host are not
  scoreboard-grade. envab mode exists for on-host env-knob A/B.
- Modal harness: tools/modal_bench.py; upload-src after EVERY commit
  before benching (keyed by exact sha); ~$0.20-1.00/container.
- Ship discipline: orchestrator reviews the diff, reruns VERIFY, commits,
  advances the backup tag, records the dated ledger entry (ship or kill)
  in docs/HANDOFF.md. Chronicle + gh-pages + artifact after board moves.
