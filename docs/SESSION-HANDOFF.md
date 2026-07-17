# Session Handoff — linprogx campaign orchestration

Read this + the tail of docs/HANDOFF.md (the research ledger, entries are
dated; everything below ~"PRESOLVE V2 SHIPPED" is the current era).

## State (2026-07-16)
- Branch perf-supernodal-simplex in THIS worktree
  (/home/evan/dev/linprogx-perf-worktree). Backup tag
  session-backup-20260702 tracks HEAD. Main checkout /home/evan/dev/linprogx
  belongs to another session (web-demo) — never touch it.
- Canonical board (AWS us-west-2 pinned, paired protocol,
  assets/pin4_chunk{1,2}.json): 14W-5L-4P +qap15 coverage = 15 wins.
  Losses: greenbea 1.69, osa_60 1.50*, osa_14 1.34*, pds_10 1.20,
  cre_a (0.966 ratio, 4/7 wins). Parity: woodw 0.996, pilot87 0.995,
  80bau3b 1.010, stocfor3 1.010. (*host-variance-conditional, see ledger.)
- GOAL (user's /goal hook): beat HiGHS on ALL 24 LPnetlib instances.

## Immediate queue
1. Measurement protocol v3: multi-container medians for the
   bandwidth-sensitive set (osa pair, pds pair, pilot87, woodw, greenbea)
   — tools/modal_bench.py is pinned cloud=aws/region=us-west-2; run N>=3
   containers per certification, median-of-hosts verdicts.
2. cre_a: 0.966 ratio but 4/7 wins — one more small IPM shave flips it
   (min-degree is its binding setup slice; AMD-style approx degree =
   outcome-gated escalation, queued in ledger).
3. greenbea 1.69: DS per-pivot slices btran/ftran remain (pivot-row reuse
   family CLOSED — see 23rd settled). Consider FT solve micro-opts.
4. pds_10 1.20: PDHG closed at unit level; only protocol-v3 remeasure +
   host-median verdict.
5. Chronicle: replay new ships into assets/campaign.db
   (tools/replay_bench.py, idempotent), update docs/CAMPAIGN.md +
   docs/campaign_report.html (re-embed via tools/build_report_data.py),
   republish: gh-pages branch (index.html) + Claude artifact
   https://claude.ai/code/artifact/a1eba80f-2b5d-426f-8527-a2d7f4545d3e
   (republish same file path from the session that owns it, or pass url).

## Orchestration protocol (hard lessons, do not relearn)
- Overmind mode: Fable plans/reviews; workers implement. NEVER
  SendMessage-resume a Claude subagent from a Fable session (resume drops
  the model pin -> silently bills Fable; memory subagent-model-policy.md).
  Fresh Agent spawns with explicit model, or codex workers.
- Codex workers: ~/.claude/skills/overmind/bin/codex-worker.sh run -C
  <worktree> --label X - (brief on stdin; gpt-5.5; background via Bash).
  Briefs: GOAL/CONTEXT/CONSTRAINTS/DONE WHEN/VERIFY; falsifier-first with
  kill criteria; "no commits"; I commit after my own verification.
- Worker briefs must say: rebuild before test (UV_CACHE_DIR=/tmp/uv-cache
  uv sync --extra dev --no-build-isolation), no git ops, foreground/
  bounded, never stop to wait for background jobs.
- Kernel experiments MUST run on the current stack (stale exp worktrees
  rediscover shipped levers — 23rd settled). exp-leaving/exp-panel
  worktrees are STALE; prefer this worktree for probes, patch-port
  anything from exp branches (never wholesale merge — fork bases predate
  ports).
- Paired interleaved protocol only; single-shot is not scoreboard-grade.
- Modal harness: tools/modal_bench.py; upload-src after EVERY commit
  before benching (keyed by exact sha); ~\$0.20-1.00/run.
