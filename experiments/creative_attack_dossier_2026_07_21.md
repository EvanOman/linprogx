# Creative-attack dossier — greenbea at 1.215 (2026-07-21)

STATE: board 23W-0P-1L. greenbea is the last loss: v3 1.215 [1.208,1.235]
on-host; local ~0.37s (post-SIMD-ship) vs HiGHS 0.24s. Flip needs ~-18%
more end-to-end. Read the prior evidence chain as needed:
greenbea_dossier_2026_07_18.md (+corrections), kernel_campaign_dossier_
2026_07_19.md, k1_census (IPC 0.3-0.6 in solves), lsa/lsb reports (level
scheduling + interleaving killed), k3/k12 (dense sweeps, flags killed),
phase1_predictions (the derived auxiliary; B* bases), k7 (2,399-pivot
native basis, aux costs 0.157-0.215s), k9 (density shaping LIVE,
pipeline-blocked), int_kernel_combined (the shipped SIMD pair).

THE PROVEN WALLS (do not re-attack head-on): gathered sparse triangular
solves at ~1.5k rows are memory-latency-bound (IPC 0.3-0.6) — immune to
dense sweeps of sparse storage, compiler flags/vectorization, software
pipelining (0.000% collisions yet slower — disambiguation-bound), and
level scheduling (overhead swamps at this size). Leaving rules, starting
bases (transferred), presolve depth, precision-of-values, cache
reordering: all have dated kill verdicts.

THE OPEN GEOMETRY: (a) the pipeline route (K7 basis 2,399 pivots + K9
shaping) is blocked ONLY by the auxiliary's construction cost; (b) BTRAN
and FTRAN within one pivot are INDEPENDENT solves — never overlapped;
(c) the factorization DATA STRUCTURE itself (CSC gather-chase) has never
been replaced, only re-traversed; (d) HiGHS's scale_strategy=4 moved its
own pivot counts — scaling as trajectory-shaper is unprobed on OUR
solver; (e) cost perturbation as an anti-stall trajectory-shaper (with
exact recovery) was proven robust FOR HiGHS but never tried FOR us;
(f) PDHG was never aimed at the HOMOGENEOUS auxiliary (Ax=0), where
approximate support suffices.

RULES (bind all agents): no network whatsoever (logs audited; violation
= discard). Never read/fetch any solver's source. No per-problem tuning
(global mechanisms only). eps=2e-5 fixed; certificate-backed optimality;
knob-gated changes byte-identical off; falsifier-first with explicit
kill criteria; honest verdicts. Build: UV_CACHE_DIR=/tmp/uv-cache uv
sync --extra dev --no-build-isolation, then uv pip install --reinstall
-e . --no-build-isolation after C changes. Fixtures /tmp/lpsuite. No git
ops. Foreground. DELIVERABLE: experiments/c<N>_<slug>_2026_07_21.md with
measurements, flip arithmetic vs the -18% need, verdict LIVE/KILLED.
