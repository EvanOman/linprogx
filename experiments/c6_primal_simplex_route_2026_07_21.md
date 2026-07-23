# C6 — dense primal-simplex route on presolved greenbea (2026-07-21)

## Verdict

**KILLED.** The existing dense two-phase primal simplex did not return a solver
status or produce a certificate within the 300-second process bound.
Experiment-only instrumentation observed at least **3,000 completed tableau
pivots**; the 3,000th completed at **295.353 s of solver wall**. This is
nowhere near the condition for assessing a sparse primal implementation.

The current public baseline is approximately 0.370 s. A flip needs an 18%
reduction, so the target is:

```text
0.370 s * (1 - 0.18) = 0.3034 s
2x flip-relevant gate = 0.6068 s
```

By the last observed pivot, the dense solve had consumed at least 295.353 s,
or **>973.5x the 0.3034 s flip target** and **>486.7x the 0.6068 s assessment
gate**, without reaching a solver status or certificate. Per C6, no
sparse-primal cost assessment follows.

## Rules and setup

- Fixture: `/tmp/lpsuite/lp_greenbea.mat`.
- Standard linprogx equality-and-bounds presolve, with no per-problem tuning.
- Raw shape: 2,392 x 5,598 x 31,070 nonzeros.
- Presolved shape: **1,525 x 3,868 x 23,274 nonzeros**.
- Solver: linprogx's existing dense `Solver`, two-phase primal simplex,
  minimization, equality rows, and the presolved variable bounds.
- `eps=2e-5` fixed.
- Foreground execution under an external timeout.
- Network disabled mechanically with `UV_OFFLINE=1`; no network access was
  attempted.
- No solver source was read or changed. No C files were changed. No Git
  operations were used.

The mandated build completed successfully offline:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  uv sync --extra dev --no-build-isolation
```

There were no C changes, so the dossier's post-C-change reinstall command was
not applicable.

## Falsifier-first method

The initial plan capped the solver at 8,001 iterations, one beyond C6's
explicit `>8,000` pivot kill gate. Calibration exposed that the public
`Solution.iterations` field reports zero on the capped calibrations despite
observed tableau pivots, so it cannot provide honest total pivot accounting
for this interrupted two-phase run.

Without reading or changing solver source, the final driver wraps the already
imported tableau-pivot callable in the experiment process. It increments a
counter after each completed pivot, reports every 1,000 pivots, and is prepared
to stop immediately at pivot 8,001. A short profiler calibration independently
confirmed the accounting: a run capped at 101 produced exactly 101 calls to
the C tableau-pivot function.

Reproducible driver: `experiments/c6_primal_simplex_probe.py`.

Decisive command:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 \
  timeout --foreground --signal=TERM --kill-after=5s 300s \
  .venv/bin/python experiments/c6_primal_simplex_probe.py \
  /tmp/lpsuite/lp_greenbea.mat \
  --max-iterations 8001 --stop-after-pivots 8001
```

## Measurements

### Calibration

| configured cap | public status | public iterations | observed pivot calls | solve wall |
|---:|---|---:|---:|---:|
| 1 | iteration_limit | 0 | not profiled | 2.917 s |
| 11 | iteration_limit | 0 | not profiled | 2.799 s |
| 101 | iteration_limit | 0 | **101** | 5.043 s under profiler |
| 1,001 | iteration_limit | 0 | not profiled | 16.014 s |

The zero public count is therefore a reporting limitation for interrupted
Phase I, not evidence of zero work.

### Decisive bounded run

| event | completed pivots | elapsed solve wall |
|---|---:|---:|
| progress | 1,000 | 12.299 s |
| progress | 2,000 | 102.166 s |
| progress | 3,000 | 295.353 s |
| external timeout | **>=3,000** | **>=295.353 s** |

- Process status: external timeout (`124`) at the 300-second process bound.
- Solver status: unavailable; the dense solve had not returned.
- Optimality certificate: unavailable; no solution was returned.
- Setup measurements on the decisive run: presolve 0.016 s, dense problem
  materialization 0.289 s, pre-solve peak RSS 292,540 KiB.
- Earlier completed calibrations reached peak RSS between 509,488 and 660,460
  KiB.

The decisive timing includes one lightweight Python counter wrapper around
each existing pivot call. It is not the cause of the verdict: the unwrapped
1,001-cap calibration took 16.014 s, while the wrapped decisive run reached
its first 1,000 pivots in 12.299 s.

The 8,001-pivot hard gate was not reached within the bounded wall time. The
route is nevertheless falsified earlier and more strongly by C6's conditional
assessment rule: even the first 1,000 pivots took 12.299 s, **20.27x slower
than the entire 0.6068 s “within 2x” gate**, and the solver was still
uncertified when the 300-second process bound fired.

## Conclusion

The dense primal route attacks the algorithm choice rather than the sparse
dual-simplex kernels, but its tableau work is catastrophically mismatched to a
1,525 x 3,868 reduced LP. It cannot contribute to the required 18% end-to-end
improvement. Because it is not remotely within 2x of flip-relevant wall time,
C6's condition for assessing a sparse primal variant is false. Stop here.
