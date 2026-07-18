# greenbea falsifier probe P-B — PRECISION (2026-07-18)

## Verdict

**KILLED.** Neither proposed variant survives:

- **In-loop fp32 simulation dies at pivot 117.** The first 116 committed
  pivots match fp64 exactly. At the next decision, fp64 chooses leaving basis
  position 178 / entering column 755, while the fp32-rounded lane chooses
  position 178 / column 577. Its rounded pricing alpha is
  `-7.2028036868e-08`, but FTRAN produces a zero pivot at the leaving
  position. The solver then skips the same tiny pivot 49,884 times and exits
  `numerical_error` at the 50,000-iteration limit.
- **The scout recovery is not cheap.** A clean fp64 solve imported from the
  fp32 terminal basis does certify, but needs 4,288 pivots and 0.4265 s, or
  100.8% of the 0.4230 s cold baseline. The observed fp32 simulation plus
  fp64 recovery is 1.2508 s.
- **The measured bandwidth ceiling is below the plan gate.** Native fp32
  containers project a 6.82% end-to-end saving on the 71.3% attacked wall,
  well below the 20% kill threshold. This isolated projection excludes
  refinement, conversion, promotion, and uncertainty-band overhead, so those
  mechanisms cannot rescue the result.

Both plan kill criteria therefore fire independently: trajectory divergence
has no cheap certified recovery, and measured kernels project less than 20%
end-to-end.

## Contract and method

Required inputs were read in the prescribed order: the research plan, dossier,
Claude Opus idea 1, GPT-5 idea 3, and Codex contrarian idea 2. The experiment
used the prescribed greenbea reduction at **1,525 x 3,868 x 23,274**, fixed
`eps=2e-5`, Dantzig leaving selection, EXPAND enabled, and the shipped
Forrest-Tomlin path.

Instrumentation is diagnostic and environment-gated:

- `LINPROGX_DS_FP32_SIM=1` performs an actual C `double -> volatile float ->
  double` round trip on every BTRAN rho value, priced pivot-row value, FTRAN
  alpha value, and updated/recomputed reduced cost.
- `LINPROGX_DS_PIVOT_TRACE=1` exports committed `(leaving basis position,
  entering column)` pairs and the first tiny-pivot failure.
- The final solver exit is unchanged: it refactorizes the terminal basis and
  recomputes primal values, dual values, reduced costs, objective, and residual
  from the original fp64 data. The scout recovery explicitly unsets fp32
  simulation before importing the speculative terminal basis.

The requested build was run with `UV_CACHE_DIR=/tmp/uv-cache` and
`UV_OFFLINE=1`, followed after each C change by the requested editable
reinstall. No network tool, external request, or solver-source inspection was
used.

## Baseline and disabled-path control

| Measure | Clean fp64 baseline |
|---|---:|
| Status | optimal |
| Pivots | 4,399 |
| Wall (pinned final run) | 0.423001 s |
| Objective | -72,555,248.1298459 |
| Original-space max equality residual | 1.76889e-7 |
| Max bound violation | 3.85749e-12 |
| Terminal basis SHA-256 | `1de4b80053030095...e9a3764af6c3` |
| Terminal bound-status SHA-256 | `3eb7603ac281476b...3aedba588122ca2` |

With all new gates disabled, the post-change solver reproduced the pre-change
status, 4,399 pivots, objective, residual, and both terminal hashes exactly.
The off path is outcome-byte-identical for the exported basis and bound-status
arrays.

## Sub-question 1: trajectory preservation

The simulated lane has an exact 116-pivot common prefix with fp64. Divergence
occurs on the **117th pivot decision**:

| Item | fp64 | fp32-rounded lane |
|---|---:|---:|
| Leaving basis position | 178 | 178 |
| Entering column | 755 | 577 |
| Rounded pricing alpha for chosen column | — | -7.2028036868e-8 |
| FTRAN pivot at leaving position | non-tiny/committed | 0.0/rejected |

This is not a harmless alternative trajectory. After the 116th commit the
rounded lane cannot make another basis exchange, records 49,884 tiny-pivot
skips, and exits `numerical_error` with a 2.02e9 original-space residual. Its
terminal basis and bound-status hashes differ from fp64. Consequently it has
neither the same final basis nor an acceptable fp64 exit certificate.

### Scout recovery from the speculative terminal basis

The recovery pass disabled simulation, rebuilt the imported basis from fp64
matrix data, and used the ordinary fp64 dual simplex and exit certificate.

| Measure | fp64 recovery |
|---|---:|
| Status | optimal |
| Additional pivots | 4,288 |
| Wall | 0.426464 s |
| Wall / cold baseline | 100.82% |
| Objective | -72,555,248.1298460 |
| Relative objective delta vs baseline | 1.03e-15 |
| Original-space max equality residual | 1.76889e-7 |
| Max bound violation | 4.55e-13 |

The certificate is valid at `eps=2e-5`, but recovery is effectively a second
cold solve. It exceeds the contrarian write-up's 8% terminal-pass budget by
more than 12x. The measured diagnostic total is **0.82434 + 0.42646 =
1.25080 s**. Even substituting the bandwidth projection for the simulation
overhead gives **0.39417 + 0.42646 = 0.82063 s**, still 1.94x baseline.

**Trajectory result: in-loop KILLED; scout KILLED.**

## Sub-question 2: bandwidth reality

The native C microbenchmark used the actual reduced greenbea CSR values and
dimensions. It set rho support to the measured median **897 / 1,525 = 58.82%**
and pivot-row/reduced-cost support to **3,625 / 3,868 = 93.72%**. Each of nine
paired trials ran 20,000 repetitions, alternated fp64-first/fp32-first order,
and the final reported batch was pinned to logical CPU 4. Values below are
paired-trial medians; times are per kernel invocation.

| Kernel | fp64 | fp32 | Measured fp32 speedup | Dossier wall share |
|---|---:|---:|---:|---:|
| BTRAN dense active sweep | 1.188 us | 1.215 us | 0.976x | 18.9% |
| FTRAN dense active sweep | 1.293 us | 1.182 us | 1.141x | 17.9% |
| Pivot-row CSR scatter | 17.179 us | 14.460 us | 1.185x | 24.8% |
| Reduced-cost indexed update | 3.943 us | 3.453 us | 1.141x | 9.7% |

The pivot-row benchmark casts the actual matrix values into separate float and
double containers and performs the CSR scatter over a deterministic,
well-spread 897-row support. The solve sweeps use identically active float and
double working containers at that density. The reduced-cost pass uses a
deterministic noncontiguous 3,625-entry support over structural plus artificial
columns. These are container/kernel measurements, not the theoretical 2x byte
ratio.

### End-to-end projection

Using paired median `fp32_time / fp64_time` ratios and the fixed dossier split:

```text
remaining wall
  = 28.7% untouched
  + 24.8% * 0.84421   (pivot row)
  + 18.9% * 1.02410   (BTRAN)
  + 17.9% * 0.87646   (FTRAN)
  +  9.7% * 0.87668   (reduced cost)
  = 93.184%

projected saving = 6.816%
```

The weighted contributions are -0.46 percentage points from BTRAN, +2.21 from
FTRAN, +3.86 from pivot-row scatter, and +1.20 from reduced-cost update. The
result misses the 20% end-to-end gate by 13.18 points and the 41% greenbea flip
requirement by 34.18 points.

## Final adjudication

| Variant | Trajectory/certificate | Recovery economics | Bandwidth gate | Verdict |
|---|---|---|---|---|
| fp32 in-loop body | Diverges/stalls at pivot 117; no certificate | n/a | 6.82% < 20% | KILLED |
| fp32 scout + fp64 terminal basis | fp64 eventually certifies | 4,288 pivots; 100.8% of cold wall | 6.82% < 20% | KILLED |

**Final verdict: KILLED.** Precision is not a live greenbea primary under this
probe. The failure is stronger than a marginal miss: selection noise breaks the
trajectory almost immediately, the speculative basis carries essentially no
useful progress into fp64, and measured native-container gains are single-digit
end-to-end.

Reproduction entry point: `python -m experiments.probe_precision` (use
`taskset -c 4` for the pinned bandwidth batch).
