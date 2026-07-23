# C4 — cost perturbation with exact recovery (2026-07-21)

## Verdict

**KILLED.** The best certificate-backed arm is `eta=1e-10`: the perturbed solve
takes 4,400 pivots and exact true-cost recovery takes 2 more, versus 4,399 cold.
Its median paired total-wall ratio is **1.000468** (**0.047% slower**), missing
the mandate's 10% live threshold by 10.047 percentage points and the board's
approximately 18% flip need by 18.047 points.

The certificate kill also fires. At `eta >= 1e-8` the perturbed solve reports
`dual_infeasible`; at `eta >= 1e-5`, importing that terminal basis/status and
continuing dual simplex with the exact true cost also returns
`dual_infeasible`, not a certificate-backed optimum. No failing status was
accepted based on objective or primal residual alone.

## Falsifier and global rule

The mandate pre-registered two kill conditions:

1. kill if the best total-wall improvement, including recovery, is below 10%;
2. kill if recovery ever fails certificates.

The perturbation is one global, nested family with no problem-specific columns
or thresholds:

```text
c'_j = c_j + eta * max(1, abs(c_j), ||A[:,j]||_2) * u_j
```

`u` is a single PCG64 `U[-1,1]` vector from fixed seed `20260721`, shared by
every magnitude so the sweep isolates size. It is zeroed only where a column is
fixed (`lo_j == hi_j`). The swept magnitudes are `1e-12, 1e-10, 1e-9, 1e-8,
1e-7, 1e-6, 1e-5, 1e-4, 1e-3`.

Each arm follows the required exact-recovery sequence:

1. solve the perturbed reduced LP with the ordinary Dantzig dual-simplex path;
2. export its terminal basis and exact nonbasic bound-status vector;
3. import both into the ordinary dual-simplex path with the **true** cost;
4. allow that solve to recompute true-cost dual/reduced-cost information and
   continue pivoting until its normal exit;
5. charge both solver walls and every recovery pivot;
6. accept only `optimal` plus original-space equality, bound, and true-objective
   checks at fixed `eps=2e-5`.

The reduced fixture is 1,525 x 3,868 with 23,274 nonzeros. Solver tolerance is
`1e-8`, EXPAND is enabled, leaving rule is Dantzig, and BFRT is off, matching the
4,399-pivot control. There are five deterministic repeats per magnitude. Every
candidate is paired with a fresh cold control, and candidate/control order is
alternated to expose cache or thermal ordering bias.

## Measurements

Walls are local foreground measurements. `total wall` is perturbed solve plus
true-cost recovery. `paired ratio` is the median of the five within-pair ratios,
which is the comparison used for the verdict.

| eta | max abs delta c | perturbed status | recovery status | pivots + recovery | perturbed wall | recovery wall | total wall | paired ratio | recovery certificate |
|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| 1e-12 | 9.93e-11 | optimal | optimal | 4,487 + 1 | 0.5550s | 0.0061s | 0.5617s | 1.027538 | pass |
| **1e-10** | **9.93e-9** | **optimal** | **optimal** | **4,400 + 2** | **0.5325s** | **0.0057s** | **0.5382s** | **1.000468** | **pass** |
| 1e-9 | 9.93e-8 | optimal | optimal | 5,019 + 4 | 0.6167s | 0.0061s | 0.6228s | 1.157061 | pass |
| 1e-8 | 9.93e-7 | dual_infeasible | optimal | 5,225 + 5 | 0.6225s | 0.0056s | 0.6293s | 1.143478 | pass |
| 1e-7 | 9.93e-6 | dual_infeasible | optimal | 5,487 + 4 | 0.6261s | 0.0054s | 0.6314s | 1.160954 | pass |
| 1e-6 | 9.93e-5 | dual_infeasible | optimal | 5,567 + 5 | 0.6355s | 0.0067s | 0.6422s | 1.165694 | pass |
| 1e-5 | 9.93e-4 | dual_infeasible | dual_infeasible | 5,606 + 4 | 0.6493s | 0.0058s | 0.6553s | 1.254275 | **fail** |
| 1e-4 | 9.93e-3 | dual_infeasible | dual_infeasible | 5,930 + 4 | 0.6602s | 0.0076s | 0.6664s | 1.222151 | **fail** |
| 1e-3 | 9.93e-2 | dual_infeasible | dual_infeasible | 5,935 + 4 | 0.6622s | 0.0074s | 0.6675s | 1.230352 | **fail** |

Pivot counts and statuses are identical across all five repeats of every arm.
The 45 paired cold controls are all `optimal` in exactly 4,399 pivots; their
median wall is 0.53585s. Every control reproduces the reference terminal basis
and bound-status SHA-256 hashes exactly, so the perturbation-disabled path is
outcome-byte-identical. Every control passes the original-space certificate.

The best arm is not hiding a trajectory win behind recovery overhead: 4,400
perturbed pivots already exceed the 4,399-pivot control, and the two true-cost
cleanup pivots take it to 4,402. Every other arm costs 4,488 to 5,939 total
pivots. Thus the proposed anti-stall mechanism moves in the wrong direction on
the attacked `pivots` term.

At `eta=1e-5`, true-cost recovery's primal point is close to the reference
(true-objective relative difference `7.46e-14`, equality residual `2.01e-7`,
bound violation `4.55e-13`) but its status is `dual_infeasible`; primal
closeness is not an optimality certificate. At `eta=1e-4` and `1e-3`, recovery
also fails the objective gate by `1.39e-4` and `7.42e-4`, respectively.

## Flip arithmetic

Using the decisive paired ratio:

```text
best improvement = 1 - 1.000468 = -0.000468 = -0.047%
projected local wall from 0.370s = 0.370 * 1.000468 = 0.37017s
projected board ratio = 1.215 * 1.000468 = 1.21557
exact reduction needed to reach ratio 1.0 = 1 - 1/1.215 = 17.70% (~18%)
```

Cost perturbation therefore supplies none of the needed reduction. Even the
10% mandate threshold would require a paired ratio at or below 0.90; measured
best is 1.000468.

## Reproduction and audit

- Probe: `experiments/c4_cost_perturbation_probe.py`
- Raw JSON: `/tmp/c4-cost-perturbation/results.json`
- Required offline build completed with
  `UV_CACHE_DIR=/tmp/uv-cache UV_OFFLINE=1 uv sync --extra dev --no-build-isolation`.
- Probe ran foreground with `LINPROGX_DS_WARM_START=1` and
  `LINPROGX_DS_EXPORT_BASIS=1`.
- No C or solver files were read or changed, so the post-C-change editable
  reinstall was not applicable.
- No network access and no Git operations were used.

**Final verdict: KILLED** — the best valid total wall is flat/slower, all valid
tiny perturbations use at least as many pivots as cold, and larger magnitudes
trip the mandatory perturbed-solve and recovery certificate gates.
