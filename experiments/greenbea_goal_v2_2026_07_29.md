# Greenbea GOAL V2 campaign (2026-07-29)

**PROVENANCE: SOURCE-INFORMED (HiGHS). Not a clean-room result.**

The DS2 pricing and bound-flipping components are independent
reimplementations informed by the HiGHS source study authorized in
`docs/PROVENANCE.md`. The equality aggregation is linprogx's pre-existing
clean-room reducer. HiGHS was otherwise used as an executable oracle and
benchmark opponent.

## Result

The board is **24W-0P-0L**. The production-default commit `152dab7` measures
greenbea at **0.986x HiGHS** under protocol v3 (three clean AWS us-west-2 hosts,
seven interleaved pairs per host). Every host median is below parity:

| host | linprogx / HiGHS | pair wins |
|---:|---:|---:|
| 0 | 0.995 | 5/7 |
| 1 | 0.942 | 6/7 |
| 2 | 0.986 | 4/7 |
| **median of hosts** | **0.986** | **15/21** |

The accepted solution is deterministic and certificate-backed in original
units:

- objective: `-72555248.12984599`;
- maximum original equality residual: `1.458272436138941e-08`;
- iterations: `2,424`;
- unchanged external acceptance tolerance: `eps=2e-5`.

Run:
<https://modal.com/apps/evan058/main/ap-2PCmH1FVOahtUWa5jDFRZK>

## Winning mechanism

Neither stronger presolve nor the DS2 composition wins alone. Their
interaction does.

1. Run the shipped presolve and the existing global IPM-stall classifier.
2. For a stall-risk equality model, probe the clean-room aggregation with a
   global Markowitz fill cap of 20.
3. Accept the fill-positive reduction only if it removes at least 20% of the
   remaining rows while growing realized matrix nnz by at most 5%.
4. Solve an accepted reduction with the completed DS2 composition: logical
   basis, exact DSE, sweep BFRT, power-of-two scaling, deterministic
   perturbation, and a 125-pivot refactorization cadence.
5. If the structural aggregation gate rejects, retain the shipped dual-simplex
   rescue unchanged.

On greenbea this changes the reduced model from
`1525 x 3868 x 23274` to `1188 x 3525 x 24045`: 22.1% fewer rows for 3.3%
more nnz. DS2 then needs 2,424 pivots, below HiGHS's 2,836. The same mechanism
helps the mandatory greenbeb control:

| instance | shipped pivots | candidate pivots | v3 candidate / shipped | pair wins |
|---|---:|---:|---:|---:|
| greenbea | 4,283 | 2,424 | 0.668 | 21/21 |
| greenbeb | 8,553 | 4,320 | 0.690 | 21/21 |

Run:
<https://modal.com/apps/evan058/main/ap-aInbt9GCL3giDJp11lb8Ia>

The gate is structural, not name- or problem-specific. Across the 14 local
fixtures matched by the raw stall signal, every candidate solution certified.
After comparing the normal and aggressive reductions, the 20% row / 5% nnz
exchange gate admits the green twins and `tuff`; the latter also improves
locally (221 to 159 pivots). Rejected cases keep their existing solver path.

## Why the interaction was previously missed

The two-way reduction exchange had shown that linprogx's shipped Dantzig solver
became `dual_infeasible` on HiGHS's 951-row greenbea reduction. Repeating the
same exchange with the completed DS2 composition reversed the result:

| model supplied to linprogx | greenbea | greenbeb |
|---|---:|---:|
| linprogx reduction | 3,476 | 5,232 |
| HiGHS reduction | **2,619** | **4,087** |

Both cells are optimal after postsolve and recertification in original units.
That experiment established that controlled fill-positive aggregation was
funded for DS2 even though it remained killed for shipped Dantzig.

## Campaign ledger

### A. Variance

Thirty shipped Dantzig permutations ranged from 3,998 to roughly 4,800
greenbea pivots; the best gain was only 6.7%. In the completed composition,
DSE leaving-row scan starts produced no variation at all (3,476/5,232 for all
30 seeds). BFRT tie permutations moved greenbea over 3,138–3,549 and greenbeb
over 4,783–6,429, but no fixed ordering approached closure while consistently
helping the control. Killed.

### B. Presolve exchange

HiGHS reduces greenbea to 951 rows versus linprogx's 1,525, but shipped
Dantzig fails on the exchanged reduction. The DS2 cross (2,619/4,087) exposed
the winning B×E interaction. Presolve alone remains killed; controlled
aggregation composed with DS2 ships.

### C. Alternate routes

IPM, PDHG, crossover hybrids, and the post-presolve/supernodal rerun all miss
the certification/time gate. Killed.

### D. Scaling

Geometric scaling worsened both twins. Power-of-two Ruiz scaling was the only
balanced composition member and ships inside DS2. Scaling alone does not close
the cell.

### E. DS2 composition and solve-cost attacks

The initial exact-DSE + BFRT + scaling composition reduced pivots but was
**12.9% slower** on greenbea (0/21) because exact DSE paid another FTRAN; it
improved greenbeb 6.3% (21/21).

- Sparse tau FTRAN: greenbea `1.206x`, greenbeb `1.237x`, 0/21 both. Killed.
- Fused two-RHS FTRAN: greenbea `0.908x` (21/21), but greenbeb `1.067x`
  (0/21) because the changed arithmetic trajectory harmed the control. Kept
  gated as research tooling, not shipped.
- Controlled aggregation + unfused exact DSE: greenbea `0.668x`, greenbeb
  `0.690x`, 21/21 both. Shipped.

Relevant Modal runs:

- composition: <https://modal.com/apps/evan058/main/ap-gk0HKVlA2RLrweOpL4ACF5>
- sparse tau: <https://modal.com/apps/evan058/main/ap-DuIdL3DsEb8oRRDj9rIaJ0>
- fused FTRAN: <https://modal.com/apps/evan058/main/ap-5kasRtHYt17ZUgiOjrQlY9>
- candidate paired: <https://modal.com/apps/evan058/main/ap-M6Mw9iwE2HcotMKmdLxhlB>

## Validation

- 24 LPnetlib native/Python aggregation equivalence cases:
  `17 passed, 7 skipped`.
- Focused DS2, sparse API, dual-simplex, and presolve suites:
  `188 passed`.
- The full repository gate is `just ci`; its final result is recorded in the
  campaign handoff commit.

Machine-readable summary:
`probe_out/greenbea-goal-v2-summary.json`.
