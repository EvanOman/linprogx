# LU↔DS permutation-boundary census (2026-07-25)

**Verdict: REOPENED, 6.09% bit-identically removable — but NOT where it was predicted.**
The predicted lever (sparse permute-in) is worth 1.01%. The actual mass is in the
two O(m) pattern rescans (5.00%), and the obvious cheap fix for those is
**KILLED on measurement**.

## Origin

`experiments/endgame_fresh_classes_2026_07_25.md` (worker D, this wave) ranked
"C-1 — eliminate the O(m) permutation and pattern-rescan passes at the LU↔DS
boundary" as its #1 candidate, estimating **10.1% of wall** from element-visit
accounting, and predeclared: **KILL if the four boundary streams together are
under 6% of wall.**

D's structural reading of `lu_ft_ftran` was verified line-by-line and is
correct. Two of its framing claims were also verified and matter:

- greenbea's DS factors the **presolved** model at **m = 1,525**, not the raw
  2,392 (confirmed: `slice_census_today.py` prints `presolved shape (1525, 3868)
  nnz=23274`).
- The dense-staged bodies are reached because the adaptive route at
  `_csparse.c` (`s_nnz * 4 > m * s_cnt`) sends this solve density to
  `lu_ft_ftran` / `lu_ft_btran`.

Decisive corroboration that this is real dead work and not a design necessity:
the **hyper-sparse GP path in the same file already uses the correct sparse
idiom** — the in-repo comment reads *"1. Permute sparse rhs:
`z[inv_perm_row[idx]] = val`"* — and `inv_perm_row` is built as the exact
inverse (`ctx->inv_perm_row[ctx->perm_row[k]] = k`). The dense-staged path
simply does not use it.

## Method

`LINPROGX_DS_PERM_CENSUS=1` brackets six whole loops with `__rdtsc()` — never
per element — plus a whole-solve TSC bracket. **Every stream is reported as a
fraction of total solve cycles, so the census is frequency-independent**: core
boosting under load cannot bias the shares. This matters because the box runs at
load ~11–12 of 12 cores.

Measured with the Harris unit disabled (`LINPROGX_DS_HARRIS_FASTPATH=0`) so the
two units are scored independently.

## Result (greenbea, 4,399 pivots, objective `-72555248.12984592`)

| stream | cyc/call | calls | share of solve |
|---|---:|---:|---:|
| ftran_pattern_scan | 16,461 | 2,583 | **3.161%** |
| btran_pattern_scan | 9,694 | 2,557 | **1.843%** |
| ftran_permute_out | 2,884 | 3,459 | 0.788% |
| ftran_permute_in | 2,151 | 3,459 | 0.587% |
| btran_permute_in | 2,083 | 2,557 | 0.420% |
| btran_permute_out | 1,739 | 2,557 | 0.351% |
| **ALL BOUNDARY** | | | **7.329%** |
| **BIT-IDENTICALLY REMOVABLE** | | | **6.092%** |

Reproduced twice (6.066% / 6.092%). "Removable" counts both permute-ins (sparse
rhs) and both pattern scans (collect in-sweep); the permute-outs move a
dense-ish result and are **not** counted.

**This clears D's 6% kill bar — but only just, and D's own ranking of the
sub-items is wrong.** D led with the permute-ins, reasoning that FTRAN gathers
1,525 permuted entries for a 5–8-nonzero rhs and BTRAN does the same for a
*unit vector*. Both readings are correct, and both streams are real dead work —
but together they are **1.007%**, not the bulk. The mass is the rescans.

## Why the rescans are expensive, and the failed fix

The rescan is `for i<m: if (x[i] != 0.0) x_pattern[nnz++] = i;` — a sequential
walk of a cache-resident 12 KB array costing **9.95 cycles/element**, far above
the ~2–3 expected. Hypothesis: the solve result is 59–94% dense, the zero
positions are scattered, so the data-dependent branch mispredicts often.

**Falsified.** A branchless predicated form (`x_pattern[nnz] = i; nnz += (x[i]
!= 0.0);` — bit-identical output, and bounds-safe because the store precedes
the increment and `nnz <= i`) measured **worse**:

| form | ftran_pattern_scan | btran_pattern_scan |
|---|---:|---:|
| branching (shipped) | 16,461 cyc/call | 9,694 |
| branchless predicated | 23,220 cyc/call | 16,236 |

15.2 vs 9.95 cyc/element. At 59–94% density the branch is **mostly taken and
predicts well**; the predicated form instead pays a store on every element plus
a loop-carried dependency on the cursor. **KILLED**, left in only as an opt-in
`LINPROGX_DS_BRANCHLESS_SCAN=1` gate, default off.

## Honest status of the remaining 6.09%

- **Permute-ins (1.007%)** — cheaply and safely removable by
  `memset(z,0,m*8)` + `n_rhs_nz` scatters through `inv_perm_row`, exactly the
  idiom the GP path already uses. Bit-identical. Low risk. Small.
- **Pattern rescans (5.004%)** — removable only by collecting the pattern
  *during* the U′ sweep. The sweep visits positions in `ft_order`, so the
  collected pattern would be in **factor order, not ascending index order**.
  Every downstream consumer of `x_pattern` must be audited for an ascending-order
  dependency before this is attempted. This is the real prize and the real risk;
  it is **not** claimed here.

Realistic near-term yield is therefore ~1%, with ~5% gated behind an
ordering audit — not the 10.1% the element-visit model predicted. Element
visits are a poor proxy for cycles when the streams differ in branch behaviour
and access pattern.

## Reproduction

```bash
cd /home/evan/dev/linprogx-harris-census
UV_CACHE_DIR=/tmp/uv-cache uv pip install --reinstall -e . --no-build-isolation
PYTHONPATH=. LINPROGX_DS_PERM_CENSUS=1 LINPROGX_DS_HARRIS_FASTPATH=0 \
  uv run python experiments/wholewall_census.py --repeats 1 --no-instrument
```

No network, no solver source read, no per-problem tuning, eps=2e-5 untouched.
No production behaviour changed by this census: all instrumentation is
env-gated and the branchless arm defaults off.

---

## Addendum (2026-07-25, later): ordering audit + permute-in status

### The 5.00% pattern rescan is NOT bit-identically removable — audit result

The report above listed the two O(m) pattern rescans (5.00% of solve cycles) as
"bit-identically removable, gated behind an ascending-order audit of every
`x_pattern` consumer". **That audit has now been done and the answer is no.**

`lu_btran_sparse`'s pattern output is `rho_nz_rows`, and its consumers include
the pivot-row scatter (`_csparse.c` ~14278 / ~14305):

```c
for (int32_t ri = 0; ri < rho_nnz; ri++) {
    int32_t row = rho_nz_rows[ri];
    double rho_val = rho[row];
    for (p in CSR row) alpha_scratch[col] += rho_val * scaled_csr_data[p];
}
```

`alpha_scratch[col]` is a floating-point **accumulator**, and FP addition is not
associative. Collecting the pattern during the U′ sweep would deliver it in
`ft_order` (factor) order rather than ascending index order, changing the
accumulation order and therefore the **bits** of `alpha_scratch` — which feeds
the Harris ratio test and so the trajectory.

So this is a **certifiable trajectory change, not a free bit-identical win**.
That is a materially higher bar: it needs objective-agreement and iteration-count
evidence, exactly like the block-row uplook ship (which was accepted at
`obj reldiff <= 2.8e-12, iterations identical`). Sorting the pattern to restore
ascending order is not an escape: the code already records that a per-pivot
qsort was measured at **+200us/pivot on greenbea** — a net loss.

Corrected accounting of the 7.33% boundary: **0.44% now removed** (permute-ins,
below), **~2.29% remains bit-identically available** (permute-outs, harder),
and **5.00% is trajectory-changing**, not free.

### Sparse permute-in: implemented, bit-identical, effect below local resolution

Implemented for both dense-staged bodies (`lu_ft_ftran_ex` / `lu_ft_btran_ex`,
gate `LINPROGX_DS_SPARSE_PERMIN`, default on), using the idiom the hyper-sparse
GP path already uses and the inverse permutations already built.

Directly measured by the cycle census (frequency-independent, same run):

| stream | before | after |
|---|---:|---:|
| ftran_permute_in | 2,151 cyc/call | **1,211** (−44%) |
| btran_permute_in | 2,083 cyc/call | **591** (−72%) |
| combined share of solve | 1.007% | **0.441%** |

Bit-identical: 4,399 pivots, objective `-72555248.12984592`, residual 1.769e-07.

**But the whole-wall A/B is INCONCLUSIVE on this box**: 15 alternating pairs gave
`btran_rho` −2.54% against a worst control drift of **6.23%**. A ~0.5% whole-wall
effect is below the local resolution at load ~11-12/12 cores. The unit is sound
and free, but it is **not certified**, and it should be folded into the next
envab run rather than claimed from local numbers.
