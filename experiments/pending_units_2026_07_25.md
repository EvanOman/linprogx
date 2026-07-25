# Pending bit-identical units — measured, not yet certified (2026-07-25)

Two units that are exact, verified at vector level, and measurably positive, but
which post-date the certified sha `d50b01a` and are therefore **default OFF**.
They should ride the next envab run.

## 1. Sparse permute-in (`LINPROGX_DS_SPARSE_PERMIN`)

The dense-staged FT solve bodies placed their right-hand side with an O(m)
**permuted gather** regardless of rhs sparsity — FTRAN's rhs is an entering
column with ~5–8 nonzeros, and BTRAN's is a **unit vector**, yet both touched
all m=1,525 slots. Replaced with a sequential clear plus `n_rhs_nz` scatters
through `inv_perm_row` / `inv_perm_col` — the idiom the hyper-sparse GP path in
the same file already uses.

Cycle census (frequency-independent, same run):

| stream | before | after |
|---|---:|---:|
| ftran_permute_in | 2,151 cyc/call | **1,211** (−44%) |
| btran_permute_in | 2,083 cyc/call | **591** (−72%) |
| combined share of solve | 1.007% | **0.441%** |

## 2. L^T empty-column skip (`LINPROGX_DS_LT_SKIP`)

The `L^T` back solve ran **all m rows unconditionally**. L holds ~1,548–1,984
nonzeros across m=1,525 columns (~1.1 per column), so most columns are empty —
and for an empty column the accumulation loop is a no-op and the store is
`z[j] = z[j]`. Skipping is bit-identical.

A/B alone: `btran_rho` −2.07%, **0.39% of wall**, control drift 1.14%.

## Combined measurement

Alternating within-process A/B, 21 pairs, both gates toggled together:

| phase | B/A median |
|---|---:|
| btran_rho (treatment) | 0.9503 |
| pivot_row | 0.9888 |
| refactor | 0.9905 |
| lu_update | 1.0032 |
| ratio_test | 0.9924 |
| rcost_update | 0.9866 |
| TOTAL | 0.9752 |

**−4.97% on `btran_rho`, 3× the 1.61% worst control drift.** TOTAL reads
−2.48%, but the untouched phases also drifted ~1% in the same direction, so the
honest range is **~1–1.5% of wall**, not 2.5%.

## Bit-identity

Both verified with the vector-level trace oracle, not merely
objective+iterations: digest `679168a4baad36d6` over 6,016 solve output vectors,
4,399 iterations, objective `-72555248.12984592`, residual 1.769e-07 — identical
with the units on and off.

## Why they are default OFF

They were added after the certified sha `d50b01a`, whose envab run measured
**only** the Harris early-outs and the narrow CSR index cache. Shipping them
on would mean shipping an uncertified default. The certified configuration and
the shipped configuration must match.

## Running total against the board

| | value |
|---|---|
| certified (Harris + index cache) | **−4.89% on-host** |
| pending (these two, local) | ~1–1.5% |
| required | **−17.7013%** |

greenbea sits at ~1.156 certified. These do not change that; they are the next
increment to certify, not a flip.
