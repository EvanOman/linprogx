# Oracle-pattern sparse whole-inverse falsifier — 2026-07-22

## Verdict

**KILLED in the tested scope.** The proposed fixed-`K` whole-inverse
sparsification has no point that is both decision-authoritative and funded on
the four actual `greenbea` bases.

- `K=16` is the only point below both favorable traffic gates before fallback,
  but it preserves the native Harris entering column in only **202/404** tests.
  Charging exact fallback at that observed 50% failure fraction raises its
  sampled-mean equivalent-pair charge to **35.842 us**, which can model only an
  **8.46%** whole-wall reduction versus the **17.7013%** board requirement.
- `K=32` also scores **202/404**, and its projected-update traffic alone is
  already **24.082 us/equivalent pair** on the four-sample mean, above both the
  **23.176 us** board ceiling and **20.028 us** probe ceiling. With observed
  decision fallback it reaches **47.795 us**.
- `K=64` scores **303/404**, but its mean traffic charge is **47.344 us** before
  fallback—essentially the whole current **47.425 us** pair.
- `K=128` is the only point to score **404/404**, but its mean traffic charge is
  **93.280 us**, nearly two current solve pairs, even though sparse-inverse
  application, pattern refresh, arithmetic, indices, residual checks, and
  certificates are all free in the model.

The earlier headline result is independently reproduced: Harris matches are
**202, 202, 303, and 404** for `K=16,32,64,128`, and maximum stored pattern
sizes are **45,595, 89,852, 176,975, and 343,857**. Its quoted **24.626 us**
(`K=32`) and **94.242 us** (`K=128`) figures are the sampled-maximum
equivalent-pair traffic charges after conversion across the audited 4,399
pivots / 4,873 equivalent pairs. The more favorable four-checkpoint means are
**24.082 us** and **93.280 us**. Neither is a full-trace bound.

No production code or test changed.

## Fixed construction and favorable oracle

At each captured exact basis `B`, the probe computes `M = B^-1`. For one global
`K` in `{16,32,64,128}`, `S_K` is the union of NumPy `argpartition`'s `K`
largest-magnitude entries in every row and every column. The approximate
inverse retains the exact coefficient `M_ij` on `S_K` and zero elsewhere.

This is more favorable than a deployable projected rank-one scheme:

- every sampled basis receives a newly oracle-selected pattern and exact
  coefficients for free;
- the experiment does not charge construction, refresh, or drift between
  refactorizations;
- tied exact-zero locations selected by `argpartition` remain stored pattern
  slots because a subsequent projected rank-one update can make them nonzero.

The NumPy 2.4.4 zero-tie selection reproduced identically across repeated runs
in this environment. The artifact also records how many selected coefficients
happen to be numerically nonzero at each capture, separately from stored
pattern slots.

The fixed solver policy is `eps=2e-5`, `tol=1e-8`, `leaving_rule=1`,
`expand=1`, and `bfrt=0`. There is no instance- or checkpoint-specific choice
of `K`.

## Native basis and next-pivot authority

The diagnostic runs the native solver to exact iteration caps 512, 1,536,
3,072, and 4,096, then repeats at `k+1`. All **8/8** prefixes return exactly
the requested iteration count. Every adjacent basis pair differs in exactly
one position, giving the actual next:

- BTRAN right-hand side `e_leaving`;
- FTRAN right-hand side: the newly entering scaled structural/logical column;
- native entering column against which Harris replay is checked.

The native 10-pass infinity-norm plus one l2 Ruiz scaling is reconstructed.
Each basis has 1,525 rows.

| checkpoint | basis nnz | leaving position | entering column | FTRAN RHS nnz |
|---:|---:|---:|---:|---:|
| 512 | 7,022 | 714 | 1,609 | 15 |
| 1,536 | 7,138 | 1,426 | 514 | 6 |
| 3,072 | 7,116 | 487 | 792 | 16 |
| 4,096 | 7,522 | 1,335 | 3,693 | 4 |

Exact inverse infinity-norm residuals range from `2.65e-13` to `1.39e-9`.
The exact BTRAN replay selects the native next entering column in **404/404**
tests while EXPAND tolerance sweeps 101 values over `[0,1e-8]` at each basis.

## Pattern size, residuals, and Harris decisions

Stored pattern slots reproduce the proposal exactly:

| K | checkpoint 512 | 1,536 | 3,072 | 4,096 | mean | max |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 42,113 | 43,980 | 45,339 | 45,595 | 44,256.75 | 45,595 |
| 32 | 84,592 | 87,700 | 89,326 | 89,852 | 87,867.50 | 89,852 |
| 64 | 169,750 | 170,440 | 173,796 | 176,975 | 172,740.25 | 176,975 |
| 128 | 343,004 | 334,353 | 340,174 | 343,857 | 340,347.00 | 343,857 |

The following cells are `FTRAN relative residual / BTRAN relative residual`:

| checkpoint | K=16 | K=32 | K=64 | K=128 |
|---:|---:|---:|---:|---:|
| 512 | 7.509e-1 / 3.039e-13 | 1.924e-2 / 3.039e-13 | 1.200e-14 / 3.039e-13 | 1.200e-14 / 3.039e-13 |
| 1,536 | 4.654e-16 / 5.276e0 | 2.021e-16 / 2.662e-1 | 1.643e-16 / 5.896e-5 | 1.580e-16 / 3.771e-14 |
| 3,072 | 1.025e0 / 1.247e1 | 1.911e0 / 1.545e0 | 8.267e-1 / 2.521e-1 | 1.401e-1 / 4.714e-1 |
| 4,096 | 2.522e0 / 2.751e1 | 1.214e0 / 1.086e1 | 6.024e-1 / 2.323e0 | 2.170e-1 / 1.025e-1 |

Only **0/4, 0/4, 1/4, and 2/4** checkpoints respectively have both residuals
at or below `eps=2e-5`. This residual evidence is deliberately not charged in
the favorable fallback arithmetic below; only observed Harris failures cause
fallback there.

| K | Harris matches | observed decision failures | residual-qualified checkpoints |
|---:|---:|---:|---:|
| 16 | 202/404 | 50% | 0/4 |
| 32 | 202/404 | 50% | 0/4 |
| 64 | 303/404 | 25% | 1/4 |
| 128 | 404/404 | 0% | 2/4 |

Thus `K=128`'s four decision matches do not constitute accurate FTRAN/BTRAN
solves under the fixed certificate tolerance. Conversely, the points cheap
enough to contemplate do not preserve even the sampled next-pivot authority.

## Complete favorable funding arithmetic

The current comparable FTRAN+BTRAN pair is **47.425 us** and the audited solve
share is **34.62%** of wall:

| target | required solve-pool reduction | allowed replacement pair |
|---|---:|---:|
| 17.7013% board gap | 51.1303% | **23.176 us** |
| 20% probe gate | 57.7701% | **20.028 us** |

The host-calibrated favorable traffic model is:

`update_us/pivot = 16 * N / 52.7e3`

`equivalent_pair_charge = update_us/pivot * 4399 / 4873`

Sixteen bytes grant exactly one eight-byte read and one eight-byte write per
stored coefficient. Reads of the two rank-one vectors, multiplication and
subtraction, sparse indices, applications of `M` and `M^T`, residual work,
pattern refresh, and certification are free. This is an optimistic model, not
a measured implementation and not a machine-independent lower-bound theorem.

| K | mean update us/pivot | mean pair charge | max pair charge | observed-fallback pair charge | modeled whole-wall reduction with fallback |
|---:|---:|---:|---:|---:|---:|
| 16 | 13.437 | 12.130 | 12.496 | 35.842 | 8.46% |
| 32 | 26.677 | 24.082 | 24.626 | 47.795 | -0.27% |
| 64 | 52.445 | 47.344 | 48.504 | 59.200 | -8.60% |
| 128 | 103.331 | 93.280 | 94.242 | 93.280 | -33.47% |

The fallback column adds the current exact pair at only the observed Harris
failure fraction. It ignores the much harsher residual-qualified fractions.
Even under that favorable treatment, no `K` funds either target.

The four-checkpoint mean is a favorable proxy, not a trajectory average. The
sampled maximum reproduces the proposing lane's quoted traffic figures, but
neither statistic proves a hardware lower bound or supplies full-trace pattern
sizes.

## Scope

This result closes the concrete construction tested here: a globally fixed
top-`K` row/column sparsification of the whole inverse, with exact retained
coefficients at refresh and projected rank-one maintenance between refreshes.
It is especially favorable to that construction because it grants a fresh
oracle pattern at each captured basis and does not simulate projection drift.

It does **not** prove that every sparse approximate inverse, learned pattern,
low-rank-plus-sparse representation, recycled Krylov space, or different basis
factorization is impossible. Such a candidate needs its own globally fixed
construction, refresh and update charge, application cost, decision authority,
and certificate behavior. It cannot inherit this oracle's free refreshes or
claim the `K=128` decision matches without also paying its traffic and residual
repair costs.

## Artifact and reproduction

- Probe: `experiments/rich_inverse_falsifier.py`
- Raw artifact: `/tmp/rich-inverse-falsifier/results.json`
- Artifact SHA-256:
  `933e62ebf22f10c54a8c865ccea1c26185f20d129a4fa0bdf9167939ac7b805c`
- Artifact mode: `0600`; parent directory mode: `0700`

The authoritative run reused the already-provisioned performance environment
with synchronization disabled, so it performed no package operation:

```bash
cd /home/evan/dev/linprogx-rich-inverse
UV_PROJECT_ENVIRONMENT=/home/evan/dev/linprogx-perf-worktree/.venv \
UV_NO_SYNC=1 \
UV_OFFLINE=1 \
UV_CACHE_DIR=/tmp/uv-cache \
OPENBLAS_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
uv run python -m experiments.rich_inverse_falsifier
```

## Validation and integrity

- Native prefix gates: **8/8** exact requested iteration counts.
- Adjacent basis-difference gates: **4/4** exactly one changed position.
- Exact Harris replay: **404/404**.
- Approximate Harris replay: **202/404**, **202/404**, **303/404**, and
  **404/404** for `K=16,32,64,128`.
- Fixed policy: `eps=2e-5`; no per-checkpoint or per-instance `K`.
- `ruff check`, `ruff format --check`, and `py_compile` pass on the probe.
- No Git command was run.
- No network tool, API, external solver source, or package download was used.
  An initial offline `uv run ruff` automatically created a local environment;
  it was moved intact to `/tmp/rich-inverse-autocreated-venv-20260722` and all
  authoritative commands used the pre-existing environment with sync disabled.
- No production source or test was edited. Repository writes are limited to
  this report and its standalone diagnostic probe.

**Final scoped verdict: KILLED.** The only point with full sampled Harris
authority is more expensive than the current exact pair before applications;
the only point that clears the favorable raw traffic gates loses half the
sampled decisions and cannot fund exact fallback.
