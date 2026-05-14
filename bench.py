from __future__ import annotations

import os

from linprogx.compare import compare_with_clarabel, compare_with_scipy
from linprogx.samples import SAMPLES, STANDARD_BENCHMARKS, SampleProblem


def main() -> int:
    repeats = int(os.getenv("LINPROGX_BENCH_REPEATS", "20"))
    samples = _samples(os.getenv("LINPROGX_BENCH_SUITE", "all"))
    print(
        f"{'sample':24} {'solver':12} {'status':10} {'obj delta':>12} "
        f"{'linprogx ms':>12} {'solver ms':>10}"
    )
    print("-" * 92)
    for sample in samples:
        for comparison in (
            compare_with_scipy(sample.problem, repeats=repeats),
            compare_with_clarabel(sample.problem, repeats=repeats),
        ):
            delta = (
                "n/a" if comparison.objective_delta is None else f"{comparison.objective_delta:.2e}"
            )
            print(
                f"{sample.name:24} {comparison.solver_name:12} "
                f"{comparison.linprogx_status:10} {delta:>12} "
                f"{comparison.linprogx_seconds * 1000:12.3f} "
                f"{comparison.solver_seconds * 1000:10.3f}"
            )
    return 0


def _samples(suite: str) -> tuple[SampleProblem, ...]:
    if suite == "samples":
        return SAMPLES
    if suite == "standard":
        return STANDARD_BENCHMARKS
    if suite == "all":
        return (*SAMPLES, *STANDARD_BENCHMARKS)
    msg = "LINPROGX_BENCH_SUITE must be one of: all, samples, standard"
    raise ValueError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
