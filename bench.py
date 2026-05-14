from __future__ import annotations

from linprogx.compare import compare_with_clarabel, compare_with_scipy
from linprogx.samples import SAMPLES


def main() -> int:
    repeats = 20
    print(
        f"{'sample':24} {'solver':12} {'status':10} {'obj delta':>12} "
        f"{'linprogx ms':>12} {'solver ms':>10}"
    )
    print("-" * 92)
    for sample in SAMPLES:
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


if __name__ == "__main__":
    raise SystemExit(main())
