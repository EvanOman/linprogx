"""Driver for the close-six W2-B paired CPU-time campaign.

Runs ``ROUNDS`` interleaved rounds.  Every round is one fresh subprocess per
instance that solves every arm of that instance back to back, so all arms of a
round see the same host load; arm order rotates with the round index.  The
verdict statistic is the per-arm median CPU time plus a paired sign test
against the shipped arm over rounds.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ("lp_25fv47", "lp_degen2", "lp_greenbeb", "lp_greenbea")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default="/tmp/lpsuite")
    args = ap.parse_args()

    env = dict(os.environ)
    # Single-threaded BLAS in every arm: on a shared, loaded host CPU time is
    # the metric, and multi-threaded BLAS would charge parallel work to it.
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": str(ROOT),
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for rnd in range(args.rounds):
            for instance in INSTANCES:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "experiments" / "w2b_dse_cost.py"),
                        "round",
                        "--instance",
                        instance,
                        "--fixtures",
                        args.fixtures,
                        "--round",
                        str(rnd),
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=str(ROOT),
                )
                if proc.returncode != 0:
                    print(f"FAILED {instance} round {rnd}: {proc.stderr[-800:]}", flush=True)
                    continue
                for line in proc.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        fh.write(line + "\n")
                fh.flush()
            print(f"round {rnd} done", flush=True)


if __name__ == "__main__":
    main()
