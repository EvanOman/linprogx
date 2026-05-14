from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from linprogx.solver import solve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve a linear program from a JSON file.")
    parser.add_argument(
        "path", type=Path, help="JSON file with c, A, b, senses, objective, and bounds"
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.path.read_text())
        result = solve(
            payload["c"],
            payload["A"],
            payload["b"],
            senses=payload.get("senses"),
            objective=payload.get("objective", "max"),
            bounds=_bounds(payload.get("bounds")),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"linprogx: {exc}", file=sys.stderr)
        return 2

    output: dict[str, Any] = {
        "status": result.status.value,
        "objective_value": result.objective_value,
        "x": result.x,
        "slacks": result.slacks,
        "iterations": result.iterations,
        "message": result.message,
    }
    print(json.dumps(output, indent=2 if args.pretty else None))
    return 0 if result.success else 1


def _bounds(raw: object) -> list[tuple[float | None, float | None]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        msg = "bounds must be a list"
        raise ValueError(msg)
    bounds = []
    for item in raw:
        if item is None:
            bounds.append((0.0, None))
            continue
        if not isinstance(item, list | tuple) or len(item) != 2:
            msg = "each bound must be [lower, upper]"
            raise ValueError(msg)
        bounds.append((item[0], item[1]))
    return bounds


if __name__ == "__main__":
    raise SystemExit(main())
