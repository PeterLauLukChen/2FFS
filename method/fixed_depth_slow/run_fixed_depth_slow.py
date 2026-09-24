#!/usr/bin/env python3
"""Run the fixed-depth expansion plus slow-confidence baseline on one tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixed_depth_slow.fixed_depth_slow import (
    FixedDepthSlowConfig,
    FixedDepthSlowRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expand-depth", type=int, default=None)
    parser.add_argument("--slow-cost", type=float, default=None)
    parser.add_argument("--slow-sigma", type=float, default=None)
    parser.add_argument("--max-rounds", type=int, default=500_000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FixedDepthSlowConfig(
        delta=args.delta,
        epsilon=args.epsilon,
        seed=args.seed,
        expand_depth=args.expand_depth,
        slow_cost=args.slow_cost,
        slow_sigma=args.slow_sigma,
        max_rounds=args.max_rounds,
        verbose=args.verbose,
    )
    result = FixedDepthSlowRunner.from_json_path(args.tree, config=config).run()
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
