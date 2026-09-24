#!/usr/bin/env python3
"""Run 2FFS on one generated minimax-tree JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twoffs.twoffs import TwoFFSConfig, TwoFFSRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True, help="Path to tree_*.json")
    parser.add_argument("--out", type=Path, default=None, help="Optional result JSON path")
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slow-cost", type=float, default=None)
    parser.add_argument("--slow-sigma", type=float, default=None)
    parser.add_argument("--max-outer-rounds", type=int, default=200_000)
    parser.add_argument("--max-scale", type=int, default=80)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TwoFFSConfig(
        delta=args.delta,
        epsilon=args.epsilon,
        seed=args.seed,
        slow_cost=args.slow_cost,
        slow_sigma=args.slow_sigma,
        max_outer_rounds=args.max_outer_rounds,
        max_scale=args.max_scale,
        verbose=args.verbose,
    )
    result = TwoFFSRunner.from_json_path(args.tree, config=config).run()
    payload = result.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
