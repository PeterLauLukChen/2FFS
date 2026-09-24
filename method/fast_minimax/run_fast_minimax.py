#!/usr/bin/env python3
"""Run the fast-expansion-only minimax baseline on one tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fast_minimax.fast_minimax import FastMinimaxRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = FastMinimaxRunner.from_json_path(args.tree, epsilon=args.epsilon).run()
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(text, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
