#!/usr/bin/env python3
"""Generate simulated two-fidelity minimax tree environments.

The generated trees are full alternating Max/Min trees with known minimax
values.  Each non-root node stores a deterministic fast-oracle value satisfying
the paper's depth-dependent bias envelope and a slow-oracle mean equal to the
true minimax value.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


SCHEMA_VERSION = "2ffs-minimax-tree-v1"

DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "num_trees": 100,
    "depth": 2,
    "branching": 3,
    "seed": 20260424,
    "beta": 0.05,
    "slow_cost": 10.0,
    "slow_sigma": 0.25,
    "root_value_low": 0.45,
    "root_value_high": 0.65,
    "root_gap_low": 0.03,
    "root_gap_high": 0.08,
    "internal_gap_low": 0.03,
    "internal_gap_high": 0.18,
    "value_floor": 0.05,
    "value_ceiling": 0.95,
}

KEY_ALIASES = {
    "d": "depth",
    "b": "branching",
    "n": "num_trees",
    "num": "num_trees",
    "trees": "num_trees",
    "out": "out_dir",
    "output": "out_dir",
    "slow_cost": "slow_cost",
    "slow_sigma": "slow_sigma",
    "root_gap": "root_gap_high",
    "internal_gap": "internal_gap_high",
}


def next_player(player: str) -> str:
    if player == "max":
        return "min"
    if player == "min":
        return "max"
    raise ValueError(f"unknown player: {player}")


def bias_bound(remaining_depth: int, depth: int, beta: float) -> float:
    if remaining_depth <= 0:
        return 0.0
    return beta * remaining_depth / depth


def rounded(x: float) -> float:
    return round(float(x), 10)


class TreeBuilder:
    def __init__(
        self,
        *,
        rng: random.Random,
        depth: int,
        branching: int,
        beta: float,
        internal_gap_range: tuple[float, float],
        value_floor: float,
        value_ceiling: float,
    ) -> None:
        self.rng = rng
        self.depth = depth
        self.branching = branching
        self.beta = beta
        self.internal_gap_range = internal_gap_range
        self.value_floor = value_floor
        self.value_ceiling = value_ceiling
        self.nodes: dict[str, dict[str, Any]] = {}

    def margin(self) -> float:
        lo, hi = self.internal_gap_range
        return self.rng.uniform(lo, hi)

    def clamp(self, x: float) -> float:
        return min(self.value_ceiling, max(self.value_floor, x))

    def add_node(
        self,
        *,
        node_id: str,
        depth_index: int,
        player: str,
        value: float,
        children: list[str],
    ) -> None:
        remaining_depth = self.depth - depth_index
        node: dict[str, Any] = {
            "id": node_id,
            "depth": depth_index,
            "remaining_depth": remaining_depth,
            "player": player,
            "children": children,
            "value": rounded(value),
            "slow_mean": rounded(value),
        }
        if node_id == "r":
            node["fast_bound"] = None
            node["fast_bias"] = None
            node["fast_value"] = None
        else:
            bound = bias_bound(remaining_depth, self.depth, self.beta)
            fast_bias = self.rng.uniform(-bound, bound) if bound > 0 else 0.0
            node["fast_bound"] = rounded(bound)
            node["fast_bias"] = rounded(fast_bias)
            node["fast_value"] = rounded(value + fast_bias)
        self.nodes[node_id] = node

    def build_subtree(
        self,
        *,
        node_id: str,
        depth_index: int,
        player: str,
        target_value: float,
    ) -> None:
        if depth_index == self.depth:
            self.add_node(
                node_id=node_id,
                depth_index=depth_index,
                player="leaf",
                value=target_value,
                children=[],
            )
            return

        child_player = next_player(player)
        principal = self.rng.randrange(self.branching)
        child_values: list[float] = []
        for child_idx in range(self.branching):
            if child_idx == principal:
                child_values.append(target_value)
            elif player == "max":
                child_values.append(self.clamp(target_value - self.margin()))
            else:
                child_values.append(self.clamp(target_value + self.margin()))

        child_ids = [f"{node_id}_{child_idx}" for child_idx in range(self.branching)]
        for child_id, child_value in zip(child_ids, child_values):
            self.build_subtree(
                node_id=child_id,
                depth_index=depth_index + 1,
                player=child_player,
                target_value=child_value,
            )

        if player == "max":
            value = max(child_values)
        else:
            value = min(child_values)
        self.add_node(
            node_id=node_id,
            depth_index=depth_index,
            player=player,
            value=value,
            children=child_ids,
        )


def build_tree(
    *,
    tree_id: str,
    seed: int,
    depth: int,
    branching: int,
    beta: float,
    slow_cost: float,
    slow_sigma: float,
    root_value_range: tuple[float, float],
    root_gap_range: tuple[float, float],
    internal_gap_range: tuple[float, float],
    value_floor: float,
    value_ceiling: float,
) -> dict[str, Any]:
    rng = random.Random(seed)
    builder = TreeBuilder(
        rng=rng,
        depth=depth,
        branching=branching,
        beta=beta,
        internal_gap_range=internal_gap_range,
        value_floor=value_floor,
        value_ceiling=value_ceiling,
    )

    root_best_value = rng.uniform(*root_value_range)
    root_gap = rng.uniform(*root_gap_range)
    best_child = rng.randrange(branching)
    second_best_child = (best_child + 1 + rng.randrange(branching - 1)) % branching

    root_child_values: list[float] = []
    for child_idx in range(branching):
        if child_idx == best_child:
            value = root_best_value
        elif child_idx == second_best_child:
            value = root_best_value - root_gap
        else:
            value = root_best_value - root_gap - rng.uniform(*internal_gap_range)
        root_child_values.append(builder.clamp(value))

    root_child_ids = [f"a{child_idx}" for child_idx in range(branching)]
    for child_id, child_value in zip(root_child_ids, root_child_values):
        builder.build_subtree(
            node_id=child_id,
            depth_index=1,
            player="min",
            target_value=child_value,
        )

    root_value = max(root_child_values)
    builder.add_node(
        node_id="r",
        depth_index=0,
        player="max",
        value=root_value,
        children=root_child_ids,
    )

    sorted_root_values = sorted(root_child_values, reverse=True)
    actual_gap = sorted_root_values[0] - sorted_root_values[1]
    optimal_root_child = root_child_ids[root_child_values.index(sorted_root_values[0])]

    nodes = [builder.nodes[node_id] for node_id in sorted(builder.nodes)]
    leaves = [node["id"] for node in nodes if node["player"] == "leaf"]

    return {
        "schema_version": SCHEMA_VERSION,
        "tree_id": tree_id,
        "seed": seed,
        "depth": depth,
        "branching": branching,
        "root": "r",
        "root_player": "max",
        "root_children": root_child_ids,
        "optimal_root_child": optimal_root_child,
        "root_value": rounded(root_value),
        "root_gap": rounded(actual_gap),
        "leaf_count": len(leaves),
        "node_count": len(nodes),
        "bias_envelope": {
            "type": "linear_remaining_depth",
            "beta": beta,
            "B_h": {
                str(h): rounded(bias_bound(h, depth, beta))
                for h in range(depth + 1)
            },
        },
        "fast_oracle": {
            "type": "deterministic_biased",
            "cost": 1.0,
            "guarantee": "|fast_value - value| <= fast_bound",
        },
        "slow_oracle": {
            "type": "gaussian_subgaussian",
            "mean": "node.value",
            "sigma": slow_sigma,
            "cost": slow_cost,
        },
        "nodes": nodes,
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=None,
        help="Optional JSON file describing multiple datasets to generate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Single-run output directory, or sweep base output directory.",
    )
    parser.add_argument("--num-trees", type=int, default=DEFAULT_RUN_CONFIG["num_trees"])
    parser.add_argument("--depth", type=int, default=DEFAULT_RUN_CONFIG["depth"])
    parser.add_argument("--branching", type=int, default=DEFAULT_RUN_CONFIG["branching"])
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN_CONFIG["seed"])
    parser.add_argument("--beta", type=float, default=DEFAULT_RUN_CONFIG["beta"])
    parser.add_argument("--slow-cost", type=float, default=DEFAULT_RUN_CONFIG["slow_cost"])
    parser.add_argument("--slow-sigma", type=float, default=DEFAULT_RUN_CONFIG["slow_sigma"])
    parser.add_argument("--root-value-low", type=float, default=DEFAULT_RUN_CONFIG["root_value_low"])
    parser.add_argument("--root-value-high", type=float, default=DEFAULT_RUN_CONFIG["root_value_high"])
    parser.add_argument("--root-gap-low", type=float, default=DEFAULT_RUN_CONFIG["root_gap_low"])
    parser.add_argument("--root-gap-high", type=float, default=DEFAULT_RUN_CONFIG["root_gap_high"])
    parser.add_argument("--internal-gap-low", type=float, default=DEFAULT_RUN_CONFIG["internal_gap_low"])
    parser.add_argument("--internal-gap-high", type=float, default=DEFAULT_RUN_CONFIG["internal_gap_high"])
    parser.add_argument("--value-floor", type=float, default=DEFAULT_RUN_CONFIG["value_floor"])
    parser.add_argument("--value-ceiling", type=float, default=DEFAULT_RUN_CONFIG["value_ceiling"])
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help=(
            "Progress update interval in trees per dataset. Use 0 for "
            "automatic TTY/log-friendly behavior."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.depth < 1:
        raise ValueError("--depth must be at least 1")
    if args.branching < 2:
        raise ValueError("--branching must be at least 2")
    if args.num_trees < 1:
        raise ValueError("--num-trees must be positive")
    if args.beta < 0:
        raise ValueError("--beta must be nonnegative")
    if args.slow_cost <= 0:
        raise ValueError("--slow-cost must be positive")
    if args.slow_sigma <= 0:
        raise ValueError("--slow-sigma must be positive")
    if args.value_floor >= args.value_ceiling:
        raise ValueError("--value-floor must be smaller than --value-ceiling")
    if not (args.value_floor <= args.root_value_low < args.root_value_high <= args.value_ceiling):
        raise ValueError("root value range must lie inside the value range")
    if not (0.0 < args.root_gap_low <= args.root_gap_high):
        raise ValueError("root gap range must be positive and ordered")
    if not (0.0 < args.internal_gap_low <= args.internal_gap_high):
        raise ValueError("internal gap range must be positive and ordered")
    if args.root_value_low - args.root_gap_high < args.value_floor:
        raise ValueError(
            "root value/gap ranges can force clamping of the second-best root child"
        )
    if args.root_value_low - args.root_gap_high - args.internal_gap_high < args.value_floor:
        raise ValueError(
            "root and internal gap ranges can force clamping of noncompetitive root children"
        )


def full_tree_node_count(depth: int, branching: int) -> int:
    return (branching ** (depth + 1) - 1) // (branching - 1)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def progress_interval(total: int, args: argparse.Namespace) -> int:
    if args.progress_every > 0:
        return args.progress_every
    if sys.stderr.isatty():
        return 1
    return max(1, total // 100)


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "." * width + "]"
    filled = min(width, int(width * done / total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def report_progress(
    *,
    dataset_name: str,
    dataset_index: int,
    num_datasets: int,
    done: int,
    total: int,
    start_time: float,
    final: bool = False,
) -> None:
    elapsed = time.perf_counter() - start_time
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    percent = 100.0 * done / total if total else 100.0
    message = (
        f"dataset {dataset_index}/{num_datasets} {dataset_name:<22} "
        f"{progress_bar(done, total)} {done}/{total} "
        f"({percent:5.1f}%) elapsed {format_duration(elapsed)} "
        f"eta {format_duration(remaining)}"
    )
    if sys.stderr.isatty() and not final:
        print(message.ljust(120), file=sys.stderr, end="\r", flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.out_dir.glob("tree_*.json")) + [args.out_dir / "manifest.json"]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{args.out_dir} already contains generated files; pass --overwrite"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    trees: list[dict[str, Any]] = []
    tree_files: list[str] = []
    id_width = max(3, len(str(args.num_trees - 1)))
    dataset_name = getattr(args, "dataset_name", None) or f"d{args.depth}"
    dataset_index = int(getattr(args, "dataset_index", 1))
    num_datasets = int(getattr(args, "num_datasets", 1))
    start_time = time.perf_counter()
    interval = progress_interval(args.num_trees, args)
    if not args.no_progress:
        report_progress(
            dataset_name=dataset_name,
            dataset_index=dataset_index,
            num_datasets=num_datasets,
            done=0,
            total=args.num_trees,
            start_time=start_time,
        )
    for idx in range(args.num_trees):
        tree_id = f"d{args.depth}_{idx:0{id_width}d}"
        tree_seed = args.seed + idx
        tree = build_tree(
            tree_id=tree_id,
            seed=tree_seed,
            depth=args.depth,
            branching=args.branching,
            beta=args.beta,
            slow_cost=args.slow_cost,
            slow_sigma=args.slow_sigma,
            root_value_range=(args.root_value_low, args.root_value_high),
            root_gap_range=(args.root_gap_low, args.root_gap_high),
            internal_gap_range=(args.internal_gap_low, args.internal_gap_high),
            value_floor=args.value_floor,
            value_ceiling=args.value_ceiling,
        )
        filename = f"tree_{idx:0{id_width}d}.json"
        write_json(args.out_dir / filename, tree)
        trees.append(tree)
        tree_files.append(filename)
        done = idx + 1
        if not args.no_progress and (done == args.num_trees or done % interval == 0):
            report_progress(
                dataset_name=dataset_name,
                dataset_index=dataset_index,
                num_datasets=num_datasets,
                done=done,
                total=args.num_trees,
                start_time=start_time,
                final=done == args.num_trees,
            )

    root_gaps = [tree["root_gap"] for tree in trees]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "num_trees": args.num_trees,
        "depth": args.depth,
        "branching": args.branching,
        "seed": args.seed,
        "tree_files": tree_files,
        "generation": {
            "beta": args.beta,
            "slow_cost": args.slow_cost,
            "slow_sigma": args.slow_sigma,
            "root_value_range": [args.root_value_low, args.root_value_high],
            "root_gap_range": [args.root_gap_low, args.root_gap_high],
            "internal_gap_range": [args.internal_gap_low, args.internal_gap_high],
            "value_range": [args.value_floor, args.value_ceiling],
        },
        "summary": {
            "root_gap_min": rounded(min(root_gaps)),
            "root_gap_max": rounded(max(root_gaps)),
            "root_gap_mean": rounded(mean(root_gaps)),
            "node_count_per_tree": trees[0]["node_count"],
            "leaf_count_per_tree": trees[0]["leaf_count"],
            "expected_node_count_per_tree": full_tree_node_count(
                args.depth, args.branching
            ),
            "expected_leaf_count_per_tree": args.branching**args.depth,
        },
    }
    write_json(args.out_dir / "manifest.json", manifest)
    return manifest


def canonicalize_keys(config: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in config.items():
        normalized_key = key.replace("-", "_")
        canonical[KEY_ALIASES.get(normalized_key, normalized_key)] = value
    return canonical


def beta_tag(beta: float) -> str:
    text = f"{beta:.6g}"
    return text.replace(".", "p").replace("-", "m")


def default_dataset_name(config: dict[str, Any]) -> str:
    return (
        f"d{int(config['depth'])}-b{int(config['branching'])}"
        f"-beta{beta_tag(float(config['beta']))}"
    )


def run_config_to_namespace(config: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        out_dir=Path(config["out_dir"]),
        num_trees=int(config["num_trees"]),
        depth=int(config["depth"]),
        branching=int(config["branching"]),
        seed=int(config["seed"]),
        beta=float(config["beta"]),
        slow_cost=float(config["slow_cost"]),
        slow_sigma=float(config["slow_sigma"]),
        root_value_low=float(config["root_value_low"]),
        root_value_high=float(config["root_value_high"]),
        root_gap_low=float(config["root_gap_low"]),
        root_gap_high=float(config["root_gap_high"]),
        internal_gap_low=float(config["internal_gap_low"]),
        internal_gap_high=float(config["internal_gap_high"]),
        value_floor=float(config["value_floor"]),
        value_ceiling=float(config["value_ceiling"]),
        overwrite=bool(config["overwrite"]),
        dataset_name=config.get("dataset_name"),
        progress_every=int(config["progress_every"]),
        no_progress=bool(config["no_progress"]),
    )


def single_run_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir is None:
        raise ValueError("--out-dir is required when --sweep-config is not provided")
    config = {
        **DEFAULT_RUN_CONFIG,
        "out_dir": args.out_dir,
        "num_trees": args.num_trees,
        "depth": args.depth,
        "branching": args.branching,
        "seed": args.seed,
        "beta": args.beta,
        "slow_cost": args.slow_cost,
        "slow_sigma": args.slow_sigma,
        "root_value_low": args.root_value_low,
        "root_value_high": args.root_value_high,
        "root_gap_low": args.root_gap_low,
        "root_gap_high": args.root_gap_high,
        "internal_gap_low": args.internal_gap_low,
        "internal_gap_high": args.internal_gap_high,
        "value_floor": args.value_floor,
        "value_ceiling": args.value_ceiling,
        "progress_every": args.progress_every,
        "no_progress": args.no_progress,
        "overwrite": args.overwrite,
    }
    return config


def load_sweep_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sweep_config is None:
        return [single_run_config_from_args(args)]

    payload = json.loads(args.sweep_config.read_text())
    if isinstance(payload, list):
        defaults: dict[str, Any] = {}
        entries = payload
        base_out_dir = args.out_dir
    elif isinstance(payload, dict):
        defaults = canonicalize_keys(payload.get("defaults", {}))
        entries = (
            payload.get("sweep")
            or payload.get("sweeps")
            or payload.get("datasets")
            or payload.get("runs")
        )
        base_out_dir = args.out_dir or payload.get("base_out_dir") or payload.get("root_out_dir")
    else:
        raise ValueError("--sweep-config must contain a JSON object or list")

    if not isinstance(entries, list) or not entries:
        raise ValueError("sweep config must contain a nonempty sweep/datasets list")

    run_configs: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"sweep entry {index} must be a JSON object")
        entry = canonicalize_keys(raw_entry)
        merged = {
            **DEFAULT_RUN_CONFIG,
            **defaults,
            **entry,
            "overwrite": bool(entry.get("overwrite", defaults.get("overwrite", args.overwrite))),
            "progress_every": int(
                entry.get("progress_every", defaults.get("progress_every", args.progress_every))
            ),
            "no_progress": bool(
                entry.get("no_progress", defaults.get("no_progress", args.no_progress))
            ),
        }

        dataset_name = merged.get("dataset_name") or merged.get("name")
        if dataset_name is None:
            dataset_name = default_dataset_name(merged)
        merged["dataset_name"] = str(dataset_name)

        if "out_dir" in merged and merged["out_dir"] is not None:
            merged["out_dir"] = Path(merged["out_dir"])
        elif base_out_dir is not None:
            merged["out_dir"] = Path(base_out_dir) / merged["dataset_name"]
        else:
            raise ValueError(
                f"sweep entry {index} needs out_dir, or provide top-level/base --out-dir"
            )
        run_configs.append(merged)
    return run_configs


def main() -> None:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    run_configs = load_sweep_entries(args)

    sweep_summary: list[dict[str, Any]] = []
    for index, run_config in enumerate(run_configs, start=1):
        run_args = run_config_to_namespace(run_config)
        run_args.dataset_index = index
        run_args.num_datasets = len(run_configs)
        validate_args(run_args)
        manifest = generate_dataset(run_args)
        summary = {
            "index": index,
            "out_dir": str(run_args.out_dir),
            "dataset": manifest["dataset"],
            "num_trees": manifest["num_trees"],
            "depth": manifest["depth"],
            "branching": manifest["branching"],
            "beta": manifest["generation"]["beta"],
            "root_gap_mean": manifest["summary"]["root_gap_mean"],
            "node_count_per_tree": manifest["summary"]["node_count_per_tree"],
        }
        sweep_summary.append(summary)
        print(json.dumps({"generated": summary}, sort_keys=True))

    if args.sweep_config is not None:
        summary_path = args.sweep_config.with_suffix(".generated.json")
        write_json(
            summary_path,
            {
                "sweep_config": str(args.sweep_config),
                "num_datasets": len(sweep_summary),
                "datasets": sweep_summary,
            },
        )
        print(json.dumps({"sweep_summary": str(summary_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
