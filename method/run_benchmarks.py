#!/usr/bin/env python3
"""Run all implemented tree-search methods on a directory of tree JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fast_minimax import FastMinimaxRunner
from fixed_depth_slow import FixedDepthSlowConfig, FixedDepthSlowRunner
from mcts_bai import MCTSBAIConfig, MCTSBAIRunner
from twoffs import TwoFFSConfig, TwoFFSRunner
from ugape_mcts import UGapEMCTSConfig, UGapEMCTSRunner


METHODS = ("twoffs", "fast_minimax", "mcts_bai", "ugape_mcts", "fixed_depth_slow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--twoffs-max-outer-rounds", type=int, default=200_000)
    parser.add_argument("--mcts-max-rounds", type=int, default=500_000)
    parser.add_argument("--ugape-max-rounds", type=int, default=500_000)
    parser.add_argument("--fixed-depth", type=int, default=None)
    parser.add_argument("--fixed-depth-max-rounds", type=int, default=500_000)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help=(
            "Progress update interval per method. Use 0 for automatic "
            "TTY/log-friendly behavior."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_method(method: str, tree_path: Path, tree_index: int, args: argparse.Namespace) -> dict[str, Any]:
    seed = args.base_seed + tree_index
    start = time.perf_counter()
    if method == "twoffs":
        config = TwoFFSConfig(
            delta=args.delta,
            epsilon=args.epsilon,
            seed=seed,
            max_outer_rounds=args.twoffs_max_outer_rounds,
        )
        result = TwoFFSRunner.from_json_path(tree_path, config=config).run().to_dict()
    elif method == "fast_minimax":
        result = FastMinimaxRunner.from_json_path(tree_path, epsilon=args.epsilon).run().to_dict()
    elif method == "mcts_bai":
        config = MCTSBAIConfig(
            delta=args.delta,
            epsilon=args.epsilon,
            seed=seed,
            max_rounds=args.mcts_max_rounds,
        )
        result = MCTSBAIRunner.from_json_path(tree_path, config=config).run().to_dict()
    elif method == "ugape_mcts":
        config = UGapEMCTSConfig(
            delta=args.delta,
            epsilon=args.epsilon,
            seed=seed,
            max_rounds=args.ugape_max_rounds,
        )
        result = UGapEMCTSRunner.from_json_path(tree_path, config=config).run().to_dict()
    elif method == "fixed_depth_slow":
        config = FixedDepthSlowConfig(
            delta=args.delta,
            epsilon=args.epsilon,
            seed=seed,
            expand_depth=args.fixed_depth,
            max_rounds=args.fixed_depth_max_rounds,
        )
        result = FixedDepthSlowRunner.from_json_path(tree_path, config=config).run().to_dict()
    else:
        raise ValueError(f"unknown method: {method}")
    elapsed = time.perf_counter() - start
    result["method"] = method
    result["tree_file"] = tree_path.name
    result["tree_index"] = tree_index
    result["seed"] = seed
    result["runtime_sec"] = round(elapsed, 6)
    add_work_metrics(result)
    return result


def add_work_metrics(result: dict[str, Any]) -> None:
    """Add paper-facing sample and BPU metrics to a method result.

    `sampling_count` is the total node-sampled/visited count Q. `computation_bpu_time`
    is the measured bookkeeping primitive-unit count T_BPU.  The legacy names are
    kept for backward compatibility with earlier experiment outputs.
    """
    fast_queries = result.get("num_fast_queries", 0) or 0
    slow_queries = result.get("num_slow_queries", 0) or 0
    sampling_count = result.get("num_nodes_sampled_visited")
    if sampling_count is None:
        sampling_count = fast_queries + slow_queries
        result["num_nodes_sampled_visited"] = sampling_count

    computation_bpu_time = result.get("bookkeeping_touches", 0) or 0
    result["sampling_count"] = sampling_count
    result["computation_bpu_time"] = computation_bpu_time
    result["adjusted_work"] = sampling_count + computation_bpu_time


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = finite_values(rows, key)
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "q25": None,
            "q75": None,
            "max": None,
        }
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "max": max(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    summary: dict[str, Any] = {}
    for method, method_rows in sorted(by_method.items()):
        n = len(method_rows)
        stopped = sum(bool(row.get("stopped")) for row in method_rows)
        correct = sum(bool(row.get("correct")) for row in method_rows)
        stopped_correct = sum(
            bool(row.get("stopped")) and bool(row.get("correct"))
            for row in method_rows
        )
        summary[method] = {
            "num_runs": n,
            "stopped": stopped,
            "not_stopped": n - stopped,
            "correct": correct,
            "incorrect": n - correct,
            "stopped_correct": stopped_correct,
            "stopped_rate": stopped / n if n else None,
            "correct_rate": correct / n if n else None,
            "stopped_correct_rate": stopped_correct / n if n else None,
            "total_cost": summarize_metric(method_rows, "total_cost"),
            "fast_cost": summarize_metric(method_rows, "fast_cost"),
            "slow_cost": summarize_metric(method_rows, "slow_cost"),
            "num_fast_queries": summarize_metric(method_rows, "num_fast_queries"),
            "num_slow_queries": summarize_metric(method_rows, "num_slow_queries"),
            "num_nodes_sampled_visited": summarize_metric(
                method_rows, "num_nodes_sampled_visited"
            ),
            "bookkeeping_touches": summarize_metric(method_rows, "bookkeeping_touches"),
            "sampling_count": summarize_metric(method_rows, "sampling_count"),
            "computation_bpu_time": summarize_metric(
                method_rows, "computation_bpu_time"
            ),
            "adjusted_work": summarize_metric(method_rows, "adjusted_work"),
            "num_expanded_nodes": summarize_metric(method_rows, "num_expanded_nodes"),
            "runtime_sec": summarize_metric(method_rows, "runtime_sec"),
        }
    return summary


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "method",
        "tree_index",
        "tree_id",
        "tree_file",
        "seed",
        "stopped",
        "correct",
        "recommendation",
        "optimal_root_child",
        "stop_reason",
        "root_gap",
        "total_cost",
        "fast_cost",
        "slow_cost",
        "num_fast_queries",
        "num_slow_queries",
        "num_nodes_sampled_visited",
        "bookkeeping_touches",
        "sampling_count",
        "computation_bpu_time",
        "adjusted_work",
        "num_expanded_nodes",
        "expand_depth",
        "frontier_size",
        "outer_rounds",
        "rounds",
        "runtime_sec",
    ]
    return {key: row.get(key, "") for key in keys}


def summary_metric(summary: dict[str, Any], metric: str, field: str) -> Any:
    value = summary.get(metric, {})
    if isinstance(value, dict):
        return value.get(field, "")
    return ""


def write_summary_table(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "method",
        "num_runs",
        "stopped",
        "not_stopped",
        "correct",
        "incorrect",
        "stopped_correct",
        "stopped_rate",
        "correct_rate",
        "stopped_correct_rate",
        "total_cost_mean",
        "total_cost_median",
        "total_cost_q25",
        "total_cost_q75",
        "total_cost_min",
        "total_cost_max",
        "fast_cost_mean",
        "slow_cost_mean",
        "num_fast_queries_mean",
        "num_slow_queries_mean",
        "num_nodes_sampled_visited_mean",
        "bookkeeping_touches_mean",
        "sampling_count_mean",
        "sampling_count_max",
        "computation_bpu_time_mean",
        "computation_bpu_time_max",
        "adjusted_work_mean",
        "adjusted_work_max",
        "num_expanded_nodes_mean",
        "runtime_sec_mean",
    ]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for method, method_summary in sorted(summary.items()):
            writer.writerow(
                {
                    "method": method,
                    "num_runs": method_summary.get("num_runs", ""),
                    "stopped": method_summary.get("stopped", ""),
                    "not_stopped": method_summary.get("not_stopped", ""),
                    "correct": method_summary.get("correct", ""),
                    "incorrect": method_summary.get("incorrect", ""),
                    "stopped_correct": method_summary.get("stopped_correct", ""),
                    "stopped_rate": method_summary.get("stopped_rate", ""),
                    "correct_rate": method_summary.get("correct_rate", ""),
                    "stopped_correct_rate": method_summary.get("stopped_correct_rate", ""),
                    "total_cost_mean": summary_metric(method_summary, "total_cost", "mean"),
                    "total_cost_median": summary_metric(method_summary, "total_cost", "median"),
                    "total_cost_q25": summary_metric(method_summary, "total_cost", "q25"),
                    "total_cost_q75": summary_metric(method_summary, "total_cost", "q75"),
                    "total_cost_min": summary_metric(method_summary, "total_cost", "min"),
                    "total_cost_max": summary_metric(method_summary, "total_cost", "max"),
                    "fast_cost_mean": summary_metric(method_summary, "fast_cost", "mean"),
                    "slow_cost_mean": summary_metric(method_summary, "slow_cost", "mean"),
                    "num_fast_queries_mean": summary_metric(
                        method_summary, "num_fast_queries", "mean"
                    ),
                    "num_slow_queries_mean": summary_metric(
                        method_summary, "num_slow_queries", "mean"
                    ),
                    "num_nodes_sampled_visited_mean": summary_metric(
                        method_summary, "num_nodes_sampled_visited", "mean"
                    ),
                    "bookkeeping_touches_mean": summary_metric(
                        method_summary, "bookkeeping_touches", "mean"
                    ),
                    "sampling_count_mean": summary_metric(
                        method_summary, "sampling_count", "mean"
                    ),
                    "sampling_count_max": summary_metric(
                        method_summary, "sampling_count", "max"
                    ),
                    "computation_bpu_time_mean": summary_metric(
                        method_summary, "computation_bpu_time", "mean"
                    ),
                    "computation_bpu_time_max": summary_metric(
                        method_summary, "computation_bpu_time", "max"
                    ),
                    "adjusted_work_mean": summary_metric(
                        method_summary, "adjusted_work", "mean"
                    ),
                    "adjusted_work_max": summary_metric(
                        method_summary, "adjusted_work", "max"
                    ),
                    "num_expanded_nodes_mean": summary_metric(
                        method_summary, "num_expanded_nodes", "mean"
                    ),
                    "runtime_sec_mean": summary_metric(method_summary, "runtime_sec", "mean"),
                }
            )


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
    method: str,
    method_idx: int,
    num_methods: int,
    done: int,
    total: int,
    method_start: float,
    final: bool = False,
) -> None:
    elapsed = time.perf_counter() - method_start
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    percent = 100.0 * done / total if total else 100.0
    message = (
        f"method {method_idx}/{num_methods} {method:<18} "
        f"{progress_bar(done, total)} {done}/{total} "
        f"({percent:5.1f}%) elapsed {format_duration(elapsed)} "
        f"eta {format_duration(remaining)}"
    )
    if sys.stderr.isatty() and not final:
        print(message.ljust(110), file=sys.stderr, end="\r", flush=True)
    else:
        print(message, file=sys.stderr, flush=True)


def main() -> None:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    tree_files = sorted(args.env_dir.glob("tree_*.json"))
    if not tree_files:
        raise FileNotFoundError(f"no tree_*.json files under {args.env_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_files = [
        args.out_dir / "benchmark_results.jsonl",
        args.out_dir / "benchmark_results.csv",
        args.out_dir / "summary_table.csv",
        args.out_dir / "summary.json",
        args.out_dir / "run_config.json",
    ]
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output files already exist: {existing}; pass --overwrite")

    config_payload = {
        "env_dir": str(args.env_dir),
        "out_dir": str(args.out_dir),
        "methods": args.methods,
        "num_trees": len(tree_files),
        "delta": args.delta,
        "epsilon": args.epsilon,
        "base_seed": args.base_seed,
        "twoffs_max_outer_rounds": args.twoffs_max_outer_rounds,
        "mcts_max_rounds": args.mcts_max_rounds,
        "ugape_max_rounds": args.ugape_max_rounds,
        "fixed_depth": args.fixed_depth,
        "fixed_depth_max_rounds": args.fixed_depth_max_rounds,
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n"
    )

    rows: list[dict[str, Any]] = []
    jsonl_path = args.out_dir / "benchmark_results.jsonl"
    csv_path = args.out_dir / "benchmark_results.csv"
    with jsonl_path.open("w") as jsonl_file:
        for method_idx, method in enumerate(args.methods, start=1):
            method_start = time.perf_counter()
            interval = progress_interval(len(tree_files), args)
            if not args.no_progress:
                report_progress(
                    method=method,
                    method_idx=method_idx,
                    num_methods=len(args.methods),
                    done=0,
                    total=len(tree_files),
                    method_start=method_start,
                )
            for tree_index, tree_path in enumerate(tree_files):
                row = run_method(method, tree_path, tree_index, args)
                rows.append(row)
                jsonl_file.write(json.dumps(row, sort_keys=True) + "\n")
                jsonl_file.flush()
                done = tree_index + 1
                if (
                    not args.no_progress
                    and (done == len(tree_files) or done % interval == 0)
                ):
                    report_progress(
                        method=method,
                        method_idx=method_idx,
                        num_methods=len(args.methods),
                        done=done,
                        total=len(tree_files),
                        method_start=method_start,
                        final=done == len(tree_files),
                    )
            method_summary = summarize([row for row in rows if row["method"] == method])
            print(json.dumps({method: method_summary[method]}, sort_keys=True))

    with csv_path.open("w", newline="") as csv_file:
        fieldnames = list(flatten_for_csv(rows[0]).keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))

    summary = summarize(rows)
    write_summary_table(args.out_dir / "summary_table.csv", summary)

    summary_payload = {
        "config": config_payload,
        "summary": summary,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
