#!/usr/bin/env python3
"""Validate generated two-fidelity minimax-tree environments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TOL = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def expected_player(depth_index: int, tree_depth: int) -> str:
    if depth_index == tree_depth:
        return "leaf"
    return "max" if depth_index % 2 == 0 else "min"


def expected_node_count(depth: int, branching: int) -> int:
    return (branching ** (depth + 1) - 1) // (branching - 1)


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def validate_tree(tree: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    depth = int(tree["depth"])
    branching = int(tree["branching"])
    nodes = {node["id"]: node for node in tree["nodes"]}
    root = tree["root"]

    if root != "r":
        errors.append(f"{path.name}: expected root id 'r', got {root!r}")
    if tree.get("node_count") != len(nodes):
        errors.append(f"{path.name}: node_count field does not match nodes length")
    if len(nodes) != expected_node_count(depth, branching):
        errors.append(
            f"{path.name}: expected {expected_node_count(depth, branching)} nodes, got {len(nodes)}"
        )

    leaf_ids = [node_id for node_id, node in nodes.items() if node["player"] == "leaf"]
    if len(leaf_ids) != branching**depth:
        errors.append(f"{path.name}: expected {branching**depth} leaves, got {len(leaf_ids)}")
    if tree.get("leaf_count") != len(leaf_ids):
        errors.append(f"{path.name}: leaf_count field does not match leaves")

    for node_id, node in nodes.items():
        node_depth = int(node["depth"])
        if node_depth < 0 or node_depth > depth:
            errors.append(f"{path.name}: node {node_id} has invalid depth {node_depth}")
        if int(node["remaining_depth"]) != depth - node_depth:
            errors.append(f"{path.name}: node {node_id} has inconsistent remaining_depth")
        player = node["player"]
        if player != expected_player(node_depth, depth):
            errors.append(
                f"{path.name}: node {node_id} at depth {node_depth} has player {player!r}"
            )
        children = node["children"]
        if player == "leaf":
            if children:
                errors.append(f"{path.name}: leaf {node_id} has children")
        else:
            if len(children) != branching:
                errors.append(f"{path.name}: node {node_id} has {len(children)} children")
            for child in children:
                if child not in nodes:
                    errors.append(f"{path.name}: child {child} of {node_id} is missing")
                elif int(nodes[child]["depth"]) != node_depth + 1:
                    errors.append(f"{path.name}: child {child} has wrong depth")

        if node_id == root:
            if node["fast_value"] is not None or node["fast_bound"] is not None:
                errors.append(f"{path.name}: root should not have a fast observation")
        else:
            bound = float(node["fast_bound"])
            fast_value = float(node["fast_value"])
            value = float(node["value"])
            if bound < -TOL:
                errors.append(f"{path.name}: node {node_id} has negative fast_bound")
            if abs(fast_value - value) > bound + TOL:
                errors.append(f"{path.name}: node {node_id} violates fast bias guarantee")
            if not close(float(node["slow_mean"]), value):
                errors.append(f"{path.name}: node {node_id} slow_mean differs from value")

    computed: dict[str, float] = {}
    for node_id in sorted(nodes, key=lambda x: int(nodes[x]["depth"]), reverse=True):
        node = nodes[node_id]
        if node["player"] == "leaf":
            computed[node_id] = float(node["value"])
        elif node["player"] == "max":
            computed[node_id] = max(computed[child] for child in node["children"])
        elif node["player"] == "min":
            computed[node_id] = min(computed[child] for child in node["children"])
        else:
            errors.append(f"{path.name}: node {node_id} has unknown player {node['player']!r}")
            continue
        if node_id in computed and not close(computed[node_id], float(node["value"])):
            errors.append(f"{path.name}: minimax value mismatch at {node_id}")

    root_children = list(tree["root_children"])
    if root_children != nodes[root]["children"]:
        errors.append(f"{path.name}: root_children differs from root.children")
    if root_children:
        root_values = {child: computed[child] for child in root_children}
        optimal = max(root_values, key=root_values.get)
        if tree["optimal_root_child"] != optimal:
            errors.append(f"{path.name}: optimal_root_child is incorrect")
        sorted_values = sorted(root_values.values(), reverse=True)
        gap = sorted_values[0] - sorted_values[1]
        if not close(float(tree["root_gap"]), gap):
            errors.append(f"{path.name}: root_gap is incorrect")
        if not close(float(tree["root_value"]), computed[root]):
            errors.append(f"{path.name}: root_value is incorrect")

    if any(not math.isfinite(float(node["value"])) for node in nodes.values()):
        errors.append(f"{path.name}: non-finite node value")
    return errors


def main() -> None:
    args = parse_args()
    tree_files = sorted(args.env_dir.glob("tree_*.json"))
    if args.expected_count is not None and len(tree_files) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} tree files under {args.env_dir}, found {len(tree_files)}"
        )
    if not tree_files:
        raise FileNotFoundError(f"no tree_*.json files under {args.env_dir}")

    all_errors: list[str] = []
    depths: set[int] = set()
    branchings: set[int] = set()
    for tree_path in tree_files:
        tree = json.loads(tree_path.read_text())
        depths.add(int(tree["depth"]))
        branchings.add(int(tree["branching"]))
        all_errors.extend(validate_tree(tree, tree_path))

    if all_errors:
        for error in all_errors:
            print(error)
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "env_dir": str(args.env_dir),
                "num_trees": len(tree_files),
                "depths": sorted(depths),
                "branchings": sorted(branchings),
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
