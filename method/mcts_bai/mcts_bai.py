"""Slow-only LUCB-style BAI-MCTS baseline.

The baseline samples leaves with the slow oracle, propagates leaf confidence
intervals by minimax backup, and stops when the empirical best root child is
separated from its challenger.  It does not use fast-oracle values.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MCTSBAIConfig:
    delta: float = 0.05
    epsilon: float = 0.0
    seed: int = 0
    slow_cost: float | None = None
    slow_sigma: float | None = None
    max_rounds: int = 500_000
    verbose: bool = False


@dataclass
class LeafStats:
    n: int = 0
    total: float = 0.0
    low: float = -math.inf
    high: float = math.inf


@dataclass
class MCTSBAIResult:
    tree_id: str
    recommendation: str
    optimal_root_child: str
    correct: bool
    stopped: bool
    rounds: int
    total_cost: float
    slow_cost: float
    num_slow_queries: int
    bookkeeping_touches: int
    root_gap: float
    final_root_intervals: dict[str, tuple[float, float]]
    slow_queries_by_leaf: dict[str, int]
    stop_reason: str
    trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "recommendation": self.recommendation,
            "optimal_root_child": self.optimal_root_child,
            "correct": self.correct,
            "stopped": self.stopped,
            "rounds": self.rounds,
            "total_cost": self.total_cost,
            "fast_cost": 0.0,
            "slow_cost": self.slow_cost,
            "num_fast_queries": 0,
            "num_slow_queries": self.num_slow_queries,
            "num_nodes_sampled_visited": self.num_slow_queries,
            "bookkeeping_touches": self.bookkeeping_touches,
            "adjusted_work": self.num_slow_queries + self.bookkeeping_touches,
            "num_expanded_nodes": 0,
            "root_gap": self.root_gap,
            "final_root_intervals": self.final_root_intervals,
            "slow_queries_by_leaf": self.slow_queries_by_leaf,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
        }


class MCTSBAIRunner:
    def __init__(self, tree: dict[str, Any], config: MCTSBAIConfig | None = None) -> None:
        self.tree = tree
        self.config = config or MCTSBAIConfig()
        self.validate_config()
        self.rng = random.Random(self.config.seed)
        self.nodes = {node["id"]: node for node in tree["nodes"]}
        self.parent: dict[str, str] = {}
        for node in tree["nodes"]:
            for child in node.get("children", []):
                self.parent[child] = node["id"]
        self.root = tree["root"]
        self.root_children = list(tree["root_children"])
        self.leaves = sorted(
            node_id for node_id, node in self.nodes.items() if node["player"] == "leaf"
        )
        self.slow_cost_unit = float(
            self.config.slow_cost
            if self.config.slow_cost is not None
            else tree["slow_oracle"]["cost"]
        )
        self.slow_sigma = float(
            self.config.slow_sigma
            if self.config.slow_sigma is not None
            else tree["slow_oracle"]["sigma"]
        )
        self.leaf_delta = self.config.delta / max(1, len(self.leaves))
        self.stats = {leaf: LeafStats() for leaf in self.leaves}
        self.intervals: dict[str, tuple[float, float]] = {}
        self.total_cost = 0.0
        self.num_slow_queries = 0
        self.bookkeeping_touches = 0
        self.trace: list[dict[str, Any]] = []

    def touch(self, count: int = 1) -> None:
        """Count O(1) bookkeeping primitive units under a shared rule."""
        if count > 0:
            self.bookkeeping_touches += count

    def scan_touches(self, n: int, scans: int = 1) -> None:
        """Charge scalar endpoint reads plus comparisons for scans."""
        if n <= 0 or scans <= 0:
            return
        self.touch(scans * (n + max(0, n - 1)))

    def interval_write_touches(self) -> None:
        self.touch(2)

    def validate_config(self) -> None:
        if not (0.0 < self.config.delta < 1.0):
            raise ValueError("delta must lie in (0, 1)")
        if self.config.epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if self.config.max_rounds < 0:
            raise ValueError("max_rounds must be nonnegative")
        if self.config.slow_cost is not None and self.config.slow_cost <= 0:
            raise ValueError("slow_cost must be positive when provided")
        if self.config.slow_sigma is not None and self.config.slow_sigma <= 0:
            raise ValueError("slow_sigma must be positive when provided")

    @classmethod
    def from_json_path(
        cls, path: str | Path, config: MCTSBAIConfig | None = None
    ) -> "MCTSBAIRunner":
        return cls(json.loads(Path(path).read_text()), config=config)

    def run(self) -> MCTSBAIResult:
        self.initialize()
        rounds = 0
        stop_reason = "separated"
        while not self.is_stopped():
            if rounds >= self.config.max_rounds:
                stop_reason = "max_rounds"
                break
            self.scan_touches(len(self.root_children))
            leader = max(self.root_children, key=lambda child: self.intervals[child][0])
            self.scan_touches(max(0, len(self.root_children) - 1))
            challenger = max(
                [child for child in self.root_children if child != leader],
                key=lambda child: self.intervals[child][1],
            )
            leaves = {
                self.critical_leaf(leader, "L"),
                self.critical_leaf(challenger, "U"),
            }
            for leaf in sorted(leaves):
                self.sample_leaf(leaf)
            self.refresh_from_leaves(leaves)
            rounds += 1
            if self.config.verbose:
                self.trace.append(
                    {
                        "round": rounds,
                        "leader": leader,
                        "challenger": challenger,
                        "sampled_leaves": sorted(leaves),
                        "total_cost": self.total_cost,
                    }
                )

        rec = self.recommendation()
        return MCTSBAIResult(
            tree_id=self.tree["tree_id"],
            recommendation=rec,
            optimal_root_child=self.tree["optimal_root_child"],
            correct=rec == self.tree["optimal_root_child"],
            stopped=self.is_stopped(),
            rounds=rounds,
            total_cost=round(self.total_cost, 10),
            slow_cost=round(self.total_cost, 10),
            num_slow_queries=self.num_slow_queries,
            bookkeeping_touches=self.bookkeeping_touches,
            root_gap=float(self.tree["root_gap"]),
            final_root_intervals={child: self.intervals[child] for child in self.root_children},
            slow_queries_by_leaf={
                leaf: st.n for leaf, st in self.stats.items() if st.n > 0
            },
            stop_reason=stop_reason,
            trace=self.trace,
        )

    def initialize(self) -> None:
        for leaf in self.leaves:
            self.sample_leaf(leaf)
        self.refresh_all()

    def sample_leaf(self, leaf: str) -> None:
        node = self.nodes[leaf]
        sample = self.rng.gauss(float(node["value"]), self.slow_sigma)
        st = self.stats[leaf]
        st.n += 1
        st.total += sample
        empirical = st.total / st.n
        rad = self.radius(st.n)
        st.low = empirical - rad
        st.high = empirical + rad
        self.touch(2)  # n and running total updates.
        self.interval_write_touches()
        self.total_cost += self.slow_cost_unit
        self.num_slow_queries += 1

    def radius(self, n: int) -> float:
        log_term = math.log((math.pi * math.pi * n * n) / (6.0 * self.leaf_delta))
        return self.slow_sigma * math.sqrt(2.0 * log_term / n)

    def refresh_all(self) -> None:
        for leaf in self.leaves:
            st = self.stats[leaf]
            self.intervals[leaf] = (st.low, st.high)
            self.interval_write_touches()
        for node_id in sorted(self.nodes, key=lambda x: self.nodes[x]["depth"], reverse=True):
            if self.nodes[node_id]["player"] == "leaf":
                continue
            self.intervals[node_id] = self.backup_children(node_id)
            self.interval_write_touches()

    def refresh_from_leaves(self, leaves: set[str]) -> None:
        """Refresh exactly the nodes whose backed-up intervals can change."""
        affected: set[str] = set()
        for leaf in leaves:
            cur = leaf
            while True:
                affected.add(cur)
                if cur == self.root:
                    break
                cur = self.parent[cur]
        for node_id in sorted(affected, key=lambda x: self.nodes[x]["depth"], reverse=True):
            if self.nodes[node_id]["player"] == "leaf":
                st = self.stats[node_id]
                self.intervals[node_id] = (st.low, st.high)
                self.interval_write_touches()
            else:
                self.intervals[node_id] = self.backup_children(node_id)
                self.interval_write_touches()

    def backup_children(self, node_id: str) -> tuple[float, float]:
        children = self.nodes[node_id]["children"]
        self.scan_touches(len(children), scans=2)
        if self.nodes[node_id]["player"] == "max":
            return (
                max(self.intervals[child][0] for child in children),
                max(self.intervals[child][1] for child in children),
            )
        if self.nodes[node_id]["player"] == "min":
            return (
                min(self.intervals[child][0] for child in children),
                min(self.intervals[child][1] for child in children),
            )
        raise ValueError(f"leaf has no children: {node_id}")

    def is_stopped(self) -> bool:
        m = len(self.root_children)
        self.touch(m)
        self.scan_touches(max(0, m - 1), scans=m)
        return any(
            self.intervals[child][0]
            >= max(self.intervals[other][1] for other in self.root_children if other != child)
            - self.config.epsilon
            for child in self.root_children
        )

    def recommendation(self) -> str:
        m = len(self.root_children)
        self.touch(m)
        self.scan_touches(max(0, m - 1), scans=m)
        separated = [
            child
            for child in self.root_children
            if self.intervals[child][0]
            >= max(self.intervals[other][1] for other in self.root_children if other != child)
            - self.config.epsilon
        ]
        if separated:
            self.scan_touches(len(separated))
            return max(separated, key=lambda child: self.intervals[child][0])
        self.scan_touches(len(self.root_children))
        return max(self.root_children, key=lambda child: self.intervals[child][0])

    def critical_leaf(self, node_id: str, side: str) -> str:
        node = self.nodes[node_id]
        if node["player"] == "leaf":
            return node_id
        children = node["children"]
        self.scan_touches(len(children))
        if node["player"] == "max":
            if side == "L":
                child = max(children, key=lambda c: self.intervals[c][0])
            else:
                child = max(children, key=lambda c: self.intervals[c][1])
        elif node["player"] == "min":
            if side == "L":
                child = min(children, key=lambda c: self.intervals[c][0])
            else:
                child = min(children, key=lambda c: self.intervals[c][1])
        else:
            raise ValueError(f"unknown player: {node['player']}")
        return self.critical_leaf(child, side)
