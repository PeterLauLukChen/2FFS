"""Fixed-depth expansion plus slow-confidence minimax baseline.

The baseline expands the tree uniformly to a chosen depth, pays fast-oracle
cost for every exposed non-root node, and then runs LUCB-style slow sampling on
the fixed frontier.  Frontier slow intervals and exposed fast intervals are
intersected before minimax interval backup.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FixedDepthSlowConfig:
    delta: float = 0.05
    epsilon: float = 0.0
    seed: int = 0
    expand_depth: int | None = None
    slow_cost: float | None = None
    slow_sigma: float | None = None
    max_rounds: int = 500_000
    verbose: bool = False


@dataclass
class FrontierStats:
    n: int = 0
    total: float = 0.0
    low: float = -math.inf
    high: float = math.inf


@dataclass
class FixedDepthSlowResult:
    tree_id: str
    recommendation: str
    optimal_root_child: str
    correct: bool
    stopped: bool
    rounds: int
    expand_depth: int
    frontier_size: int
    total_cost: float
    fast_cost: float
    slow_cost: float
    num_fast_queries: int
    num_slow_queries: int
    bookkeeping_touches: int
    num_expanded_nodes: int
    root_gap: float
    final_root_intervals: dict[str, tuple[float, float]]
    slow_queries_by_frontier_node: dict[str, int]
    expanded_nodes: list[str]
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
            "expand_depth": self.expand_depth,
            "frontier_size": self.frontier_size,
            "total_cost": self.total_cost,
            "fast_cost": self.fast_cost,
            "slow_cost": self.slow_cost,
            "num_fast_queries": self.num_fast_queries,
            "num_slow_queries": self.num_slow_queries,
            "num_nodes_sampled_visited": self.num_fast_queries + self.num_slow_queries,
            "bookkeeping_touches": self.bookkeeping_touches,
            "adjusted_work": (
                self.num_fast_queries + self.num_slow_queries + self.bookkeeping_touches
            ),
            "num_expanded_nodes": self.num_expanded_nodes,
            "root_gap": self.root_gap,
            "final_root_intervals": self.final_root_intervals,
            "slow_queries_by_frontier_node": self.slow_queries_by_frontier_node,
            "expanded_nodes": self.expanded_nodes,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
        }


class FixedDepthSlowRunner:
    def __init__(
        self, tree: dict[str, Any], config: FixedDepthSlowConfig | None = None
    ) -> None:
        self.tree = tree
        self.config = config or FixedDepthSlowConfig()
        self.validate_config()
        self.rng = random.Random(self.config.seed)
        self.nodes = {node["id"]: node for node in tree["nodes"]}
        self.parent: dict[str, str] = {}
        for node in tree["nodes"]:
            for child in node.get("children", []):
                self.parent[child] = node["id"]
        self.root = tree["root"]
        self.root_children = list(tree["root_children"])
        self.tree_depth = int(tree["depth"])
        self.expand_depth = (
            self.tree_depth
            if self.config.expand_depth is None
            else int(self.config.expand_depth)
        )
        if self.expand_depth < 1 or self.expand_depth > self.tree_depth:
            raise ValueError(
                f"expand_depth must be in [1, {self.tree_depth}], got {self.expand_depth}"
            )
        self.fast_cost_unit = float(tree["fast_oracle"]["cost"])
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
        self.frontier = sorted(
            node_id
            for node_id, node in self.nodes.items()
            if int(node["depth"]) == self.expand_depth
        )
        self.node_delta = self.config.delta / max(1, len(self.frontier))
        self.stats = {node_id: FrontierStats() for node_id in self.frontier}
        self.fast_intervals: dict[str, tuple[float, float]] = {}
        self.intervals: dict[str, tuple[float, float]] = {}
        self.exposed: set[str] = set()
        self.expanded: set[str] = set()
        self.total_cost = 0.0
        self.fast_cost = 0.0
        self.slow_cost = 0.0
        self.num_fast_queries = 0
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
        cls, path: str | Path, config: FixedDepthSlowConfig | None = None
    ) -> "FixedDepthSlowRunner":
        return cls(json.loads(Path(path).read_text()), config=config)

    def run(self) -> FixedDepthSlowResult:
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
            frontier_nodes = {
                self.critical_frontier_node(leader, "L"),
                self.critical_frontier_node(challenger, "U"),
            }
            for node_id in sorted(frontier_nodes):
                self.sample_frontier_node(node_id)
            self.refresh_from_frontier_nodes(frontier_nodes)
            rounds += 1
            if self.config.verbose:
                self.trace.append(
                    {
                        "round": rounds,
                        "leader": leader,
                        "challenger": challenger,
                        "sampled_frontier_nodes": sorted(frontier_nodes),
                        "total_cost": self.total_cost,
                    }
                )

        rec = self.recommendation()
        return FixedDepthSlowResult(
            tree_id=self.tree["tree_id"],
            recommendation=rec,
            optimal_root_child=self.tree["optimal_root_child"],
            correct=rec == self.tree["optimal_root_child"],
            stopped=self.is_stopped(),
            rounds=rounds,
            expand_depth=self.expand_depth,
            frontier_size=len(self.frontier),
            total_cost=round(self.total_cost, 10),
            fast_cost=round(self.fast_cost, 10),
            slow_cost=round(self.slow_cost, 10),
            num_fast_queries=self.num_fast_queries,
            num_slow_queries=self.num_slow_queries,
            bookkeeping_touches=self.bookkeeping_touches,
            num_expanded_nodes=len(self.expanded),
            root_gap=float(self.tree["root_gap"]),
            final_root_intervals={child: self.intervals[child] for child in self.root_children},
            slow_queries_by_frontier_node={
                node_id: st.n for node_id, st in self.stats.items() if st.n > 0
            },
            expanded_nodes=sorted(self.expanded),
            stop_reason=stop_reason,
            trace=self.trace,
        )

    def initialize(self) -> None:
        self.exposed.add(self.root)
        for node_id, node in sorted(self.nodes.items(), key=lambda item: item[1]["depth"]):
            depth = int(node["depth"])
            if node_id == self.root or depth > self.expand_depth:
                continue
            self.expose_node(node_id)
        self.expanded = {
            node_id
            for node_id, node in self.nodes.items()
            if node_id != self.root
            and int(node["depth"]) < self.expand_depth
            and bool(node["children"])
        }
        for node_id in self.frontier:
            self.sample_frontier_node(node_id)
        self.refresh_all()

    def expose_node(self, node_id: str) -> None:
        if node_id in self.exposed:
            return
        self.exposed.add(node_id)
        node = self.nodes[node_id]
        center = float(node["fast_value"])
        bound = float(node["fast_bound"])
        self.fast_intervals[node_id] = (center - bound, center + bound)
        self.interval_write_touches()
        self.total_cost += self.fast_cost_unit
        self.fast_cost += self.fast_cost_unit
        self.num_fast_queries += 1

    def sample_frontier_node(self, node_id: str) -> None:
        sample = self.rng.gauss(float(self.nodes[node_id]["slow_mean"]), self.slow_sigma)
        st = self.stats[node_id]
        st.n += 1
        st.total += sample
        empirical = st.total / st.n
        rad = self.radius(st.n)
        st.low = empirical - rad
        st.high = empirical + rad
        self.touch(2)  # n and running total updates.
        self.interval_write_touches()
        self.total_cost += self.slow_cost_unit
        self.slow_cost += self.slow_cost_unit
        self.num_slow_queries += 1

    def radius(self, n: int) -> float:
        log_term = math.log((math.pi * math.pi * n * n) / (6.0 * self.node_delta))
        return self.slow_sigma * math.sqrt(2.0 * log_term / n)

    def refresh_all(self) -> None:
        for node_id in self.frontier:
            self.refresh_frontier_interval(node_id)

        for depth in range(self.expand_depth - 1, -1, -1):
            for node_id, node in self.nodes.items():
                if int(node["depth"]) != depth:
                    continue
                self.refresh_internal_interval(node_id)

    def refresh_from_frontier_nodes(self, frontier_nodes: set[str]) -> None:
        """Refresh only sampled frontier nodes and their affected ancestors."""
        affected: set[str] = set()
        for node_id in frontier_nodes:
            self.refresh_frontier_interval(node_id)
            cur = node_id
            while cur != self.root:
                cur = self.parent[cur]
                affected.add(cur)
        for node_id in sorted(affected, key=lambda x: self.nodes[x]["depth"], reverse=True):
            self.refresh_internal_interval(node_id)

    def refresh_frontier_interval(self, node_id: str) -> None:
        slow_l, slow_u = self.stats[node_id].low, self.stats[node_id].high
        fast_l, fast_u = self.fast_intervals[node_id]
        self.intervals[node_id] = (max(slow_l, fast_l), min(slow_u, fast_u))
        self.touch(4)
        self.interval_write_touches()

    def refresh_internal_interval(self, node_id: str) -> None:
        child_interval = self.backup_children(node_id)
        if node_id == self.root:
            self.intervals[node_id] = child_interval
            self.interval_write_touches()
            return
        fast_l, fast_u = self.fast_intervals[node_id]
        child_l, child_u = child_interval
        self.intervals[node_id] = (max(fast_l, child_l), min(fast_u, child_u))
        self.touch(4)
        self.interval_write_touches()

    def backup_children(self, node_id: str) -> tuple[float, float]:
        node = self.nodes[node_id]
        children = node["children"]
        self.scan_touches(len(children), scans=2)
        if node["player"] == "max":
            return (
                max(self.intervals[child][0] for child in children),
                max(self.intervals[child][1] for child in children),
            )
        if node["player"] == "min":
            return (
                min(self.intervals[child][0] for child in children),
                min(self.intervals[child][1] for child in children),
            )
        raise ValueError(f"leaf/frontier node has no children to back up: {node_id}")

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
        self.scan_touches(len(self.root_children), scans=2)
        return max(
            self.root_children,
            key=lambda child: 0.5 * (self.intervals[child][0] + self.intervals[child][1]),
        )

    def critical_frontier_node(self, node_id: str, side: str) -> str:
        if int(self.nodes[node_id]["depth"]) == self.expand_depth:
            return node_id
        node = self.nodes[node_id]
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
            raise ValueError(f"unexpected leaf above frontier: {node_id}")
        return self.critical_frontier_node(child, side)
