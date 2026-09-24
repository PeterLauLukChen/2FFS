"""Fast-expansion-only minimax baseline.

This baseline never queries the slow oracle.  It initializes root children with
their deterministic fast intervals and greedily expands ambiguous fast-only
subtrees until the root is separated or the whole tree has been exposed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class FastMinimaxResult:
    tree_id: str
    recommendation: str
    optimal_root_child: str
    correct: bool
    stopped: bool
    total_cost: float
    fast_cost: float
    num_fast_queries: int
    bookkeeping_touches: int
    num_expanded_nodes: int
    root_gap: float
    final_root_intervals: dict[str, tuple[float, float]]
    expanded_nodes: list[str]
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "recommendation": self.recommendation,
            "optimal_root_child": self.optimal_root_child,
            "correct": self.correct,
            "stopped": self.stopped,
            "total_cost": self.total_cost,
            "fast_cost": self.fast_cost,
            "slow_cost": 0.0,
            "num_fast_queries": self.num_fast_queries,
            "num_slow_queries": 0,
            "num_nodes_sampled_visited": self.num_fast_queries,
            "bookkeeping_touches": self.bookkeeping_touches,
            "adjusted_work": self.num_fast_queries + self.bookkeeping_touches,
            "num_expanded_nodes": self.num_expanded_nodes,
            "root_gap": self.root_gap,
            "final_root_intervals": self.final_root_intervals,
            "expanded_nodes": self.expanded_nodes,
            "stop_reason": self.stop_reason,
        }


class FastMinimaxRunner:
    def __init__(
        self,
        tree: dict[str, Any],
        epsilon: float = 0.0,
        max_expansions: int | None = None,
    ) -> None:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        if max_expansions is not None and max_expansions < 0:
            raise ValueError("max_expansions must be nonnegative when provided")
        self.tree = tree
        self.epsilon = epsilon
        self.max_expansions = max_expansions
        self.nodes = {node["id"]: node for node in tree["nodes"]}
        self.parent: dict[str, str] = {}
        for node in tree["nodes"]:
            for child in node.get("children", []):
                self.parent[child] = node["id"]
        self.root = tree["root"]
        self.root_children = list(tree["root_children"])
        self.fast_query_cost = float(tree["fast_oracle"]["cost"])
        self.explored: set[str] = set()
        self.expanded: set[str] = set()
        self.interval: dict[str, tuple[float, float]] = {}
        self.total_cost = 0.0
        self.num_fast_queries = 0
        self.bookkeeping_touches = 0

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

    @classmethod
    def from_json_path(
        cls,
        path: str | Path,
        epsilon: float = 0.0,
        max_expansions: int | None = None,
    ) -> "FastMinimaxRunner":
        return cls(
            json.loads(Path(path).read_text()),
            epsilon=epsilon,
            max_expansions=max_expansions,
        )

    def run(self) -> FastMinimaxResult:
        self.initialize()
        stop_reason = "separated"
        while not self.is_stopped():
            if (
                self.max_expansions is not None
                and len(self.expanded) >= self.max_expansions
            ):
                stop_reason = "max_expansions"
                break
            node_id = self.choose_expansion_node()
            if node_id is None:
                stop_reason = "fully_expanded"
                break
            self.expand(node_id)

        rec = self.recommendation()
        return FastMinimaxResult(
            tree_id=self.tree["tree_id"],
            recommendation=rec,
            optimal_root_child=self.tree["optimal_root_child"],
            correct=rec == self.tree["optimal_root_child"],
            stopped=self.is_stopped(),
            total_cost=round(self.total_cost, 10),
            fast_cost=round(self.total_cost, 10),
            num_fast_queries=self.num_fast_queries,
            bookkeeping_touches=self.bookkeeping_touches,
            num_expanded_nodes=len(self.expanded),
            root_gap=float(self.tree["root_gap"]),
            final_root_intervals={child: self.interval[child] for child in self.root_children},
            expanded_nodes=sorted(self.expanded),
            stop_reason=stop_reason,
        )

    def initialize(self) -> None:
        self.explored.add(self.root)
        for child in self.root_children:
            self.expose(child)
        self.refresh_all()

    def expose(self, node_id: str) -> None:
        if node_id in self.explored:
            return
        self.explored.add(node_id)
        node = self.nodes[node_id]
        center = float(node["fast_value"])
        bound = float(node["fast_bound"])
        self.interval[node_id] = (center - bound, center + bound)
        self.interval_write_touches()
        self.total_cost += self.fast_query_cost
        self.num_fast_queries += 1

    def expand(self, node_id: str) -> None:
        if node_id in self.expanded:
            return
        for child in self.nodes[node_id]["children"]:
            self.expose(child)
        self.expanded.add(node_id)
        self.refresh_from_node(node_id)

    def refresh_all(self) -> None:
        for node_id in sorted(self.explored, key=lambda x: self.nodes[x]["depth"], reverse=True):
            self.refresh_node(node_id)

    def refresh_from_node(self, node_id: str) -> None:
        """Refresh exactly the expanded node and its ancestors."""
        affected = self.ancestors_to_root(node_id)
        for cur in sorted(affected, key=lambda x: self.nodes[x]["depth"], reverse=True):
            self.refresh_node(cur)

    def ancestors_to_root(self, node_id: str) -> list[str]:
        path = [node_id]
        cur = node_id
        while cur != self.root:
            cur = self.parent[cur]
            path.append(cur)
        return path

    def refresh_node(self, node_id: str) -> None:
        node = self.nodes[node_id]
        if node_id == self.root:
            self.interval[node_id] = self.backup_children(node_id)
            self.interval_write_touches()
            return
        if node_id not in self.expanded:
            return
        fast_l, fast_u = self.interval[node_id]
        child_l, child_u = self.backup_children(node_id)
        self.interval[node_id] = (max(fast_l, child_l), min(fast_u, child_u))
        self.touch(4)  # fast/child endpoint comparisons.
        self.interval_write_touches()

    def backup_children(self, node_id: str) -> tuple[float, float]:
        children = self.nodes[node_id]["children"]
        self.scan_touches(len(children), scans=2)
        if self.nodes[node_id]["player"] == "max":
            return (
                max(self.interval[child][0] for child in children),
                max(self.interval[child][1] for child in children),
            )
        if self.nodes[node_id]["player"] == "min":
            return (
                min(self.interval[child][0] for child in children),
                min(self.interval[child][1] for child in children),
            )
        raise ValueError(f"leaf has no children: {node_id}")

    def is_stopped(self) -> bool:
        m = len(self.root_children)
        self.touch(m)
        self.scan_touches(max(0, m - 1), scans=m)
        return any(
            self.interval[child][0]
            >= max(self.interval[other][1] for other in self.root_children if other != child)
            - self.epsilon
            for child in self.root_children
        )

    def recommendation(self) -> str:
        m = len(self.root_children)
        self.touch(m)
        self.scan_touches(max(0, m - 1), scans=m)
        separated = [
            child
            for child in self.root_children
            if self.interval[child][0]
            >= max(self.interval[other][1] for other in self.root_children if other != child)
            - self.epsilon
        ]
        if separated:
            self.scan_touches(len(separated))
            return max(separated, key=lambda child: self.interval[child][0])
        self.scan_touches(len(self.root_children), scans=2)
        return max(
            self.root_children,
            key=lambda child: 0.5 * (self.interval[child][0] + self.interval[child][1]),
        )

    def choose_expansion_node(self) -> str | None:
        self.scan_touches(len(self.root_children))
        leader = max(self.root_children, key=lambda child: self.interval[child][0])
        self.scan_touches(max(0, len(self.root_children) - 1))
        challenger = max(
            [child for child in self.root_children if child != leader],
            key=lambda child: self.interval[child][1],
        )
        target = max([leader, challenger], key=self.width)
        node = self.descend_to_expandable(target)
        if node is not None:
            return node
        candidates = [
            node_id
            for node_id in self.explored
            if self.is_expandable(node_id)
        ]
        if not candidates:
            return None
        return max(candidates, key=self.width)

    def descend_to_expandable(self, node_id: str) -> str | None:
        cur = node_id
        while True:
            if self.is_expandable(cur):
                return cur
            if cur not in self.expanded:
                return None
            children = self.nodes[cur]["children"]
            expandable_children = [
                child for child in children if self.width(child) > 0 and self.has_expandable_descendant(child)
            ]
            if not expandable_children:
                return None
            self.scan_touches(len(expandable_children))
            cur = max(expandable_children, key=self.width)

    def has_expandable_descendant(self, node_id: str) -> bool:
        if self.is_expandable(node_id):
            return True
        if node_id not in self.expanded:
            return False
        return any(self.has_expandable_descendant(child) for child in self.nodes[node_id]["children"])

    def is_expandable(self, node_id: str) -> bool:
        return (
            node_id in self.explored
            and node_id != self.root
            and node_id not in self.expanded
            and bool(self.nodes[node_id]["children"])
        )

    def width(self, node_id: str) -> float:
        self.touch(2)
        low, high = self.interval.get(node_id, (-math.inf, math.inf))
        return max(0.0, high - low)
