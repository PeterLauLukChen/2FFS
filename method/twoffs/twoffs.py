"""Budgeted 2FFS on generated two-fidelity minimax trees.

This module implements the algorithmic skeleton from the paper for numerical
experiments.  It intentionally mirrors the theoretical objects: fast intervals,
slow local intervals, propagated child intervals, side certificates, capped
recursive scales, and budgeted fallback to local sampling.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Side = Literal["L", "U"]
Status = Literal["cert", "progress", "blocked"]


@dataclass(frozen=True)
class TwoFFSConfig:
    """Runtime configuration for 2FFS."""

    delta: float = 0.05
    epsilon: float = 0.0
    slow_cost: float | None = None
    slow_sigma: float | None = None
    max_outer_rounds: int = 200_000
    max_scale: int = 80
    seed: int = 0
    alpha_power: float = 2.0
    alpha_offset: float = 1.0
    confidence_style: Literal["peeling"] = "peeling"
    verbose: bool = False


@dataclass
class NodeState:
    node_id: str
    n_slow: int = 0
    slow_sum: float = 0.0
    local_l: float = -math.inf
    local_u: float = math.inf
    fast_l: float = -math.inf
    fast_u: float = math.inf
    child_l: float = -math.inf
    child_u: float = math.inf
    l: float = -math.inf
    u: float = math.inf


@dataclass
class ScaleState:
    m_loc: float = 0.0
    m_exp: float = 0.0
    done_l: bool = False
    done_u: bool = False


@dataclass
class TwoFFSResult:
    tree_id: str
    recommendation: str | None
    optimal_root_child: str
    correct: bool
    stopped: bool
    outer_rounds: int
    total_cost: float
    fast_cost: float
    slow_cost: float
    num_fast_queries: int
    num_slow_queries: int
    bookkeeping_touches: int
    num_expanded_nodes: int
    root_gap: float
    final_root_intervals: dict[str, tuple[float, float]]
    slow_queries_by_node: dict[str, int]
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
            "outer_rounds": self.outer_rounds,
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
            "slow_queries_by_node": self.slow_queries_by_node,
            "expanded_nodes": self.expanded_nodes,
            "stop_reason": self.stop_reason,
            "trace": self.trace,
        }


class TwoFFSRunner:
    """Run 2FFS on one generated tree JSON object."""

    def __init__(self, tree: dict[str, Any], config: TwoFFSConfig | None = None) -> None:
        self.tree = tree
        self.config = config or TwoFFSConfig()
        self.validate_config()
        self.rng = random.Random(self.config.seed)
        self.nodes: dict[str, dict[str, Any]] = {
            node["id"]: node for node in self.tree["nodes"]
        }
        self.parent: dict[str, str] = {}
        for node in self.tree["nodes"]:
            for child in node.get("children", []):
                self.parent[child] = node["id"]
        self.root = self.tree["root"]
        self.root_children = list(self.tree["root_children"])
        self.depth = int(self.tree["depth"])
        self.fast_query_cost = float(self.tree["fast_oracle"]["cost"])
        self.slow_query_cost = float(
            self.config.slow_cost
            if self.config.slow_cost is not None
            else self.tree["slow_oracle"]["cost"]
        )
        self.slow_sigma = float(
            self.config.slow_sigma
            if self.config.slow_sigma is not None
            else self.tree["slow_oracle"]["sigma"]
        )
        non_root_count = max(1, len(self.nodes) - 1)
        self.node_delta = self.config.delta / non_root_count

        self.state: dict[str, NodeState] = {
            node_id: NodeState(node_id=node_id) for node_id in self.nodes
        }
        self.scale_state: dict[tuple[str, int], ScaleState] = {}
        self.explored: set[str] = set()
        self.expanded: set[str] = set()
        self.rho0: float = 1.0
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
        if self.config.max_outer_rounds < 0:
            raise ValueError("max_outer_rounds must be nonnegative")
        if self.config.max_scale < 0:
            raise ValueError("max_scale must be nonnegative")
        if self.config.slow_cost is not None and self.config.slow_cost <= 0:
            raise ValueError("slow_cost must be positive when provided")
        if self.config.slow_sigma is not None and self.config.slow_sigma <= 0:
            raise ValueError("slow_sigma must be positive when provided")
        if self.config.alpha_offset <= 0:
            raise ValueError("alpha_offset must be positive")
        if self.config.alpha_power < 0:
            raise ValueError("alpha_power must be nonnegative")

    @classmethod
    def from_json_path(
        cls, path: str | Path, config: TwoFFSConfig | None = None
    ) -> "TwoFFSRunner":
        tree = json.loads(Path(path).read_text())
        return cls(tree, config=config)

    def run(self) -> TwoFFSResult:
        self.initialize()
        outer_rounds = 0
        stop_reason = "separated"

        while not self.is_stopped():
            if outer_rounds >= self.config.max_outer_rounds:
                stop_reason = "max_outer_rounds"
                break
            x, side, scale = self.root_obligation()
            if self.is_inf_scale(scale):
                stop_reason = "no_finite_root_obligation"
                break
            q, status = self.resolve(x, side, int(scale), math.inf)
            outer_rounds += 1
            if self.config.verbose:
                self.trace.append(
                    {
                        "round": outer_rounds,
                        "node": x,
                        "side": side,
                        "scale": int(scale),
                        "cost": q,
                        "status": status,
                        "total_cost": self.total_cost,
                    }
                )

        recommendation = self.recommendation()
        return TwoFFSResult(
            tree_id=self.tree["tree_id"],
            recommendation=recommendation,
            optimal_root_child=self.tree["optimal_root_child"],
            correct=recommendation == self.tree["optimal_root_child"],
            stopped=self.is_stopped(),
            outer_rounds=outer_rounds,
            total_cost=round(self.total_cost, 10),
            fast_cost=round(self.fast_cost, 10),
            slow_cost=round(self.slow_cost, 10),
            num_fast_queries=self.num_fast_queries,
            num_slow_queries=self.num_slow_queries,
            bookkeeping_touches=self.bookkeeping_touches,
            num_expanded_nodes=len(self.expanded),
            root_gap=float(self.tree["root_gap"]),
            final_root_intervals={
                child: (self.state[child].l, self.state[child].u)
                for child in self.root_children
            },
            slow_queries_by_node={
                node_id: st.n_slow
                for node_id, st in sorted(self.state.items())
                if st.n_slow > 0
            },
            expanded_nodes=sorted(self.expanded),
            stop_reason=stop_reason,
            trace=self.trace,
        )

    # ------------------------------------------------------------------
    # Initialization and interval propagation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.explored = {self.root}
        for child in self.root_children:
            self.expose_node(child)
        self.refresh_intervals()
        widths = [self.width(child) for child in self.root_children]
        self.rho0 = max(max(widths), 1e-12)

    def expose_node(self, node_id: str) -> None:
        if node_id in self.explored:
            return
        self.explored.add(node_id)
        node = self.nodes[node_id]
        st = self.state[node_id]
        bound = float(node["fast_bound"])
        fast_value = float(node["fast_value"])
        st.fast_l = fast_value - bound
        st.fast_u = fast_value + bound
        self.interval_write_touches()
        self.total_cost += self.fast_query_cost
        self.fast_cost += self.fast_query_cost
        self.num_fast_queries += 1

    def refresh_intervals(self) -> None:
        for node_id in sorted(self.explored, key=lambda x: self.nodes[x]["depth"], reverse=True):
            self.refresh_node_interval(node_id)

    def refresh_intervals_for(self, node_ids: set[str] | list[str]) -> None:
        for node_id in sorted(
            (node for node in node_ids if node in self.explored),
            key=lambda x: self.nodes[x]["depth"],
            reverse=True,
        ):
            self.refresh_node_interval(node_id)

    def ancestors_to_root(self, node_id: str) -> list[str]:
        path = [node_id]
        cur = node_id
        while cur != self.root:
            cur = self.parent[cur]
            path.append(cur)
        return path

    def refresh_node_interval(self, node_id: str) -> None:
        node = self.nodes[node_id]
        st = self.state[node_id]
        if node_id == self.root:
            st.child_l, st.child_u = self.backup_children(node_id)
            st.l, st.u = st.child_l, st.child_u
            self.touch(4)
            return

        lows = [st.fast_l]
        ups = [st.fast_u]
        self.touch(2)
        if st.n_slow > 0:
            lows.append(st.local_l)
            ups.append(st.local_u)
            self.touch(2)
        if node_id in self.expanded:
            st.child_l, st.child_u = self.backup_children(node_id)
            lows.append(st.child_l)
            ups.append(st.child_u)
            self.touch(2)
        st.l = max(lows)
        st.u = min(ups)
        self.scan_touches(len(lows), scans=2)
        self.interval_write_touches()

    def backup_children(self, node_id: str) -> tuple[float, float]:
        node = self.nodes[node_id]
        children = node["children"]
        if not children:
            st = self.state[node_id]
            self.touch(2)
            return st.l, st.u
        self.scan_touches(len(children), scans=2)
        if node["player"] == "max":
            return (
                max(self.state[child].l for child in children),
                max(self.state[child].u for child in children),
            )
        if node["player"] == "min":
            return (
                min(self.state[child].l for child in children),
                min(self.state[child].u for child in children),
            )
        raise ValueError(f"cannot backup children of leaf {node_id}")

    def refresh_and_latch(self, scale: int, affected: set[str] | list[str] | None = None) -> None:
        if affected is None:
            self.refresh_intervals()
        else:
            self.refresh_intervals_for(affected)
        self.latch_scale(scale)

    def latch_scale(self, scale: int) -> None:
        for node_id in sorted(self.explored):
            if node_id == self.root:
                continue
            self.ensure_scale(node_id, scale)
            self.touch(1)
            for side in ("L", "U"):
                if self.cert(node_id, side, scale):
                    self.set_done(node_id, side, scale)

    # ------------------------------------------------------------------
    # Confidence, costs, and dyadic scales
    # ------------------------------------------------------------------

    def radius(self, n: int, delta: float | None = None) -> float:
        if n <= 0:
            return math.inf
        delta = self.node_delta if delta is None else delta
        # A simple time-uniform peeling radius using sum_n 6/(pi^2 n^2)=1.
        log_term = math.log((math.pi * math.pi * n * n) / (6.0 * delta))
        return self.slow_sigma * math.sqrt(2.0 * log_term / n)

    def m_required(self, rho: float, node_id: str) -> int:
        target = rho / 4.0
        if target <= 0:
            return math.inf
        n = 1
        while self.radius(n) > target:
            n *= 2
            if n > 10_000_000:
                raise RuntimeError(f"m_required too large for {node_id} at rho={rho}")
        lo = n // 2 + 1
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if self.radius(mid) <= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def gamma(self, node_id: str, rho: float) -> float:
        fast_bound = float(self.nodes[node_id]["fast_bound"] or 0.0)
        if fast_bound <= rho / 4.0:
            return 0.0
        return self.slow_query_cost * self.m_required(rho, node_id)

    def gamma_k(self, node_id: str, scale: int) -> float:
        return self.gamma(node_id, self.rho(scale))

    def alpha(self, node_id: str) -> float:
        h = int(self.nodes[node_id]["remaining_depth"])
        return (h + self.config.alpha_offset) ** self.config.alpha_power

    def race_budget(self, node_id: str, scale: int) -> float:
        return self.alpha(node_id) * self.gamma_k(node_id, scale)

    def rho(self, scale: int) -> float:
        return self.rho0 / (2.0**scale)

    def width(self, node_id: str) -> float:
        st = self.state[node_id]
        self.touch(2)
        return max(0.0, st.u - st.l)

    def width_scale(self, node_id: str) -> int | float:
        w = self.width(node_id)
        if w <= 1e-14:
            return math.inf
        for scale in range(self.config.max_scale + 1):
            rho = self.rho(scale)
            if rho / 2.0 < w <= rho:
                return scale
        if w > self.rho0:
            return 0
        return self.config.max_scale

    @staticmethod
    def is_inf_scale(scale: int | float) -> bool:
        return isinstance(scale, float) and math.isinf(scale)

    def active_scale(self, node_id: str, side: Side) -> int | float:
        underline = self.width_scale(node_id)
        if self.is_inf_scale(underline):
            return math.inf
        for scale in range(int(underline), self.config.max_scale + 1):
            if not self.comp(node_id, side, scale):
                return scale
        return math.inf

    def capped_scale(self, node_id: str, side: Side, parent_scale: int) -> int | float:
        for scale in range(parent_scale + 1):
            if not self.comp(node_id, side, scale):
                return scale
        return math.inf

    # ------------------------------------------------------------------
    # Certificates and active witnesses
    # ------------------------------------------------------------------

    def ensure_scale(self, node_id: str, scale: int) -> ScaleState:
        key = (node_id, scale)
        if key not in self.scale_state:
            self.scale_state[key] = ScaleState()
        return self.scale_state[key]

    def set_done(self, node_id: str, side: Side, scale: int) -> None:
        st = self.ensure_scale(node_id, scale)
        if side == "L":
            st.done_l = True
        else:
            st.done_u = True
        self.touch(1)

    def is_done(self, node_id: str, side: Side, scale: int) -> bool:
        st = self.ensure_scale(node_id, scale)
        self.touch(1)
        return st.done_l if side == "L" else st.done_u

    def comp(self, node_id: str, side: Side, scale: int) -> bool:
        return self.is_done(node_id, side, scale) or self.cert(node_id, side, scale)

    def cert(self, node_id: str, side: Side, scale: int) -> bool:
        if node_id == self.root:
            return False
        tolerance = self.rho(scale) / 2.0
        st = self.state[node_id]
        if self.width(node_id) <= tolerance:
            return True
        if st.n_slow > 0 and (st.local_u - st.local_l) <= tolerance:
            self.touch(3)
            return True
        if node_id not in self.expanded:
            return False

        node = self.nodes[node_id]
        children = node["children"]
        if not children:
            return False

        if node["player"] == "max" and side == "L":
            self.scan_touches(len(children))
            lambda_l = max(self.state[child].l for child in children)
            self.scan_touches(len(children))
            return all(
                self.state[child].u <= lambda_l + tolerance
                or self.comp(child, "L", scale)
                for child in children
            )
        if node["player"] == "min" and side == "U":
            self.scan_touches(len(children))
            lambda_u = min(self.state[child].u for child in children)
            self.scan_touches(len(children))
            return all(
                self.state[child].l >= lambda_u - tolerance
                or self.comp(child, "U", scale)
                for child in children
            )

        active = self.active_children(node_id, side, scale)
        return bool(active) and all(self.comp(child, side, scale) for child in active)

    def is_comparison_case(self, node_id: str, side: Side) -> bool:
        node = self.nodes[node_id]
        return (node["player"] == "max" and side == "L") or (
            node["player"] == "min" and side == "U"
        )

    def active_children(self, node_id: str, side: Side, scale: int) -> list[str]:
        node = self.nodes[node_id]
        children = node["children"]
        tolerance = self.rho(scale) / 2.0
        if not children:
            return []
        if node["player"] == "min" and side == "L":
            self.scan_touches(len(children), scans=2)
            best = min(self.state[child].l for child in children)
            return [child for child in children if abs(self.state[child].l - best) <= 1e-14]
        if node["player"] == "max" and side == "U":
            self.scan_touches(len(children), scans=2)
            best = max(self.state[child].u for child in children)
            return [child for child in children if abs(self.state[child].u - best) <= 1e-14]
        if node["player"] == "max" and side == "L":
            self.scan_touches(len(children), scans=2)
            lambda_l = max(self.state[child].l for child in children)
            return [
                child
                for child in children
                if self.state[child].u > lambda_l + tolerance
                and not self.comp(child, "L", scale)
            ]
        if node["player"] == "min" and side == "U":
            self.scan_touches(len(children), scans=2)
            lambda_u = min(self.state[child].u for child in children)
            return [
                child
                for child in children
                if self.state[child].l < lambda_u - tolerance
                and not self.comp(child, "U", scale)
            ]
        raise ValueError(f"invalid node/side combination: {node_id}, {side}")

    # ------------------------------------------------------------------
    # Top-level and recursive obligation selection
    # ------------------------------------------------------------------

    def is_stopped(self) -> bool:
        return self.recommendation() is not None

    def recommendation(self) -> str | None:
        m = len(self.root_children)
        self.touch(m)
        self.scan_touches(max(0, m - 1), scans=m)
        for child in self.root_children:
            lower = self.state[child].l
            other_upper = max(
                self.state[other].u for other in self.root_children if other != child
            )
            if lower >= other_upper - self.config.epsilon:
                return child
        return None

    def contender_scale(self, node_id: str) -> tuple[Side, int | float]:
        k_l = self.active_scale(node_id, "L")
        k_u = self.active_scale(node_id, "U")
        if not self.is_inf_scale(k_l) and (
            self.is_inf_scale(k_u) or self.rho(int(k_l)) >= self.rho(int(k_u))
        ):
            return "L", k_l
        return "U", k_u

    def root_obligation(self) -> tuple[str, Side, int | float]:
        self.scan_touches(len(self.root_children))
        leader = max(self.root_children, key=lambda child: self.state[child].l)
        challengers = [child for child in self.root_children if child != leader]
        self.scan_touches(len(challengers))
        challenger = max(challengers, key=lambda child: self.state[child].u)
        k_leader = self.active_scale(leader, "L")
        side_chal, k_chal = self.contender_scale(challenger)
        if not self.is_inf_scale(k_leader) and (
            self.is_inf_scale(k_chal) or self.rho(int(k_leader)) >= self.rho(int(k_chal))
        ):
            return leader, "L", k_leader
        return challenger, side_chal, k_chal

    def child_obligation(self, node_id: str, side: Side, scale: int) -> tuple[str, Side, int | float] | None:
        if self.is_comparison_case(node_id, side):
            active = self.active_children(node_id, side, scale)
            if not active:
                return None
            if self.nodes[node_id]["player"] == "max":
                self.scan_touches(len(active))
                blocker = max(active, key=lambda child: self.state[child].u)
            else:
                self.scan_touches(len(active))
                blocker = min(active, key=lambda child: self.state[child].l)
            opposite: Side = "U" if side == "L" else "L"
            k_opp = self.capped_scale(blocker, opposite, scale)
            if not self.is_inf_scale(k_opp):
                return blocker, opposite, k_opp
            k_same = self.capped_scale(blocker, side, scale)
            return blocker, side, k_same

        candidates = [
            child
            for child in self.active_children(node_id, side, scale)
            if not self.comp(child, side, scale)
        ]
        if not candidates:
            return None
        self.scan_touches(len(candidates))
        child = max(candidates, key=self.width)
        return child, side, self.capped_scale(child, side, scale)

    # ------------------------------------------------------------------
    # Atomic actions and budgeted recursive resolver
    # ------------------------------------------------------------------

    def local_step(self, node_id: str, scale: int, cap: float) -> tuple[float, Status]:
        if cap < self.slow_query_cost:
            return 0.0, "blocked"
        st = self.state[node_id]
        sample = self.rng.gauss(float(self.nodes[node_id]["slow_mean"]), self.slow_sigma)
        st.n_slow += 1
        st.slow_sum += sample
        empirical = st.slow_sum / st.n_slow
        rad = self.radius(st.n_slow)
        st.local_l = empirical - rad
        st.local_u = empirical + rad
        self.touch(2)  # n_slow and slow_sum updates.
        self.interval_write_touches()
        self.ensure_scale(node_id, scale).m_loc += self.slow_query_cost
        self.touch(1)
        self.total_cost += self.slow_query_cost
        self.slow_cost += self.slow_query_cost
        self.num_slow_queries += 1
        self.refresh_and_latch(scale, self.ancestors_to_root(node_id))
        return self.slow_query_cost, "progress"

    def expand_node(self, node_id: str, scale: int) -> tuple[float, Status]:
        if node_id in self.expanded:
            return 0.0, "cert"
        children = self.nodes[node_id]["children"]
        for child in children:
            self.expose_node(child)
            self.ensure_scale(child, scale)
            self.touch(1)
        self.expanded.add(node_id)
        cost = self.fast_query_cost * len(children)
        self.ensure_scale(node_id, scale).m_exp += cost
        self.touch(1)
        affected = set(children)
        affected.update(self.ancestors_to_root(node_id))
        self.refresh_and_latch(scale, affected)
        return cost, "progress"

    def resolve(self, node_id: str, side: Side, scale: int, cap: float) -> tuple[float, Status]:
        self.ensure_scale(node_id, scale)
        if self.comp(node_id, side, scale):
            self.set_done(node_id, side, scale)
            return 0.0, "cert"

        if int(self.nodes[node_id]["remaining_depth"]) == 0:
            return self.local_step(node_id, scale, cap)

        scale_state = self.ensure_scale(node_id, scale)
        remaining_race_budget = self.race_budget(node_id, scale) - scale_state.m_exp
        if remaining_race_budget <= 0:
            return self.local_step(node_id, scale, cap)

        recursive_cap = min(cap, remaining_race_budget)

        if node_id not in self.expanded:
            child_cost = self.fast_query_cost * len(self.nodes[node_id]["children"])
            if child_cost > remaining_race_budget:
                return self.local_step(node_id, scale, cap)
            if child_cost > cap:
                return 0.0, "blocked"
            return self.expand_node(node_id, scale)

        obligation = self.child_obligation(node_id, side, scale)
        if obligation is None:
            self.latch_scale(scale)
            if self.comp(node_id, side, scale):
                self.set_done(node_id, side, scale)
                return 0.0, "cert"
            return self.local_step(node_id, scale, cap)

        child, child_side, child_scale = obligation
        if self.is_inf_scale(child_scale):
            self.latch_scale(scale)
            return self.local_step(node_id, scale, cap)

        q, status = self.resolve(child, child_side, int(child_scale), recursive_cap)
        if status == "blocked":
            if cap < remaining_race_budget:
                return 0.0, "blocked"
            return self.local_step(node_id, scale, cap)

        scale_state.m_exp += q
        self.latch_scale(scale)
        return q, status
