import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import Deque, Dict, List, Optional, Self, Set

from tree import Node, NodeState


class NoPossibleSolutions(BaseException):
    "Exception to raise when search exhausted all paths and no solution was found"


@dataclass(frozen=True)
class SearchResult:
    success: bool
    solution: Optional[Node]
    path_cost: float
    expanded_nodes: int
    frontier_nodes: int
    processing_time_sec: float

    @property
    def solution_path(self) -> List[object]:
        if not self.solution:
            return []
        return self.solution.path()

    def __str__(self: Self) -> str:
        if not self.success or not self.solution:
            return (
                "No solution found "
                f"(expanded={self.expanded_nodes}, frontier={self.frontier_nodes}, "
                f"time={self.processing_time_sec:.4f}s)"
            )
        return (
            f"{self.solution}\n"
            f"Cost: {self.path_cost}\n"
            f"Expanded nodes: {self.expanded_nodes}\n"
            f"Frontier nodes: {self.frontier_nodes}\n"
            f"Time: {self.processing_time_sec:.4f}s"
        )


class Search(ABC):
    "Base class for searches"

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        self.root = Node(init_state, None, None, 0, init_state.heuristic())
        self.eval_repeated = eval_repeated
        self.expanded_nodes = 0
        self.start_time = 0.0

    @abstractmethod
    def search(self: Self) -> SearchResult:
        "Performs the search and returns a solution or raises NoPossibleSolutions"
        raise NotImplementedError()

    def _build_result(self: Self, solution: Optional[Node]) -> SearchResult:
        return SearchResult(
            success=solution is not None,
            solution=solution,
            path_cost=solution.cost if solution else 0,
            expanded_nodes=self.expanded_nodes,
            frontier_nodes=self.frontier_size(),
            processing_time_sec=time.perf_counter() - self.start_time,
        )

    @abstractmethod
    def frontier_size(self: Self) -> int:
        raise NotImplementedError()


class QueueSearch(Search):
    "Search implementation for FIFO/LIFO frontiers."

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        super().__init__(init_state, eval_repeated)
        self.frontier_states: Set[NodeState] = {init_state}
        self.explored_states: Set[NodeState] = set()

    def _should_skip_child(self: Self, child: Node) -> bool:
        if self.eval_repeated:
            return False
        return child.state in self.explored_states or child.state in self.frontier_states

    def _register_child(self: Self, child: Node) -> None:
        if not self.eval_repeated:
            self.frontier_states.add(child.state)

    def _mark_expanded(self: Self, node: Node) -> None:
        if not self.eval_repeated:
            self.explored_states.add(node.state)
        self.expanded_nodes += 1


class BFS(QueueSearch):
    "Explores least deep nodes first"

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        super().__init__(init_state, eval_repeated)
        self.frontier: Deque[Node] = deque([self.root])

    def frontier_size(self: Self) -> int:
        return len(self.frontier)

    def search(self: Self) -> SearchResult:
        self.start_time = time.perf_counter()
        while self.frontier:
            node = self.frontier.popleft()
            self.frontier_states.discard(node.state)

            if node.is_goal():
                return self._build_result(node)

            self._mark_expanded(node)
            for child in node.expand():
                if self._should_skip_child(child):
                    continue
                self.frontier.append(child)
                self._register_child(child)

        raise NoPossibleSolutions()


class DFS(QueueSearch):
    "Explores deepest nodes first"

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        super().__init__(init_state, eval_repeated)
        self.frontier: List[Node] = [self.root]

    def frontier_size(self: Self) -> int:
        return len(self.frontier)

    def search(self: Self) -> SearchResult:
        self.start_time = time.perf_counter()
        while self.frontier:
            node = self.frontier.pop()
            self.frontier_states.discard(node.state)

            if node.is_goal():
                return self._build_result(node)
            if not self.eval_repeated and node.state in self.explored_states:
                continue

            self._mark_expanded(node)
            for child in node.expand():
                if self._should_skip_child(child):
                    continue
                self.frontier.append(child)
                self._register_child(child)

        raise NoPossibleSolutions()


class DLS(DFS):
    "Depth limit search, like DFS but limited"

    def __init__(
        self: Self,
        init_state: NodeState,
        limit: int,
        eval_repeated: bool = False,
    ):
        super().__init__(init_state, eval_repeated)
        self.limit = limit

    def search(self: Self) -> SearchResult:
        self.start_time = time.perf_counter()
        while self.frontier:
            node = self.frontier.pop()
            self.frontier_states.discard(node.state)

            if node.is_goal():
                return self._build_result(node)
            if node.depth >= self.limit:
                continue
            if not self.eval_repeated and node.state in self.explored_states:
                continue

            self._mark_expanded(node)
            for child in node.expand():
                if self._should_skip_child(child):
                    continue
                self.frontier.append(child)
                self._register_child(child)

        raise NoPossibleSolutions()


class IDDFS(Search):
    "Repeats depth limited search with increasing limits."

    def __init__(
        self: Self,
        init_state: NodeState,
        initial_limit: int,
        growth_factor: int,
        max_limit: Optional[int] = None,
    ):
        super().__init__(init_state)
        self.initial_limit = initial_limit
        self.growth_factor = growth_factor
        self.max_limit = max_limit
        self._frontier_size = 0

    def frontier_size(self: Self) -> int:
        return self._frontier_size

    def search(self: Self) -> SearchResult:
        self.start_time = time.perf_counter()
        limit = self.initial_limit

        while self.max_limit is None or limit <= self.max_limit:
            dls = DLS(self.root.state, limit)
            try:
                result = dls.search()
                self.expanded_nodes += dls.expanded_nodes
                self._frontier_size = dls.frontier_size()
                return SearchResult(
                    True,
                    result.solution,
                    result.path_cost,
                    self.expanded_nodes,
                    self._frontier_size,
                    time.perf_counter() - self.start_time,
                )
            except NoPossibleSolutions:
                self.expanded_nodes += dls.expanded_nodes
                self._frontier_size = dls.frontier_size()
                limit += self.growth_factor

        raise NoPossibleSolutions()


class PrioritySearch(Search):
    "Search implementation for heap-backed priority frontiers."

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        super().__init__(init_state, eval_repeated)
        self._push_counter = count()
        self.frontier: List[tuple[float, float, int, Node]] = []
        self.best_costs: Dict[NodeState, float] = {init_state: 0}
        self._push(self.root)

    def frontier_size(self: Self) -> int:
        return len(self.frontier)

    @abstractmethod
    def priority(self: Self, node: Node) -> tuple[float, float]:
        raise NotImplementedError()

    def _push(self: Self, node: Node) -> None:
        primary, secondary = self.priority(node)
        heapq.heappush(self.frontier, (primary, secondary, next(self._push_counter), node))

    def search(self: Self) -> SearchResult:
        self.start_time = time.perf_counter()
        while self.frontier:
            _, _, _, node = heapq.heappop(self.frontier)

            if node.is_goal():
                return self._build_result(node)
            if not self.eval_repeated and node.cost > self.best_costs.get(node.state, float("inf")):
                continue

            self.expanded_nodes += 1
            for child in node.expand():
                if self.eval_repeated or child.cost < self.best_costs.get(child.state, float("inf")):
                    self.best_costs[child.state] = child.cost
                    self._push(child)

        raise NoPossibleSolutions()


class LocalGreedy(PrioritySearch):
    "Like DFS but expanding deepest best heuristically-estimated nodes first"

    def priority(self: Self, node: Node) -> tuple[float, float]:
        return (-node.depth, node.heuristic_value)


class GlobalGreedy(PrioritySearch):
    "Expands best heuristically-estimated nodes first"

    def priority(self: Self, node: Node) -> tuple[float, float]:
        return (node.heuristic_value, node.cost)


Greedy = GlobalGreedy


class AStar(PrioritySearch):
    "Expands the lowest cost + heuristic nodes first"

    def priority(self: Self, node: Node) -> tuple[float, float]:
        return (node.cost + node.heuristic_value, node.heuristic_value)
