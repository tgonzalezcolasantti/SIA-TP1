from enum import Enum
import heapq
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import (
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Self,
    Sequence,
    TypeVar,
    override,
)

from tree import Node, NodeState


class NoPossibleSolutions(BaseException):
    "Exception to raise when search exhausted all paths and no solution was found"


@dataclass(frozen=True)
class SearchResult:
    "The solution found by the algorithm."

    success: bool
    solution: Optional[Node]
    path_cost: float
    expanded_nodes: int
    frontier_nodes: int
    processing_time_sec: float

    @property
    def solution_path(self) -> List[Enum]:
        "List of actions to solve the problem"
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


FrontierList = TypeVar(
    "FrontierList", bound=Sequence[Node]
)


class Search(ABC, Generic[FrontierList]):
    "Base class for searches"

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        self.root = Node(init_state, None, None, 0, init_state.heuristic())
        self.eval_repeated = eval_repeated
        self.expanded_nodes: int = 0
        self.start_time: float = 0.0
        self.limit: Optional[int] = None
        self.frontier: FrontierList
        self.frontier_overflow: List[Node] = []
        self.skipped_nodes: int = 0

    @abstractmethod
    def _should_skip_child(self: Self, child: Node) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def _register_child(self: Self, child: Node) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _mark_expanded(self: Self, node: Node) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _get_next_node(self: Self) -> Node:
        raise NotImplementedError()

    def search(self: Self) -> SearchResult:
        "Performs the search and returns a solution or raises NoPossibleSolutions"
        self.start_time = time.perf_counter()
        while self.frontier:
            node = self._get_next_node()

            if node.is_goal():
                return self._build_result(node)
            if self.limit and node.depth >= self.limit:
                self.frontier_overflow.append(node)
                continue

            self._mark_expanded(node)
            for child in node.expand():
                if not self._should_skip_child(child):
                    self._register_child(child)
                else:
                    self.skipped_nodes += 1

        raise NoPossibleSolutions()

    def _build_result(self: Self, solution: Optional[Node]) -> SearchResult:
        return SearchResult(
            success=solution is not None,
            solution=solution,
            path_cost=solution.cost if solution else 0,
            expanded_nodes=self.expanded_nodes,
            frontier_nodes=self.frontier_size(),
            processing_time_sec=time.perf_counter() - self.start_time,
        )

    def frontier_size(self: Self) -> int:
        "Returns how many nodes are currently in the tree frontier"
        return len(self.frontier)


class QueueSearch(Search):
    "Search implementation for FIFO/LIFO frontiers."

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        super().__init__(init_state, eval_repeated)
        self.frontier: Deque[Node] = deque([self.root])
        self.explored_states: Dict[NodeState, int] = {}
        self.limit: Optional[int] = None

    @override
    def _should_skip_child(self: Self, child: Node) -> bool:
        if not self.eval_repeated:
            if child.depth >= self.explored_states.get(child.state, float("inf")):
                return True
            # If child is evaluable, we write it immediately to avoid generating extra useless nodes
            # Instead of waiting to pop the node next iteration and registering it there
            # The end result is the same: only the first encounter of a state will be explored
            # But we save *so much* execution time doing it this way
            self.explored_states[child.state] = child.depth
        return False
    
    @override
    def _register_child(self: Self, child: Node) -> None:
        self.frontier.append(child)

    @override
    def _mark_expanded(self: Self, node: Node) -> None:
        if not self.eval_repeated:
            self.explored_states[node.state] = node.depth
        self.expanded_nodes += 1

    def expand_frontier(self: Self):
        "Restores any nodes excluded for being too deep to the current frontier"
        self.frontier.extend(self.frontier_overflow)
        self.frontier_overflow.clear()


class BFS(QueueSearch):
    "Explores least deep nodes first"
    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        super().__init__(init_state, eval_repeated)
        self.current_depth = 0
    @override
    def _get_next_node(self: Self) -> Node:
        node = self.frontier.popleft()
        # if node.depth > self.current_depth:
        #     self.current_depth = node.depth
        #     print(f"Level {node.depth} (front. {len(self.frontier)}, eval. {len(self.explored_states)}, expanded {self.expanded_nodes}, ratio {((self.skipped_nodes)*100 / (self.expanded_nodes + self.skipped_nodes+1)):.4}%)")
        return node


class DFS(QueueSearch):
    "Explores deepest nodes first"

    @override
    def _get_next_node(self: Self) -> Node:
        node = self.frontier.pop()
        return node


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


class IDDFS(DLS):
    "Repeats depth limited search with increasing limits."

    def __init__(
        self: Self,
        init_state: NodeState,
        initial_limit: int,
        growth_factor: int,
        max_limit: Optional[int] = None,
    ):
        super().__init__(init_state, initial_limit)
        self.growth_factor = growth_factor
        self.max_limit = max_limit

    @override
    def search(self: Self) -> SearchResult:
        self.start_time = time.perf_counter()

        while self.max_limit is None or (self.limit <= self.max_limit):  # type: ignore
            try:
                result = super().search()
                return SearchResult(
                    True,
                    result.solution,
                    result.path_cost,
                    result.expanded_nodes,
                    self.frontier_size(),
                    time.perf_counter() - self.start_time,
                )
            except NoPossibleSolutions:
                self.expand_frontier()
                self.limit += self.growth_factor  # type: ignore

        raise NoPossibleSolutions()


class PrioritySearch(Search):
    "Search implementation for heap-backed priority frontiers."

    def __init__(self: Self, init_state: NodeState, eval_repeated: bool = False):
        super().__init__(init_state, eval_repeated)
        self._push_counter = count()
        self.frontier: List[tuple[float, float, int, Node]] = []
        self.best_costs: Dict[NodeState, float] = {init_state: 0}
        self._push(self.root)

    @abstractmethod
    def priority(self: Self, node: Node) -> tuple[float, float]:
        "Returns a tuple indicating the priority of the given node"
        raise NotImplementedError()

    def _push(self: Self, node: Node) -> None:
        primary, secondary = self.priority(node)
        heapq.heappush(
            self.frontier, (primary, secondary, next(self._push_counter), node)
        )

    def _get_next_node(self: Self) -> Node:
        _, _, _, node = heapq.heappop(self.frontier)
        return node

    def _mark_expanded(self: Self, node: Node) -> None:
        self.expanded_nodes += 1

    def _register_child(self: Self, child: Node) -> None:
        self.best_costs[child.state] = child.cost
        self._push(child)

    def _should_skip_child(self: Self, child: Node) -> bool:
        return (not self.eval_repeated) and child.cost >= self.best_costs.get(
            child.state, float("inf")
        )


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
