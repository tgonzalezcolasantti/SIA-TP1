
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, List, Optional, Self, Tuple, TypeVar

TNodeState = TypeVar("TNodeState", bound="NodeState")
TNodeAction = TypeVar("TNodeAction", bound="Enum")


class NodeState(ABC, Generic[TNodeState, TNodeAction]):
    @abstractmethod
    def is_solved(self: Self) -> bool:
        pass

    @abstractmethod
    def apply_possible_moves(self: Self) -> List[Tuple[TNodeState, TNodeAction]]:
        pass

    def get_cost(self: Self, action: TNodeAction) -> float:
        return 1

    def heuristic(self: Self) -> float:
        return 0


class Node:
    def __init__(
        self: Self,
        node_state: NodeState,
        node_action: Optional[Enum],
        parent: Optional[Node],
        cost: float,
        heuristic_value: float,
    ):
        self.parent = parent
        self.state = node_state
        self.cost = cost
        self.depth = (parent.depth + 1) if parent else 1
        self.action = node_action
        self.heuristic_value = heuristic_value

    def __lt__(self: Self, other: "Node") -> bool:
        if self.heuristic_value + self.cost == other.heuristic_value + other.cost:
            return self.heuristic_value < other.heuristic_value
        return self.heuristic_value + self.cost < other.heuristic_value + other.cost

    def is_goal(self: Self) -> bool:
        return self.state.is_solved()

    def expand(self: Self) -> List[Node]:
        children = []
        for child_state, action in self.state.apply_possible_moves():
            node = Node(
                child_state,
                action,
                self,
                self.cost + self.state.get_cost(action),
                child_state.heuristic(),
            )
            children.append(node)
        return children

    def path(self: Self) -> List[Enum]:
        actions = []
        node = self
        while node.parent:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def __str__(self: Self) -> str:
        return "Start" + "".join(f" -> {action}" for action in self.path())
