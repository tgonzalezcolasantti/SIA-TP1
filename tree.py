

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
        self.children: List[Node]
        self.state = node_state
        self.cost = cost
        self.depth = (parent.depth + 1) if parent else 1
        self.action = node_action
        self.heuristic_value = heuristic_value

    def is_goal(self: Self) -> bool:
        return self.state.is_solved()

    def expand(self: Self) -> List[Node]:
        self.children = []
        for move in self.state.apply_possible_moves():
            node = Node(move[0], move[1], self, 0, 0) #TODO Costs n stuff
            self.children.append(node)
        return self.children

    def __str__(self: Self) -> str:
        if self.parent:
            return f"{self.parent} -> {self.action}"
        return "Start"
