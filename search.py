from typing import List, Optional, Self


class NoPossibleSolutions(BaseException):
    "Exception to raise when search exhausted all paths and no solution was found"


class NodeState:
    pass


class NodeAction:
    pass


class Node:
    def __init__(
        self: Self,
        state: NodeState,
        action: Optional[NodeAction],
        parent: Optional[Node],
        cost: float,
    ):
        self.parent = parent
        self.children: List[Node]
        self.state = state
        self.cost = cost
        self.depth = (parent.depth + 1) if parent else 1
        self.action = action

    def is_goal(self: Self) -> bool:
        raise NotImplementedError()

    def expand(self: Self) -> List[Node]:
        # TODO Set parent-child relationship of new nodes
        raise NotImplementedError()


class Search:
    "Base class for searches"

    def __init__(self: Self, eval_repeated: bool = False):
        self.root: Node  # Tree is modeled linking nodes
        self.frontier: List[Node] = []
        self.explored_nodes: List[Node] = []
        self.eval_repeated = eval_repeated

    def __search_node(self: Self) -> Optional[Node]:
        try:
            node = self.frontier.pop(0)
        except IndexError as e:
            raise NoPossibleSolutions() from e
        if node.is_goal():
            return node
        children = node.expand()
        self.explored_nodes.append(node)

        if self.eval_repeated:
            self.frontier.extend(children)
        else:
            for new_node in children:
                for old_node in self.explored_nodes:
                    # smart exclude: if shallower, it might be a better solution
                    # so we include it anyways. If deeper, we can safely ignore
                    if (
                        new_node.depth >= old_node.depth
                        and new_node.state == old_node.state
                    ):
                        break # We decided to exclude the node
                else:
                    # Finished loop without excluding -> include
                    self.frontier.append(new_node)
        self.reorder_fr()
        return None

    def reorder_fr(self: Self) -> None:
        "Reorders current frontier depending on search method used"
        raise NotImplementedError()

    def search(self: Self) -> Node:
        "Performs the search and returns a solution or raises NoPossibleSolutions"
        result: Optional[Node] = None
        while not result:
            result = self.__search_node()
        return result


class BFS(Search):
    "Explores least deep nodes first"

    def reorder_fr(self: Self) -> None:
        self.frontier.sort(key=lambda x: x.depth)


# class UCS(Search):
#     "Explores cheaper nodes first"
#     def reorder_fr(self: Self) -> None:
#         self.frontier.sort(key=lambda x: x.cost)


class DFS(Search):
    "Explores deepest nodes first"

    def reorder_fr(self: Self) -> None:
        self.frontier.sort(key=lambda x: x.depth, reverse=True)


class DLS(DFS, Search):
    "Depth limit search, like DFS but _limited_"

    def __init__(
        self: Self,
        limit: int,
    ):
        super().__init__()
        self.limit = limit
        self.node_dump: List[Node] = []

    def reorder_fr(self: Self) -> None:
        new_frontier = []
        for x in self.frontier:
            if x.depth <= self.limit:
                new_frontier.append(x)
            else:
                self.node_dump.append(x)
        self.frontier = new_frontier
        super().reorder_fr()


class IDDFS(DLS, Search):
    "DLS until limit, then expands limit and re-explores"

    def __init__(
        self: Self,
        initial_limit: int,
        growth_factor: int,
        max_limit: Optional[int] = None,
    ):
        super().__init__(initial_limit)
        self.growth_factor = growth_factor
        self.max_limit = max_limit

    def reorder_fr(self: Self) -> None:
        if not self.frontier and (not self.max_limit or self.limit < self.max_limit):
            self.limit += self.growth_factor
            if self.max_limit and self.limit > self.max_limit:
                self.limit = self.max_limit
            self.__rebuild_frontier()
        super().reorder_fr()

    def __rebuild_frontier(self: Self) -> None:
        self.frontier = self.node_dump
        self.node_dump.clear()
