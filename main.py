import argparse
from pathlib import Path

from search import AStar, BFS, DFS, DLS, Greedy, IDDFS, NoPossibleSolutions
from sokoban import Sokoban


def build_search(name: str, sokoban: Sokoban, limit: int, eval_repeated: bool):
    if name == "bfs":
        return BFS(sokoban, eval_repeated)
    if name == "dfs":
        return DFS(sokoban, eval_repeated)
    if name == "dls":
        return DLS(sokoban, limit, eval_repeated)
    if name == "iddfs":
        return IDDFS(sokoban, initial_limit=limit, growth_factor=limit)
    if name == "greedy":
        return Greedy(sokoban, eval_repeated)
    if name == "astar":
        return AStar(sokoban, eval_repeated)
    raise ValueError(f"Unknown algorithm: {name}")


def main():
    parser = argparse.ArgumentParser(description="Run Sokoban search algorithms")
    parser.add_argument(
        "--level",
        default="level_easy.csv",
        help="Path to the CSV level file",
    )
    parser.add_argument(
        "--algorithm",
        choices=["bfs", "dfs", "dls", "iddfs", "greedy", "astar"],
        default="bfs",
        help="Search algorithm to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Depth limit for dls/iddfs",
    )
    parser.add_argument(
        "--eval-repeated",
        action="store_true",
        help="Allow repeated states instead of pruning them",
    )
    args = parser.parse_args()

    base_soko = Sokoban()
    base_soko.init_board(Path(args.level))
    search = build_search(args.algorithm, base_soko, args.limit, args.eval_repeated)

    try:
        result = search.search()
        print(result)
    except NoPossibleSolutions:
        print("No solution found")


if __name__ == "__main__":
    main()
