from pathlib import Path

from search import BFS, DFS, DLS
from sokoban import Sokoban


def main():
    base_soko = Sokoban()
    base_soko.init_board(Path("./level_easy.csv"))
    ans = BFS(base_soko, False).search()
    print(ans)


if __name__ == "__main__":
    main()
