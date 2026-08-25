import csv
from collections import deque
from enum import Enum
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple

import numpy as np

from tree import NodeState


class SokobanGridState(Enum):
    "Possible cell states to parse from level file"

    EMPTY = 0
    WALL = 1
    TARGET = 2

    @staticmethod
    def from_cell(cell: str) -> SokobanGridState:
        if "W" in cell:
            return SokobanGridState.WALL
        if "T" in cell:
            return SokobanGridState.TARGET
        return SokobanGridState.EMPTY


class SokobanDirection(Enum):
    "Directions to move player"

    UP = 1
    RIGHT = -2
    DOWN = -1
    LEFT = 2

    def __str__(self: Self) -> str:
        return self.name

    def reverse(self: Self) -> SokobanDirection:
        return SokobanDirection(-self.value)


class SokobanHeuristic(Enum):
    "Available Sokoban heuristics"

    ZERO = "zero"
    SUM_NEAREST_TARGET = "sum_nearest_target"
    MATCHING_MIN_DISTANCE = "matching_min_distance"
    MATCHING_REAL_DISTANCE = "matching_real_distance"


class Sokoban(NodeState["Sokoban", "SokobanDirection"]):
    "Holds current level state"

    def __init__(self: Self) -> None:
        self.board: np.ndarray
        self.boxes: List[Tuple[int, int]] = []
        self.targets: List[Tuple[int, int]] = []
        self.target_distances: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self.player: Tuple[int, int] = (0, 0)
        self.last_direction: Optional[SokobanDirection] = None
        self.heuristic_type = SokobanHeuristic.ZERO
        self.last_pushed_box: bool = False

    @staticmethod
    def available_heuristics() -> List[str]:
        return [heuristic.value for heuristic in SokobanHeuristic]

    def set_heuristic(self: Self, heuristic: str) -> None:
        self.heuristic_type = SokobanHeuristic(heuristic)

    def __parse_cell(self: Self, cell: str) -> Tuple[SokobanGridState, bool, bool]:
        cell = cell.upper()
        return (SokobanGridState.from_cell(cell), "B" in cell, "P" in cell)

    def init_board(self: Self, level_file: Path) -> None:
        "Initializes the board using a level file"
        board = []
        with open(level_file, "r", encoding="ascii") as file:
            reader = csv.reader(file)
            for idx_y, line in enumerate(reader):
                row = []
                for idx_x, cell in enumerate(line):
                    state, is_box, is_player = self.__parse_cell(cell)
                    if is_box:
                        self.boxes.append((idx_x, idx_y))
                    if is_player:
                        self.player = (idx_x, idx_y)
                    if state == SokobanGridState.TARGET:
                        self.targets.append((idx_x, idx_y))
                    row.append(state)
                board.append(row)
        self.board = np.rot90(np.fliplr(np.array(board)), 1)  # black magic
        self.target_distances = {
            target: self.__distances_from_target(target) for target in self.targets
        }

    def is_softlock(self: Self) -> bool:
        "Checks static deadlocks where at least one box can no longer reach a solution."
        return any(self.__is_box_deadlocked(box) for box in self.boxes)

    def __is_box_deadlocked(self: Self, box: tuple[int, int]) -> bool:
        if self.board[box] == SokobanGridState.TARGET:
            return False
        return self.__is_corner_deadlock(box)

    def __is_corner_deadlock(self: Self, box: tuple[int, int]) -> bool:
        horizontal_lock = self.__is_blocked(
            self.__new_position(box, SokobanDirection.RIGHT)
        ) or self.__is_blocked(self.__new_position(box, SokobanDirection.LEFT))
        vertical_lock = self.__is_blocked(
            self.__new_position(box, SokobanDirection.UP)
        ) or self.__is_blocked(self.__new_position(box, SokobanDirection.DOWN))
        return horizontal_lock and vertical_lock

    def __is_blocked(self: Self, position: tuple[int, int]) -> bool:
        return self.board[position] == SokobanGridState.WALL or position in self.boxes

    def __is_static_wall(self: Self, position: tuple[int, int]) -> bool:
        return self.board[position] == SokobanGridState.WALL

    def __move_box(
        self: Self, box: tuple[int, int], direction: SokobanDirection
    ) -> bool:
        new_position = self.__new_position(box, direction)
        if self.board[new_position] == SokobanGridState.WALL:
            return False
        if new_position in self.boxes:
            return False
        self.boxes.remove(box)
        self.boxes.append(new_position)
        return True

    @staticmethod
    def __new_position(
        position: tuple[int, int], direction: SokobanDirection
    ) -> tuple[int, int]:
        if direction == SokobanDirection.UP:
            return (position[0], position[1] - 1)
        if direction == SokobanDirection.DOWN:
            return (position[0], position[1] + 1)
        if direction == SokobanDirection.LEFT:
            return (position[0] - 1, position[1])
        if direction == SokobanDirection.RIGHT:
            return (position[0] + 1, position[1])
        raise ValueError()

    def __distances_from_target(
        self: Self, target: tuple[int, int]
    ) -> Dict[Tuple[int, int], int]:
        distances = {target: 0}
        frontier = deque([target])

        while frontier:
            current = frontier.popleft()
            for direction in SokobanDirection:
                candidate = self.__new_position(current, direction)
                if not self.__is_static_wall(candidate) and candidate not in distances:
                    distances[candidate] = distances[current] + 1
                    frontier.append(candidate)

        return distances

    def move(self: Self, direction: SokobanDirection) -> bool:
        "Applies a movement to the player, if possible"
        new_position = self.__new_position(self.player, direction)
        if self.board[new_position] == SokobanGridState.WALL:
            return False
        if new_position in self.boxes:
            if not self.__move_box(new_position, direction):
                return False
            self.last_pushed_box = True
        self.player = new_position
        self.last_direction = direction
        return True

    def copy(self: Self) -> Sokoban:
        "Clones this sokoban state"
        soko = Sokoban()
        soko.board = self.board
        soko.targets = self.targets
        soko.target_distances = self.target_distances
        soko.boxes = list(self.boxes)
        soko.player = self.player
        soko.last_direction = self.last_direction
        soko.heuristic_type = self.heuristic_type
        return soko

    def apply_possible_moves(self: Self) -> List[Tuple[Sokoban, SokobanDirection]]:
        "Applies all possible movements and returns a list of board statuses and applied movements"
        if self.is_softlock():
            return []
        moves = []
        for move in SokobanDirection:
            if self.last_direction and move == self.last_direction.reverse() and not self.last_pushed_box:
                continue
            soko = self.copy()
            if soko.move(move):
                moves.append((soko, move))
        return moves

    def is_solved(self: Self) -> bool:
        "Checks to see if this board is solved"
        for box in self.boxes:
            if self.board[box] != SokobanGridState.TARGET:
                return False
        return True

    def heuristic(self: Self) -> float:
        if self.heuristic_type == SokobanHeuristic.ZERO:
            return 0
        if self.heuristic_type == SokobanHeuristic.MATCHING_MIN_DISTANCE:
            return self.__matching_min_distance()
        if self.heuristic_type == SokobanHeuristic.MATCHING_REAL_DISTANCE:
            return self.__matching_real_distance()
        return self.__sum_nearest_target_distance()

    def __sum_nearest_target_distance(self: Self) -> float:
        "Admissible lower bound: sum of each box distance to its nearest target."
        total = 0
        for box in self.boxes:
            if self.board[box] == SokobanGridState.TARGET:
                continue
            total += min(
                abs(box[0] - target[0]) + abs(box[1] - target[1])
                for target in self.targets
            )
        return total

    def __matching_min_distance(self: Self) -> float:
        "Admissible lower bound using the minimum box-target distance assignment."
        best_distance = float("inf")

        for target_assignment in permutations(self.targets, len(self.boxes)):
            total_distance = 0
            for box, target in zip(self.boxes, target_assignment):
                total_distance += abs(box[0] - target[0]) + abs(box[1] - target[1])
                if total_distance >= best_distance:
                    break
            else:
                best_distance = total_distance

        return best_distance

    def __matching_real_distance(self: Self) -> float:
        "Admissible lower bound using BFS-preprocessed distances through the board."
        best_distance = float("inf")

        for target_assignment in permutations(self.targets, len(self.boxes)):
            total_distance = 0
            for box, target in zip(self.boxes, target_assignment):
                total_distance += self.target_distances[target].get(box, float("inf"))
                if total_distance >= best_distance:
                    break
            else:
                best_distance = total_distance

        return best_distance

    def __hash__(self: Self) -> int:
        return hash((self.player, tuple(sorted(self.boxes))))

    def __eq__(self: Self, value: object) -> bool:
        if isinstance(value, Sokoban):
            if self.player == value.player:
                for box in self.boxes:
                    if box not in value.boxes:
                        return False
                return True
        return False

    def __cell_to_char(self: Self, cell: Tuple[int, int]) -> str:
        x, y = cell
        if self.board[x, y] == SokobanGridState.WALL:
            return "#"
        if self.board[x, y] == SokobanGridState.TARGET:
            if (x, y) not in self.boxes:
                return "@"
            return "$"
        if self.player == (x, y):
            return "P"
        if (x, y) in self.boxes:
            return "X"
        return " "

    def __str__(self: Self) -> str:
        res = ""
        for y in range(self.board.shape[1]):
            for x in range(self.board.shape[0]):
                res += self.__cell_to_char((x, y)) + " "
            res += "\n"
        return res
