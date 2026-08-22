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
    BOX = 2
    TARGET = 3
    PLAYER = 4


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
        self.boxes:  List[Tuple[int, int]] = []
        self.targets: List[Tuple[int, int]] = []
        self.target_distances: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self.player: Tuple[int, int] = (0, 0)
        self.last_direction: Optional[SokobanDirection] = None
        self.heuristic_type = SokobanHeuristic.SUM_NEAREST_TARGET

    @staticmethod
    def available_heuristics() -> List[str]:
        return [heuristic.value for heuristic in SokobanHeuristic]

    def set_heuristic(self: Self, heuristic: str) -> None:
        self.heuristic_type = SokobanHeuristic(heuristic)

    def __parse_cell(self: Self, cell: str) -> SokobanGridState:
        cell = cell.upper()
        if "E" in cell:
            return SokobanGridState.EMPTY
        if "W" in cell:
            return SokobanGridState.WALL
        if "B" in cell:
            return SokobanGridState.BOX
        if "T" in cell:
            return SokobanGridState.TARGET
        if "P" in cell:
            return SokobanGridState.PLAYER
        raise ValueError("Invalid game level data")

    def init_board(self: Self, level_file: Path) -> None:
        "Initializes the board using a level file"
        board = []
        with open(level_file, "r", encoding="ascii") as file:
            reader = csv.reader(file)
            for idx_y, line in enumerate(reader):
                row = []
                for idx_x, cell in enumerate(line):
                    state = self.__parse_cell(cell)
                    if state == SokobanGridState.BOX:
                        self.boxes.append((idx_x, idx_y))
                        row.append(SokobanGridState.EMPTY)
                    elif state == SokobanGridState.PLAYER:
                        self.player = (idx_x, idx_y)
                        row.append(SokobanGridState.EMPTY)
                    elif state == SokobanGridState.TARGET:
                        self.targets.append((idx_x, idx_y))
                        row.append(state)
                    else:
                        row.append(state)
                board.append(row)
        self.board = np.rot90(np.array(board), 1)
        board_width = len(board[0])
        self.targets = [self.__rotate_position(target, board_width) for target in self.targets]
        self.boxes = [self.__rotate_position(box, board_width) for box in self.boxes]
        self.player = self.__rotate_position(self.player, board_width)
        self.target_distances = {
            target: self.__distances_from_target(target) for target in self.targets
        }

    def is_softlock(self: Self) -> bool:
        "Checks static deadlocks where at least one box can no longer reach a solution."
        return any(self.__is_box_deadlocked(box) for box in self.boxes) or self.__has_2x2_deadlock()

    def __is_box_deadlocked(self: Self, box: tuple[int, int]) -> bool:
        if self.board[box] == SokobanGridState.TARGET:
            return False
        return (
            self.__is_corner_deadlock(box)
            or self.__cannot_reach_any_target(box)
        )

    def __is_corner_deadlock(self: Self, box: tuple[int, int]) -> bool:
        horizontal_lock = (
            self.__is_blocked(self.__new_position(box, SokobanDirection.RIGHT))
            or self.__is_blocked(self.__new_position(box, SokobanDirection.LEFT))
        )
        vertical_lock = (
            self.__is_blocked(self.__new_position(box, SokobanDirection.UP))
            or self.__is_blocked(self.__new_position(box, SokobanDirection.DOWN))
        )
        return horizontal_lock and vertical_lock

    def __cannot_reach_any_target(self: Self, box: tuple[int, int]) -> bool:
        return all(box not in distances for distances in self.target_distances.values())

    def __has_2x2_deadlock(self: Self) -> bool:
        box_positions = set(self.boxes)
        for box in self.boxes:
            if self.board[box] == SokobanGridState.TARGET:
                continue
            x, y = box
            for origin in ((x - 1, y - 1), (x - 1, y), (x, y - 1), (x, y)):
                block = (
                    origin,
                    (origin[0] + 1, origin[1]),
                    (origin[0], origin[1] + 1),
                    (origin[0] + 1, origin[1] + 1),
                )
                if box not in block:
                    continue
                if any(self.__is_target(cell) for cell in block):
                    continue
                if all(self.__is_static_wall(cell) or cell in box_positions for cell in block):
                    return True
        return False

    def __is_blocked(self: Self, position: tuple[int, int]) -> bool:
        if not self.__is_inside_board(position):
            return True
        return self.board[position] == SokobanGridState.WALL

    def __is_static_wall(self: Self, position: tuple[int, int]) -> bool:
        return not self.__is_inside_board(position) or self.board[position] == SokobanGridState.WALL

    def __is_target(self: Self, position: tuple[int, int]) -> bool:
        return self.__is_inside_board(position) and self.board[position] == SokobanGridState.TARGET

    def __move_box(self: Self, box: tuple[int, int], direction: SokobanDirection) -> bool:
        new_position = self.__new_position(box, direction)
        if not self.__is_inside_board(new_position):
            return False
        if self.board[new_position] == SokobanGridState.WALL:
            return False
        if new_position in self.boxes:
            return False
        self.boxes.remove(box)
        self.boxes.append(new_position)
        return True


    @staticmethod
    def __new_position(position: tuple[int, int], direction: SokobanDirection) -> tuple[int, int]:
        if direction == SokobanDirection.UP:
            return (position[0], position[1]-1)
        if direction == SokobanDirection.DOWN:
            return (position[0], position[1]+1)
        if direction == SokobanDirection.LEFT:
            return (position[0]-1, position[1])
        if direction == SokobanDirection.RIGHT:
            return (position[0]+1, position[1])
        raise ValueError()

    @staticmethod
    def __rotate_position(position: tuple[int, int], board_width: int) -> tuple[int, int]:
        return (board_width - position[0] - 1, position[1])

    def __is_inside_board(self: Self, position: tuple[int, int]) -> bool:
        return (
            0 <= position[0] < self.board.shape[0]
            and 0 <= position[1] < self.board.shape[1]
        )

    def __distances_from_target(self: Self, target: tuple[int, int]) -> Dict[Tuple[int, int], int]:
        distances = {target: 0}
        frontier = deque([target])

        while frontier:
            current = frontier.popleft()
            for direction in SokobanDirection:
                candidate = self.__new_position(current, direction)
                if (
                    not self.__is_static_wall(candidate)
                    and candidate not in distances
                ):
                    distances[candidate] = distances[current] + 1
                    frontier.append(candidate)

        return distances

    def move(self: Self, direction: SokobanDirection) -> bool:
        "Applies a movement to the player, if possible"
        new_position = self.__new_position(self.player, direction)
        if not self.__is_inside_board(new_position):
            return False
        if self.board[new_position] == SokobanGridState.WALL:
            return False
        if new_position in self.boxes:
            if not self.__move_box(new_position, direction):
                return False
        self.player = new_position
        self.last_direction = direction
        return True


    def copy(self: Self) -> Sokoban:
        "Clones this sokoban state"
        soko = Sokoban()
        soko.board = self.board # Immutable, no need to clone
        soko.targets = self.targets
        soko.target_distances = self.target_distances
        soko.boxes = list(self.boxes) # Mutable, must clone!
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
        if not self.boxes:
            return 0
        if len(self.targets) < len(self.boxes):
            return float("inf")

        sorted_boxes = tuple(sorted(self.boxes))
        sorted_targets = tuple(sorted(self.targets))
        best_distance = float("inf")

        for target_assignment in permutations(sorted_targets, len(sorted_boxes)):
            total_distance = 0
            for box, target in zip(sorted_boxes, target_assignment):
                total_distance += abs(box[0] - target[0]) + abs(box[1] - target[1])
                if total_distance >= best_distance:
                    break
            else:
                best_distance = total_distance

        return best_distance

    def __matching_real_distance(self: Self) -> float:
        "Admissible lower bound using BFS-preprocessed distances through the board."
        if not self.boxes:
            return 0
        if len(self.targets) < len(self.boxes):
            return float("inf")

        sorted_boxes = tuple(sorted(self.boxes))
        sorted_targets = tuple(sorted(self.targets))
        best_distance = float("inf")

        for target_assignment in permutations(sorted_targets, len(sorted_boxes)):
            total_distance = 0
            for box, target in zip(sorted_boxes, target_assignment):
                distance = self.target_distances[target].get(box, float("inf"))
                total_distance += distance
                if total_distance >= best_distance:
                    break
            else:
                best_distance = total_distance

        return best_distance

    def __hash__(self: Self) -> int:
        return hash((self.player, tuple(sorted(self.boxes))))

    def __eq__(self: Self, value: object) -> bool:
        if isinstance(value, Sokoban):
            return self.player == value.player and set(self.boxes) == set(value.boxes)
        return False

