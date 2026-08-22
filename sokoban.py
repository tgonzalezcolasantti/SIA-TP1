import collections
import csv
from enum import Enum
from pathlib import Path
from typing import List, Optional, Self, Tuple
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
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __str__(self: Self) -> str:
        return self.name

class Sokoban(NodeState["Sokoban", "SokobanDirection"]):
    "Holds current level state"
    def __init__(self: Self) -> None:
        self.board: np.ndarray
        self.boxes:  List[Tuple[int, int]] = []
        self.player: Tuple[int, int] = (0, 0)
        self.last_direction: Optional[SokobanDirection] = None

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
                    else:
                        row.append(state)
                board.append(row)
        self.board = np.array(board)

    def is_softlock(self: Self) -> bool:
        "Checks for a softlock where a box is not in target and cannot be moved"
        for box in self.boxes:
            if self.board[box] != SokobanGridState.TARGET:
                try:
                    horizontal_lock = SokobanGridState.WALL in (
                        self.__new_position(box, SokobanDirection.RIGHT),
                        self.__new_position(box, SokobanDirection.LEFT),
                    )
                except IndexError:
                    horizontal_lock = True
                try:
                    vertical_lock = SokobanGridState.WALL in (
                        self.__new_position(box, SokobanDirection.UP),
                        self.__new_position(box, SokobanDirection.DOWN),
                    )
                except IndexError:
                    vertical_lock = True
                if horizontal_lock and vertical_lock:
                    return True
        return False


    def __move_box(self: Self, box: tuple[int, int], direction: SokobanDirection) -> bool:
        new_position = self.__new_position(box, direction)
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


    def move(self: Self, direction: SokobanDirection) -> bool:
        "Applies a movement to the player, if possible"
        new_position = self.__new_position(self.player, direction)
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
        soko.boxes = list(self.boxes) # Mutable, must clone!
        soko.player = self.player
        return soko


    def apply_possible_moves(self: Self) -> List[Tuple[Sokoban, SokobanDirection]]:
        "Applies all possible movements and returns a list of board statuses and applied movements"
        if self.is_softlock():
            return []
        moves = []
        for move in SokobanDirection:
            if move == self.last_direction:
                continue #Prevent backtracking
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


    def __eq__(self: Self, value: object) -> bool:
        if isinstance(value, Sokoban):
            return self.player == value.player and set(self.boxes) == set(value.boxes)
        return False

