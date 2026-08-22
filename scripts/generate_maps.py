import argparse
import csv
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def add_positions(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def sub_positions(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0] - b[0], a[1] - b[1])


def interior_cells(width: int, height: int) -> List[Tuple[int, int]]:
    return [
        (x, y)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
    ]


def border_walls(width: int, height: int) -> Set[Tuple[int, int]]:
    return {
        (x, y)
        for y in range(height)
        for x in range(width)
        if x == 0 or y == 0 or x == width - 1 or y == height - 1
    }


def reachable_cells(
    start: Tuple[int, int],
    boxes: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> Set[Tuple[int, int]]:
    reached = {start}
    frontier = deque([start])

    while frontier:
        current = frontier.popleft()
        for delta in DIRECTIONS.values():
            candidate = add_positions(current, delta)
            if (
                0 <= candidate[0] < width
                and 0 <= candidate[1] < height
                and candidate not in walls
                and candidate not in boxes
                and candidate not in reached
            ):
                reached.add(candidate)
                frontier.append(candidate)

    return reached


def random_walls(
    rng: random.Random,
    width: int,
    height: int,
    wall_count: int,
) -> Set[Tuple[int, int]]:
    cells = interior_cells(width, height)
    rng.shuffle(cells)
    return set(cells[:wall_count])


def choose_open_cells(
    rng: random.Random,
    width: int,
    height: int,
    walls: Set[Tuple[int, int]],
    count: int,
) -> List[Tuple[int, int]]:
    cells = [cell for cell in interior_cells(width, height) if cell not in walls]
    if len(cells) < count:
        raise ValueError("Not enough open cells for the requested map")
    rng.shuffle(cells)
    return cells[:count]


def open_cells(width: int, height: int, walls: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [cell for cell in interior_cells(width, height) if cell not in walls]


def valid_reverse_pulls(
    player: Tuple[int, int],
    boxes: Set[Tuple[int, int]],
    walls: Set[Tuple[int, int]],
    width: int,
    height: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    reachable = reachable_cells(player, boxes, walls, width, height)
    pulls = []

    for box in boxes:
        for delta in DIRECTIONS.values():
            previous_box = sub_positions(box, delta)
            if (
                previous_box in reachable
                and previous_box not in walls
                and previous_box not in boxes
            ):
                pulls.append((box, previous_box))

    return pulls


def generate_level(
    width: int,
    height: int,
    boxes_count: int,
    reverse_steps: int,
    wall_count: int,
    seed: int,
    max_attempts: int,
) -> Tuple[List[List[str]], int]:
    rng = random.Random(seed)

    for attempt in range(1, max_attempts + 1):
        walls = border_walls(width, height) | random_walls(rng, width, height, wall_count)
        targets = set(choose_open_cells(rng, width, height, walls, boxes_count))
        boxes = set(targets)
        player_candidates = [cell for cell in open_cells(width, height, walls) if cell not in boxes]
        rng.shuffle(player_candidates)
        if not player_candidates:
            continue
        player = player_candidates[0]

        applied_steps = 0
        for _ in range(reverse_steps):
            pulls = valid_reverse_pulls(player, boxes, walls, width, height)
            if not pulls:
                break
            box, previous_box = rng.choice(pulls)
            boxes.remove(box)
            boxes.add(previous_box)
            player = box
            applied_steps += 1

        if applied_steps == 0:
            continue
        if boxes & targets or player in targets:
            continue

        board = [["E" for _ in range(width)] for _ in range(height)]
        for x, y in walls:
            board[y][x] = "W"
        for x, y in targets:
            board[y][x] = "T"
        for x, y in boxes:
            board[y][x] = "B"
        board[player[1]][player[0]] = "P"
        return board, applied_steps

    raise RuntimeError("Could not generate a valid map with the requested parameters")


def write_level(board: List[List[str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="ascii") as file:
        writer = csv.writer(file)
        writer.writerows(board)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate solvable Sokoban CSV maps")
    parser.add_argument("--width", type=int, default=9)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--boxes", type=int, default=1)
    parser.add_argument("--reverse-steps", type=int, default=12)
    parser.add_argument("--walls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="levels/generated_level.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.width < 5 or args.height < 5:
        raise ValueError("Width and height must be at least 5")
    if args.boxes < 1:
        raise ValueError("At least one box is required")

    board, applied_steps = generate_level(
        args.width,
        args.height,
        args.boxes,
        args.reverse_steps,
        args.walls,
        args.seed,
        args.max_attempts,
    )
    output = ROOT_DIR / args.output
    write_level(board, output)
    print(f"Generated {output} with {applied_steps} reverse construction steps")


if __name__ == "__main__":
    main()
