import argparse
import html
from io import BytesIO
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from main import build_search
from search import NoPossibleSolutions
from sokoban import Sokoban, SokobanGridState
from tree import Node
from cairosvg import svg2png

CELL_COLORS = {
    "wall": "#2f3437",
    "floor": "#f4f0e8",
    "target": "#ffe066",
    "box": "#c47f35",
    "player": "#3b82f6",
    "box_on_target": "#2f9e44",
}


def node_sequence(solution: Node) -> List[Node]:
    nodes = []
    current: Optional[Node] = solution
    while current:
        nodes.append(current)
        current = current.parent
    return list(reversed(nodes))


def cell_kind(state: Sokoban, position: Tuple[int, int]) -> str:
    is_box = position in state.boxes
    is_player = position == state.player
    is_target = state.board[position] == SokobanGridState.TARGET

    if state.board[position] == SokobanGridState.WALL:
        return "wall"
    if is_player:
        return "player"
    if is_box and is_target:
        return "box_on_target"
    if is_box:
        return "box"
    if is_target:
        return "target"
    return "floor"


def frame_svg(state: Sokoban, step: int, action: str, cell_size: int = 42) -> str:
    width = state.board.shape[0] * cell_size
    height = state.board.shape[1] * cell_size
    parts = [
        f'<svg class="frame" data-step="{step}" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        f'<title>Step {step}: {html.escape(action)}</title>',
    ]

    for y in range(state.board.shape[1]):
        for x in range(state.board.shape[0]):
            kind = cell_kind(state, (x, y))
            color = CELL_COLORS[kind]
            parts.append(
                f'<rect x="{x * cell_size}" y="{y * cell_size}" '
                f'width="{cell_size}" height="{cell_size}" fill="{color}" '
                f'stroke="#d8d0c2" stroke-width="1"/>'
            )
            if kind == "target":
                cx = x * cell_size + cell_size / 2
                cy = y * cell_size + cell_size / 2
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="#f08c00"/>')
            elif kind in {"box", "box_on_target"}:
                margin = 7
                parts.append(
                    f'<rect x="{x * cell_size + margin}" y="{y * cell_size + margin}" '
                    f'width="{cell_size - 2 * margin}" height="{cell_size - 2 * margin}" '
                    f'rx="4" fill="{color}" stroke="#6b3f18" stroke-width="2"/>'
                )
            elif kind == "player":
                cx = x * cell_size + cell_size / 2
                cy = y * cell_size + cell_size / 2
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="13" fill="#1d4ed8"/>')

    parts.append("</svg>")
    return "\n".join(parts)


def render_video(nodes: List[Node], output: Path, fps: int) -> None:
    actions = ["Start"] + [str(node.action) for node in nodes[1:]]
    frames = [
        frame_svg(node.state, index, actions[index])
        for index, node in enumerate(nodes)
    ]
    # Video writer to create .avi file
    video = cv2.VideoWriter(output, cv2.VideoWriter.fourcc(*'mp4v'), fps, (1000, 1000), True)
    frames.append(frames[-1])
    # Appending images to video
    for image in frames:
        png = svg2png(bytestring=image, output_width=1000, output_height=1000)
        if png:
            pil_img = Image.open(BytesIO(png))
            cv_img = np.array(pil_img.convert('RGB'))[:, :, ::-1].copy()  # Taken from https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
            video.write(cv_img)

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    return

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML visualization for a Sokoban solution")
    parser.add_argument("--level", default="level.csv")
    parser.add_argument(
        "--algorithm",
        choices=["bfs", "dfs", "dls", "iddfs", "greedy", "astar"],
        default="astar",
    )
    parser.add_argument(
        "--heuristic",
        choices=Sokoban.available_heuristics(),
        default="matching_real_distance",
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--output", default="results/visualizations/solution.mp4")
    args = parser.parse_args()

    sokoban = Sokoban()
    sokoban.init_board(ROOT_DIR / args.level)
    sokoban.set_heuristic(args.heuristic)
    search = build_search(args.algorithm, sokoban, args.limit, False)

    try:
        result = search.search()
    except NoPossibleSolutions:
        raise SystemExit("No solution found")

    if not result.solution:
        raise SystemExit("No solution found")

    output = ROOT_DIR / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    render_video(node_sequence(result.solution), output, args.fps)
    # output.write_text(
    #     encoding="utf-8",
    # )
    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
