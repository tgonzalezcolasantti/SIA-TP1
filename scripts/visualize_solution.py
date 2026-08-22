import argparse
import html
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from main import build_search
from search import NoPossibleSolutions
from sokoban import Sokoban, SokobanGridState
from tree import Node

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


def render_html(nodes: List[Node], level: str, algorithm: str, heuristic: str) -> str:
    actions = ["Start"] + [str(node.action) for node in nodes[1:]]
    frames = [
        frame_svg(node.state, index, actions[index])
        for index, node in enumerate(nodes)
    ]
    escaped_actions = ", ".join(f'"{html.escape(action)}"' for action in actions)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sokoban Solution</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #fbfaf7;
      color: #202124;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    p {{
      margin: 0;
      color: #5f6368;
    }}
    .viewer {{
      display: grid;
      grid-template-columns: minmax(280px, max-content) minmax(220px, 1fr);
      gap: 22px;
      align-items: start;
    }}
    .board {{
      background: #ffffff;
      border: 1px solid #d8d0c2;
      padding: 10px;
      overflow: auto;
    }}
    .frame {{
      display: none;
    }}
    .frame.active {{
      display: block;
    }}
    .controls {{
      display: grid;
      gap: 12px;
      align-content: start;
    }}
    .buttons {{
      display: flex;
      gap: 8px;
    }}
    button {{
      border: 1px solid #c7ced6;
      background: #ffffff;
      color: #202124;
      padding: 8px 12px;
      border-radius: 6px;
      cursor: pointer;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .step {{
      font-size: 18px;
      font-weight: 700;
    }}
    .legend {{
      display: grid;
      grid-template-columns: repeat(2, minmax(120px, 1fr));
      gap: 8px;
      font-size: 13px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 16px;
      height: 16px;
      border: 1px solid #c7ced6;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Sokoban Solution</h1>
        <p>{html.escape(level)} | {html.escape(algorithm)} | {html.escape(heuristic)}</p>
      </div>
      <p>{len(nodes) - 1} moves</p>
    </header>
    <section class="viewer">
      <div class="board">
        {"".join(frames)}
      </div>
      <aside class="controls">
        <div class="step" id="stepLabel"></div>
        <input id="stepRange" type="range" min="0" max="{len(nodes) - 1}" value="0">
        <div class="buttons">
          <button id="prevButton">Previous</button>
          <button id="nextButton">Next</button>
        </div>
        <div class="legend">
          <span><i class="swatch" style="background:{CELL_COLORS["wall"]}"></i>Wall</span>
          <span><i class="swatch" style="background:{CELL_COLORS["floor"]}"></i>Floor</span>
          <span><i class="swatch" style="background:{CELL_COLORS["target"]}"></i>Target</span>
          <span><i class="swatch" style="background:{CELL_COLORS["box"]}"></i>Box</span>
          <span><i class="swatch" style="background:{CELL_COLORS["player"]}"></i>Player</span>
          <span><i class="swatch" style="background:{CELL_COLORS["box_on_target"]}"></i>Box on target</span>
        </div>
      </aside>
    </section>
  </main>
  <script>
    const actions = [{escaped_actions}];
    const frames = Array.from(document.querySelectorAll(".frame"));
    const range = document.getElementById("stepRange");
    const label = document.getElementById("stepLabel");

    function showStep(step) {{
      frames.forEach((frame, index) => frame.classList.toggle("active", index === step));
      range.value = step;
      label.textContent = `Step ${{step}} / ${{frames.length - 1}}: ${{actions[step]}}`;
    }}

    document.getElementById("prevButton").addEventListener("click", () => {{
      showStep(Math.max(0, Number(range.value) - 1));
    }});
    document.getElementById("nextButton").addEventListener("click", () => {{
      showStep(Math.min(frames.length - 1, Number(range.value) + 1));
    }});
    range.addEventListener("input", () => showStep(Number(range.value)));
    showStep(0);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML visualization for a Sokoban solution")
    parser.add_argument("--level", default="level_easy.csv")
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
    parser.add_argument("--output", default="results/visualizations/solution.html")
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
    output.write_text(
        render_html(node_sequence(result.solution), args.level, args.algorithm, args.heuristic),
        encoding="utf-8",
    )
    print(f"Saved visualization to {output}")


if __name__ == "__main__":
    main()
