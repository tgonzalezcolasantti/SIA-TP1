import argparse
import csv
import multiprocessing
import sys
import time
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from main import build_search
from search import NoPossibleSolutions
from sokoban import Sokoban

DEFAULT_ALGORITHMS = ["bfs", "dfs", "greedy", "astar"]
DEFAULT_HEURISTICS = ["sum_nearest_target", "matching_min_distance"]
INFORMED_ALGORITHMS = {"greedy", "astar"}
FIELDNAMES = [
    "level",
    "algorithm",
    "heuristic",
    "status",
    "steps",
    "cost",
    "expanded_nodes",
    "frontier_nodes",
    "time_sec",
    "timeout_sec",
]


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_case(
    level: str,
    algorithm: str,
    heuristic: Optional[str],
    limit: int,
    eval_repeated: bool,
) -> Dict[str, object]:
    sokoban = Sokoban()
    sokoban.init_board(ROOT_DIR / level)
    if heuristic:
        sokoban.set_heuristic(heuristic)

    search = build_search(algorithm, sokoban, limit, eval_repeated)
    started_at = time.perf_counter()

    try:
        result = search.search()
        return {
            "level": level,
            "algorithm": algorithm,
            "heuristic": heuristic or "N/A",
            "status": "success" if result.success else "no_solution",
            "steps": len(result.solution_path),
            "cost": result.path_cost,
            "expanded_nodes": result.expanded_nodes,
            "frontier_nodes": result.frontier_nodes,
            "time_sec": result.processing_time_sec,
        }
    except NoPossibleSolutions:
        return {
            "level": level,
            "algorithm": algorithm,
            "heuristic": heuristic or "N/A",
            "status": "no_solution",
            "steps": "",
            "cost": "",
            "expanded_nodes": search.expanded_nodes,
            "frontier_nodes": search.frontier_size(),
            "time_sec": time.perf_counter() - started_at,
        }


def run_case_worker(queue, level, algorithm, heuristic, limit, eval_repeated) -> None:
    try:
        queue.put(run_case(level, algorithm, heuristic, limit, eval_repeated))
    except Exception as exc:
        queue.put({
            "level": level,
            "algorithm": algorithm,
            "heuristic": heuristic or "N/A",
            "status": f"error: {exc}",
            "steps": "",
            "cost": "",
            "expanded_nodes": "",
            "frontier_nodes": "",
            "time_sec": "",
        })


def run_with_timeout(
    level: str,
    algorithm: str,
    heuristic: Optional[str],
    limit: int,
    eval_repeated: bool,
    timeout: float,
) -> Dict[str, object]:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=run_case_worker,
        args=(queue, level, algorithm, heuristic, limit, eval_repeated),
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "level": level,
            "algorithm": algorithm,
            "heuristic": heuristic or "N/A",
            "status": "timeout",
            "steps": "",
            "cost": "",
            "expanded_nodes": "",
            "frontier_nodes": "",
            "time_sec": timeout,
        }

    try:
        return queue.get_nowait()
    except Empty:
        return {
            "level": level,
            "algorithm": algorithm,
            "heuristic": heuristic or "N/A",
            "status": "error: empty worker result",
            "steps": "",
            "cost": "",
            "expanded_nodes": "",
            "frontier_nodes": "",
            "time_sec": "",
        }


def benchmark_cases(levels: List[str], algorithms: List[str], heuristics: List[str]) -> List[tuple[str, str, Optional[str]]]:
    cases = []
    for level in levels:
        for algorithm in algorithms:
            if algorithm in INFORMED_ALGORITHMS:
                for heuristic in heuristics:
                    cases.append((level, algorithm, heuristic))
            else:
                cases.append((level, algorithm, None))
    return cases


def write_results(rows: List[Dict[str, object]], output: Path, timeout: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            row = {**row, "timeout_sec": timeout}
            writer.writerow(row)


def print_row(row: Dict[str, object]) -> None:
    heuristic = row["heuristic"]
    time_sec = row["time_sec"]
    time_text = f"{time_sec:.4f}s" if isinstance(time_sec, float) else "-"
    print(
        f"{row['level']:<15} {row['algorithm']:<8} {heuristic:<22} "
        f"{row['status']:<12} cost={row['cost']!s:<5} "
        f"expanded={row['expanded_nodes']!s:<8} time={time_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparative Sokoban search benchmarks")
    parser.add_argument(
        "--levels",
        default="level_easy.csv,level.csv",
        help="Comma-separated level files",
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithms",
    )
    parser.add_argument(
        "--heuristics",
        default=",".join(DEFAULT_HEURISTICS),
        help="Comma-separated heuristics for greedy/astar",
    )
    parser.add_argument("--limit", type=int, default=30, help="Depth limit for dls/iddfs")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout per run in seconds")
    parser.add_argument(
        "--eval-repeated",
        action="store_true",
        help="Allow repeated states instead of pruning them",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_results.csv",
        help="CSV output path",
    )
    args = parser.parse_args()

    levels = parse_csv_arg(args.levels)
    algorithms = parse_csv_arg(args.algorithms)
    heuristics = parse_csv_arg(args.heuristics)
    rows = []

    for level, algorithm, heuristic in benchmark_cases(levels, algorithms, heuristics):
        row = run_with_timeout(
            level,
            algorithm,
            heuristic,
            args.limit,
            args.eval_repeated,
            args.timeout,
        )
        rows.append(row)
        print_row(row)

    output = ROOT_DIR / args.output
    write_results(rows, output, args.timeout)
    print(f"\nSaved benchmark results to {output}")


if __name__ == "__main__":
    main()
