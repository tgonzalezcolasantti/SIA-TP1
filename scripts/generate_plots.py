import csv
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]

colors = ["lightcoral", "red", "sandybrown", "darkgoldenrod", "yellowgreen", "chartreuse", "deepskyblue", "royalblue", "mediumpurple", "hotpink"]
order = ["bfs", "dfs", "iddfs", "greedy sum", "greedy matching min", "greedy matching real", "astar sum", "astar matching min", "astar matching real"]

def algo_sort_key(algo: str) -> int:
    for idx, name in enumerate(order):
        if name in algo:
            return idx
    return -1

def bar_graph(output_file, data: Mapping[str, Mapping[str, float | int | List[float]]], xlabel: str, ylabel: str):
    bar_width = 1 / (len(data) + 2)
    _, ax = plt.subplots(figsize =(20, 8))
    levels = list(data[list(data.keys())[0]].keys())

    for idx, algo in enumerate(sorted(data, key=algo_sort_key)):
        bars_x = np.arange(len(data[algo])) + bar_width * idx
        algo_data = [data[algo][level] for level in levels]
        if isinstance(algo_data[0], List):
            means = [np.mean(x) for x in algo_data] # type: ignore
            errors = [np.std(x) for x in algo_data] # type: ignore
            b = plt.bar(bars_x, means, width=bar_width, color = colors[idx],
                         edgecolor='grey', label=algo, yerr=errors)
            plt.bar_label(b, [f'{x:.4f}' for x in means], padding=5, rotation=90)
        else:
            b = plt.bar(bars_x, algo_data, color=colors[idx], width = bar_width,
                    edgecolor ='grey', label=algo)
            plt.bar_label(b, algo_data, padding=5, rotation=90)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.5-bar_width for r in range(len(data[list(data.keys())[0]]))], levels)
    ax.set_yscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    plt.savefig(output_file)

def parse_csv(path):
    times: Dict[str, Dict[str, List[float]]] = {}
    expanded_nodes: Dict[str, Dict[str, int]] = {}
    frontier_nodes: Dict[str, Dict[str, int]] = {}
    cost: Dict[str, Dict[str, int]] = {}
    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for line in reader:
            algokey = f'{line["algorithm"]}{" " + line["heuristic"].replace("_", " ") if line["heuristic"] != "N/A" else ""}'
            if not algokey in times:
                times[algokey] = {}
                expanded_nodes[algokey] = {}
                frontier_nodes[algokey] = {}
                cost[algokey] = {}
            if line['level'] not in times[algokey]:
                times[algokey][line['level']] = []
                expanded_nodes[algokey][line['level']] = int(line["expanded_nodes"]) if len(line["expanded_nodes"]) > 0 else 0
                frontier_nodes[algokey][line['level']] = int(line["frontier_nodes"]) if len(line["frontier_nodes"]) > 0 else 0
                cost[algokey][line['level']] = int(line["cost"]) if len(line["cost"]) > 0 else 0
            times[algokey][line['level']].append(float(line["time_sec"]) if line["status"]=='success' else float("inf"))
    return times, expanded_nodes, frontier_nodes, cost

def main():
    times, expanded_nodes, frontier_nodes, cost = parse_csv(ROOT_DIR / 'results' / 'benchmark_results.csv')
    print(times)
    plot_folder = ROOT_DIR / 'results' / 'plots'
    bar_graph(plot_folder / 'time.png', times, "Tiempo de ejecucion por algoritmo por nivel", "Tiempo")
    bar_graph(plot_folder / 'expanded.png', expanded_nodes, "Nodos expandidos por algoritmo por nivel", "Nodos expandidos")
    bar_graph(plot_folder / 'frontier.png', frontier_nodes, "Nodos de frontera por algoritmo por nivel", "Nodos en frontera")
    bar_graph(plot_folder / 'cost.png', cost, "Costo de solucion por algoritmo por nivel", "Costo de solucion")

if __name__ == "__main__":
    main()