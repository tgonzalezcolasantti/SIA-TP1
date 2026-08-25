# SIA TP1 - Metodos de Busqueda

Trabajo Practico 1 de la materia Sistemas de Inteligencia Artificial.

El proyecto implementa un motor de busqueda para resolver niveles de Sokoban. A partir
de un tablero inicial, el agente debe mover las cajas hasta las posiciones objetivo,
intentando minimizar la cantidad de movimientos realizados.

## Descripcion

Sokoban se modela como un problema de busqueda en espacio de estados:

- Estado inicial: tablero cargado desde un archivo `.csv`.
- Estado: posicion del jugador, posiciones de cajas, tablero fijo y ultimo movimiento.
- Acciones: movimientos `UP`, `DOWN`, `LEFT` y `RIGHT`.
- Transicion: una accion mueve al jugador y, si corresponde, empuja una caja.
- Costo: cada movimiento valido tiene costo 1.
- Objetivo: todas las cajas deben quedar ubicadas sobre casilleros objetivo.

El ambiente es completamente observable, determinista, secuencial, estatico y discreto,
por lo que resulta adecuado para metodos de busqueda clasica.

## Caracteristicas

- Implementacion de Sokoban como problema de busqueda.
- Representacion de nodos con padre, accion, costo, profundidad y heuristica.
- Busquedas desinformadas: BFS, DFS e IDDFS.
- Busquedas informadas: Greedy y A*.
- Tres heuristicas disponibles para los metodos informados.
- Deteccion simple de deadlocks por cajas trabadas en esquinas.
- Evitacion de estados repetidos.
- Ejecucion por consola con argumentos.
- Benchmark automatico con multiples niveles, algoritmos y heuristicas.
- Benchmark paralelizable con timeout por corrida, util para niveles costosos.
- Generacion de graficos comparativos en escala logaritmica.
- Generacion de video `.mp4` con la solucion encontrada.

## Estructura del proyecto

```text
SIA-TP1/
|-- main.py                         # CLI principal para ejecutar una busqueda
|-- sokoban.py                      # Modelo del juego, estados, acciones y heuristicas
|-- search.py                       # BFS, DFS, DLS, IDDFS, Greedy y A*
|-- tree.py                         # Nodo generico y contrato de estado buscable
|-- level_easy.csv                  # Nivel simple
|-- level_mid.csv                   # Nivel intermedio
|-- level.csv                       # Nivel base
|-- level_2.csv                     # Nivel adicional
|-- level_3.csv                     # Nivel adicional
|-- level_4.csv                     # Nivel adicional
|-- scripts/
|   |-- run_benchmarks.py           # Corridas comparativas y exportacion CSV
|   |-- generate_plots.py           # Graficos a partir del CSV de benchmark
|   `-- visualize_solution.py       # Video de una solucion
|-- results/
|   |-- benchmark_results.csv       # Resultados de benchmark
|   |-- plots/                      # Graficos generados
|   `-- visualizations/             # Videos generados
|-- pyproject.toml                  # Dependencias del proyecto
`-- README.md
```

## Requisitos

El proyecto usa Python y las dependencias declaradas en `pyproject.toml`.

Instalacion recomendada con `uv`:

```bash
uv sync
```

Alternativamente, instalar las dependencias principales con `pip`:

```bash
python -m pip install numpy rich matplotlib pillow opencv-python cairosvg
```

## Uso rapido

Ejecutar A* sobre el nivel facil:

```bash
python main.py --level level_easy.csv --algorithm astar --heuristic matching_real_distance
```

Ejecutar BFS:

```bash
python main.py --level level_easy.csv --algorithm bfs
```

Ejecutar Greedy con una heuristica especifica:

```bash
python main.py --level level_easy.csv --algorithm greedy --heuristic matching_min_distance
```

Ejecutar IDDFS indicando limite inicial:

```bash
python main.py --level level_easy.csv --algorithm iddfs --limit 20
```

## Parametros de ejecucion

`main.py` recibe los siguientes argumentos:

```text
--level            Archivo CSV del nivel a resolver.
--algorithm        Algoritmo de busqueda: bfs, dfs, dls, iddfs, greedy o astar.
--heuristic        Heuristica para busquedas informadas.
--limit            Limite de profundidad para dls/iddfs.
--eval-repeated    Permite evaluar estados repetidos.
```

Ejemplo completo:

```bash
python main.py --level level_mid.csv --algorithm astar --heuristic matching_real_distance --limit 30
```

## Salida del programa

Al finalizar, la ejecucion informa:

- resultado de la busqueda;
- costo de la solucion;
- cantidad de nodos expandidos;
- cantidad de nodos que quedaron en frontera;
- tiempo de procesamiento;
- solucion como camino desde el estado inicial hasta el objetivo.

Ejemplo:

```text
Start -> UP -> UP -> UP
Cost: 3
Expanded nodes: 3
Frontier nodes: 7
Time: 0.0001s
```

## Algoritmos implementados

### BFS

Expande primero los nodos de menor profundidad. Como todas las acciones cuestan 1,
encuentra una solucion optima si existe, aunque puede consumir mucha memoria.

### DFS

Expande primero los nodos de mayor profundidad. Suele usar menos memoria que BFS,
pero no garantiza optimalidad.

### Greedy

Ordena la frontera usando solo la heuristica `h(n)`. Suele ser rapido, pero no
garantiza encontrar la solucion de menor costo.

### A*

Ordena la frontera usando:

```text
f(n) = g(n) + h(n)
```

Donde `g(n)` es el costo acumulado y `h(n)` es la estimacion heuristica hasta el
objetivo. Con heuristicas admisibles y costos positivos, A* encuentra soluciones
optimas.

### IDDFS

Ejecuta busquedas en profundidad con limites crecientes. Esta incluido como metodo
opcional de la consigna.

## Heuristicas

El proyecto incluye las siguientes heuristicas:

```text
zero
sum_nearest_target
matching_min_distance
matching_real_distance
```

### zero

Siempre devuelve 0. Sirve como baseline y vuelve a A* equivalente a una busqueda por
costo acumulado.

### sum_nearest_target

Suma, para cada caja, la distancia Manhattan al objetivo mas cercano.

Es admisible porque ignora restricciones reales del Sokoban, como la posicion del
jugador, paredes que obligan rodeos y la direccion necesaria para empujar una caja.

### matching_min_distance

Busca una asignacion caja-objetivo que minimice la suma total de distancias Manhattan.
Evita contar varias cajas contra el mismo objetivo como si todas pudieran ocuparlo.

Tambien es admisible porque sigue siendo una relajacion del problema real.

### matching_real_distance

Precalcula distancias desde los objetivos mediante BFS sobre el tablero fijo, teniendo
en cuenta paredes pero ignorando cajas moviles. Luego busca la mejor asignacion entre
cajas y objetivos usando esas distancias.

Es la heuristica mas informada de las implementadas y tambien funciona como cota
inferior del costo real.

## Formato de niveles

Los niveles se definen como archivos CSV. Cada celda puede contener:

```text
W  pared
E  espacio vacio
T  objetivo
B  caja
P  jugador
```

Ejemplo:

```csv
W,W,W,W,W
W,E,T,E,W
W,E,E,E,W
W,E,B,E,W
W,E,E,E,W
W,E,P,E,W
W,W,W,W,W
```

## Benchmarks

El script `scripts/run_benchmarks.py` ejecuta combinaciones de niveles, algoritmos y
heuristicas. Guarda los resultados en formato CSV y usa procesos separados para poder
cortar corridas largas por timeout.

Ejemplo:

```bash
python scripts/run_benchmarks.py --levels level_easy.csv,level_mid.csv --runs 5 --algorithms bfs,dfs,iddfs,greedy,astar --heuristics sum_nearest_target,matching_min_distance,matching_real_distance --limit 20 --timeout 120 --output results/benchmark_results.csv --tasks 8
```

Parametros utiles:

```text
--levels       Niveles separados por coma.
--runs         Cantidad de repeticiones por combinacion.
--algorithms   Algoritmos separados por coma.
--heuristics   Heuristicas para Greedy y A*.
--limit        Limite de profundidad para DLS/IDDFS.
--timeout      Tiempo maximo por corrida, en segundos.
--tasks        Cantidad de tareas paralelas.
--output       Archivo CSV de salida.
```

Columnas generadas:

```text
run, level, algorithm, heuristic, status, steps, cost,
expanded_nodes, frontier_nodes, time_sec
```

Estos datos sirven para comparar:

- costo de la solucion;
- tiempo de procesamiento;
- nodos expandidos;
- nodos en frontera;
- exito, fracaso o timeout.

## Graficos

Luego de correr benchmarks, se pueden generar graficos comparativos:

```bash
python scripts/generate_plots.py
```

Los graficos se guardan en:

```text
results/plots/
```

Archivos generados:

```text
time.png       Tiempo de ejecucion por algoritmo y nivel.
expanded.png   Nodos expandidos por algoritmo y nivel.
frontier.png   Nodos de frontera por algoritmo y nivel.
cost.png       Costo de solucion por algoritmo y nivel.
```

## Visualizacion

Para generar un video de la solucion encontrada:

```bash
python scripts/visualize_solution.py --level level_easy.csv --algorithm astar --heuristic matching_real_distance --fps 3 --output results/visualizations/solution.mp4
```

El archivo se guarda en:

```text
results/visualizations/solution.mp4
```

El video se construye generando un frame SVG por cada estado de la solucion y
convirtiendolo a video con OpenCV.

En caso de ejecutar en windows, se necesita instalar previamente GTK runtime, disponible en:
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases

## Ejemplos utiles para la presentacion

Comparar busquedas desinformadas:

```bash
python main.py --level level_easy.csv --algorithm bfs
python main.py --level level_easy.csv --algorithm dfs
```

Comparar heuristicas con A*:

```bash
python main.py --level level_mid.csv --algorithm astar --heuristic sum_nearest_target
python main.py --level level_mid.csv --algorithm astar --heuristic matching_min_distance
python main.py --level level_mid.csv --algorithm astar --heuristic matching_real_distance
```

Comparar Greedy contra A*:

```bash
python main.py --level level_mid.csv --algorithm greedy --heuristic matching_real_distance
python main.py --level level_mid.csv --algorithm astar --heuristic matching_real_distance
```

## Entregables

- Codigo fuente del motor de busqueda.
- Implementacion de estructura de estado.
- BFS, DFS, Greedy y A*.
- IDDFS como metodo opcional.
- Heuristicas admisibles.
- Resultados con costo, nodos expandidos, frontera, solucion y tiempo.
- README con instrucciones de ejecucion.
