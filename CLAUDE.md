# CLAUDE.md

SFUSD-Zone optimizes school district zoning for San Francisco Unified School District using OR-Tools CP-SAT (primary) and Gurobi MIP. It creates geographically contiguous zones balancing demographic diversity, geographic access, school capacity, program access, and school quality. Includes a web dashboard with LLM-powered exploration.

## Commands

All commands use `uv`. Python 3.13+ required.

```bash
uv sync                          # Install dependencies
uv run python -m Zone_Generation.Optimization.optimizer  # Single optimization run (uses Zone_Generation/Config/config.yaml)

# Benchmarking (from project root)
uv run python -m Zone_Generation.benchmark.run path/to/sweep.yaml
uv run python -m Zone_Generation.benchmark.run path/to/sweep.yaml --mode metrics

# Website
cd website/backend && uv run uvicorn app:app --reload --port 8000

# LLM agent CLI
uv run python -m LLM.exploration.run_agent [path/to/summary.csv]
```

## Directory Structure

- `Zone_Generation/Config/` - config.yaml, centroids.yaml, Constants.py, metrics_config.py
- `Zone_Generation/Optimization/` - Optimizer implementations (cp_int, cp_bool, mip), design_zones.py, create_larger_areas.py
- `Zone_Generation/benchmark/` - config.py, runner.py, results.py, parallel.py
- `Zone_Generation/metrics/` - calculator.py, diversity.py, distance.py, programs.py, quality.py, choice.py
- `LLM/exploration/` - zoning_agent.py, prompts.py, filters.py, pareto.py, clusters.py, tool_defs.py
- `website/backend/` - app.py (FastAPI), data_loader.py
- `website/frontend/` - index.html, app.js, shared.js, style.css, admin.html/js/css
- `Helper_Functions/` - util.py, Graph.py

## Graph Object Structure

The optimization uses NetworkX undirected graphs. Nodes are geographic areas, edges are adjacency. Graphs are created by `create_larger_areas.py` at hierarchical levels:
- **BlockGroup_0** - Finest (~500+ nodes, one per Census BlockGroup)
- **BlockGroup_1** - Aggregated (~100-200 nodes)
- **BlockGroup_2** - Coarsest (~30-60 nodes)

### Node Attributes

```python
{
    'area_id': int,              # Census BlockGroup GEOID
    'ge_students': float,        # General education students (count)
    'ge_capacity': float,        # GE school seats
    'all_prog_students': float,  # All program students
    'all_prog_capacity': float,  # All program capacity
    'num_schools': int,
    'school_ids': list,
    'FRL': float,                # Free/Reduced Lunch students (count, not proportion)
    'lat': float,
    'lon': float,
    # Ethnicity counts (see AREA_ETHNICITIES in Constants.py):
    'Ethnicity_Black_or_African_American': float,
    'Ethnicity_Hispanic/Latinx': float,
    'Ethnicity_White': float,
    'Ethnicity_Asian': float,
    'Ethnicity_Pacific_Islander': float,
    'block_ids': list,           # Only on aggregated graphs: original BlockGroup IDs
}
```

### Graph-Level Attributes (G.graph)

```python
{
    'distance_dict': dict,  # {node_idx: {node_idx: distance}} - indexed by node index, NOT area_id
    'school_data': dict,    # {school_id: school_info_dict}
    'F': float,             # District-wide FRL proportion (0-1)
    'R': dict,              # District-wide ethnicity proportions {ethnicity: proportion}
    'partition': dict,      # Aggregated graphs only: {original_node: aggregated_node}
}
```

### Edges

Unweighted undirected edges from shapefile geometry adjacency (touches). Used to enforce zone contiguity.

### Hierarchical Aggregation

`create_larger_areas.py` builds multi-level graphs:
1. `create_base_graph()` - Creates BlockGroup_0.pickle from DesignZones + census shapefiles
2. `recursively_split_with_zones()` - METIS partitioning into coarse zones, produces zone_dict `{node_idx: zone_id}`
3. `aggregate_zone_dict(partition, G)` - Creates coarser graph: sums node attributes, recomputes adjacency from dissolved geometries, recalculates distance_dict, stores mapping in `G.graph['partition']`
4. Result: BlockGroup_1.pickle, BlockGroup_2.pickle (fewer nodes, same total students/capacity)

To convert aggregated solutions back to fine-grained geography:
```python
for orig_node, agg_node in G.graph['partition'].items():
    original_solution[orig_node] = solution[agg_node]
```

## Benchmarking

### CLI Commands

Benchmarking is configured from one simulation sweep YAML file. The same entrypoint can run the full sweep or recalculate metrics from saved stage results. Aggregation always runs after either mode.

- `mode: run` - Generate tasks from YAML and run the full optimization sweep.
- `mode: metrics` - Reconstruct saved `ZoneSolution` stages, rewrite `result.json` with updated metrics, and aggregate outputs.

See `Zone_Generation/benchmark/sweep.example.yaml` for the YAML shape.

### Output Structure

Each run produces a folder at `{centroids_type}/seed{seed}/frl{frl}_racial{racial}/...`:
```
benchmark_manifest.json  # Task id, config hash, stage paths, status, timing
result.json              # Status, metrics, zone_data, config, run metadata
zone_dict_<level>.json   # Final root-level assignment alias
zone_dict_area_<level>.json
solution_<level>.json
stages/<stage>/<files>   # Every recursive/iterative level result
```

Aggregation produces `summary.csv` with one row per run and `stages.csv` with one row per saved stage.

### Key Classes

- `SimulationSweep` (benchmark/config.py) - YAML-backed sweep definition
- `BenchmarkTask` (benchmark/config.py) - Concrete optimization task
- `run_sweep` (benchmark/parallel.py) - Capacity-aware process executor with worker recycling
- `MetricsCalculator` (metrics/calculator.py) - Optimization-native metrics over `ZoneSolution` stages

## Website

### Backend (website/backend/)

FastAPI server serving the frontend and API. Uses pre-calculated metrics from `result.json` (no graph recalculation).

Key endpoints:
- `GET /api/solution/{path}` - Load solution: zone mapping, demographics, metrics, percentile ranks, colors
- `GET /api/geojson` - SF blockgroup geometries for map
- `GET /api/schools` - School locations
- `GET /api/metrics-config` - Metrics metadata (names, directions, categories)
- `POST /api/chat` - Chat with ZoningAgent (session-based, Gemini via OpenAI SDK)
- `POST /api/admin/filter` - Direct filter/centroid endpoint
- `GET /api/health` - Health check

Data path: `DEFAULT_CSV_PATH` in app.py points to summary.csv. Solutions loaded by path from `result.json` + `zone_dict_BlockGroup_0.json`.

### Frontend (website/frontend/)

- Leaflet map with CartoDB basemap, GeoJSON zones colored by assignment
- Chart.js zone-level bar charts (demographics, programs, quality)
- Comparison table with percentile rankings per metric
- Chat interface with markdown rendering and cluster selector
- Solution history (auto-saves up to 30, with pros/cons editing)
- Admin console (admin.html) for direct filter manipulation

### Agent Integration

`ZoningAgent` (LLM/exploration/zoning_agent.py) manages:
- Filter state with versioning and undo
- Pareto frontier computation and filtering
- Cluster exploration (themed solution groups)
- Zone-level data queries
- Uses Gemini via OpenAI-compatible API (`OPENAI_API_KEY` in `.env`)

## Config Reference (Zone_Generation/Config/config.yaml)

| Parameter | Example | Description |
|---|---|---|
| `centroids_type` | `'5-zone-AF'` | Zone count and centroid configuration |
| `frl_dev` | `0.3` | Max FRL deviation from district average |
| `racial_dev` | `0.3` | Max racial/ethnic deviation |
| `optimizer` | `'cp_int'` | Solver: cp_int, cp_bool, or mip |
| `recursive_levels` | `['BlockGroup_1','BlockGroup_0']` | Hierarchical solve order |
| `solve_time_limits` | `[30, 30]` | Seconds per recursive level |
| `overage` / `shortage` | `0.8` / `0.2` | Capacity tolerance (proportion) |
| `is_local` | `False` | Data path toggle (local vs HPC) |
| `use_hints` | `True` | Use coarse solution as hint for fine level |
| `random_seed` | `42` | Solver seed |

Data paths: HPC at `/share/data/school_choice/`, local at `~/SFUSD/`.
