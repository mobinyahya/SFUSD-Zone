# CLAUDE.md

SFUSD-Zone optimizes school district zoning for San Francisco Unified School District using OR-Tools CP-SAT (primary) and Gurobi MIP. It creates geographically contiguous zones balancing demographic diversity, geographic access, school capacity, program access, and school quality. Includes a web dashboard with LLM-powered exploration.

## Commands

All commands use `uv`. Python 3.13+ required.

```bash
uv sync                          # Install dependencies
uv run python -m Zone_Generation.Optimization.optimizer  # Single optimization run (uses Zone_Generation/Config/config.yaml)

# Benchmarking (from project root)
uv run python -m Zone_Generation.Running_Analysis.comparitive_benchmarks run -o /path/to/output
uv run python -m Zone_Generation.Running_Analysis.comparitive_benchmarks batch --mode recursive -o /path/to/output -w 5
uv run python -m Zone_Generation.Running_Analysis.comparitive_benchmarks aggregate -i /path/to/results
uv run python -m Zone_Generation.Running_Analysis.comparitive_benchmarks regenerate -i /path/to/results

# Website
cd website/backend && uv run uvicorn app:app --reload --port 8000

# LLM agent CLI
uv run python -m LLM.exploration.run_agent [path/to/summary.csv]
```

## Directory Structure

- `Zone_Generation/Config/` - config.yaml, centroids.yaml, Constants.py, metrics_config.py
- `Zone_Generation/Optimization/` - Optimizer implementations (cp_int, cp_bool, mip), design_zones.py, create_larger_areas.py
- `Zone_Generation/Running_Analysis/benchmark/` - config.py, runner.py, results.py, parallel.py
- `Zone_Generation/Running_Analysis/metrics/` - calculator.py, diversity.py, distance.py, programs.py, quality.py, choice.py
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

- `run` - Single benchmark. Key flags: `--output`, `--centroids`, `--frl-dev`, `--racial-dev`, `--seed`, `--recursive`
- `batch` - Parameter sweep. Key flags: `--mode recursive|single`, `--output`, `--workers`, `--skip-existing`, `--sequential`
- `aggregate` - Collect results into CSV. Key flags: `--input`, `--output`, `--zone-data-dir`
- `regenerate` - Recompute metrics for existing results. Key flags: `--input`, `--include-choice`

### Predefined Sweeps

Recursive sweep: 14 centroid types x 4 FRL devs (0.12-0.25) x 4 racial devs x 3 seeds x overages/shortages. Default 5 parallel workers.

### Output Structure

Each run produces a folder at `{centroids_type}/seed{seed}/frl{frl}_racial{racial}/...`:
```
result.json              # Status, metrics (40+), zone_data (per-zone demographics/programs/quality), config
zone_dict_BlockGroup_0.json  # {area_id: zone_id} mapping
solution_info.json       # Solver metadata
```

`aggregate` produces `summary.csv`: one row per solution, columns for all metrics + config values prefixed `config_`.

### Key Classes

- `BenchmarkConfig` (benchmark/config.py) - Single run configuration
- `ScenarioSweep` (benchmark/config.py) - Parameter sweep generator
- `ParallelRunner` (benchmark/parallel.py) - Process pool executor with worker recycling
- `ZoneMetricsCalculator` (metrics/calculator.py) - Computes diversity, distance, program, quality, structure metrics

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
