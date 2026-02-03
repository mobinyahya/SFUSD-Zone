# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SFUSD-Zone is a school district zoning optimization system for San Francisco Unified School District. It uses mathematical optimization (OR-Tools CP-SAT, Gurobi MIP) to create geographically contiguous school zones that balance:
- Demographic diversity (racial/ethnic balance, free/reduced lunch eligibility)
- Geographic access (minimizing travel distances)
- School capacity (matching students to available seats)
- Educational programs (language programs, special education access)
- School quality metrics

The system includes an LLM-powered agent for exploring zoning solutions via natural language and a web dashboard for interactive visualization.

## Development Commands

### Environment Setup
```bash
# Using UV package manager (Python 3.13+ required)
uv sync
source .venv/bin/activate
```

### Running Optimizations

**Single optimization run:**
```bash
cd Zone_Generation/Optimization
python optimizer.py  # Uses ../Config/config.yaml
```

**Benchmark with parameter sweep:**
```bash
# From project root
python -m Zone_Generation.Running_Analysis.comparitive_benchmarks run-single --config path/to/config.yaml
python -m Zone_Generation.Running_Analysis.comparitive_benchmarks run-batch --sweep recursive  # or 'single-level'
```

**Parallel benchmark execution:**
```bash
python -m Zone_Generation.Running_Analysis.comparitive_benchmarks run-parallel --sweep recursive --workers 8
```

### LLM Agent (Interactive CLI)
```bash
python -m LLM.exploration.run_agent [path/to/summary.csv]
# Default CSV: ~/sfusd-local-data/zones/SFUSD/local_runs/new_benchmarks_test/summary.csv
```

### Web Dashboard
```bash
cd website/backend
python app.py  # Starts FastAPI on http://localhost:8000
```

**Note:** The backend now uses pre-calculated metrics from `result.json` instead of recalculating from the graph. This provides 600x faster response times and complete access to all 40+ metrics including zone-level demographics, programs, and quality scores. See `website/BACKEND_REFACTORING_SUMMARY.md` for details.

### Testing
```bash
# No formal test framework; run individual test scripts:
python Zone_Generation/Running_Analysis/metrics/test_metrics.py
python Zone_Generation/Running_Analysis/benchmark/test_aggregation.py
```

## Architecture

### Core Data Flow

```
Student/School Data (CSV)
    ↓
design_zones.py → Loads and aggregates by geographic level
    ↓
Graph Construction → NetworkX graph (nodes=areas, edges=adjacency)
    ↓
Optimizer (CP-SAT/MIP) → Solves constraint program
    ↓
SolutionOutput → zone_dict (area → zone mapping)
    ↓
Metrics Calculator → Computes 40+ metrics
    ↓
Benchmark System → Aggregates results across parameter sweeps
    ↓
LLM Agent → Explores Pareto frontier
    ↓
Web Dashboard → Interactive visualization
```

### Key Directories

- **Zone_Generation/** - Core optimization engine
  - `Config/` - config.yaml (main configuration), centroids.yaml, Constants.py
  - `Optimization/` - Optimizer implementations (cp_int, cp_bool, mip), design_zones.py, recursive_zoning.py
  - `Running_Analysis/` - Benchmarking infrastructure (benchmark/, metrics/)

- **LLM/** - AI-powered exploration system
  - `exploration/` - zoning_agent.py (uses OpenAI SDK with Gemini), run_agent.py, pareto.py, filters.py, clusters.py

- **website/** - Web dashboard
  - `backend/` - FastAPI server (app.py loads pre-calculated metrics from result.json, data_loader.py)
  - `frontend/` - HTML/JS/CSS (Leaflet maps, Chart.js visualizations)

- **redistricting/** - GerryChain MCMC alternative approach
- **Helper_Functions/** - Shared utilities (util.py, Graph.py, ReCom.py)
- **Graphic_Visualization/** - Matplotlib/GeoPandas plotting utilities

### Optimization Pipeline

1. **Configuration** (Zone_Generation/Config/config.yaml):
   - Zone count via `centroids_type` (e.g., '5-zone-AF', '13-zone-6')
   - Constraint thresholds: `frl_dev`, `racial_dev`, `max_distance`
   - Geographic level: `level` ('Block_0', 'BlockGroup_1', 'attendance_area')
   - Recursive levels: `recursive_levels` for hierarchical optimization
   - Solver settings: `solve_time_limit`, `relative_gap_limit`, `optimizer` ('cp_int', 'cp_bool', 'mip')

2. **Optimization Strategies**:
   - **Single-level**: Optimizes at one geographic granularity
   - **Recursive/Hierarchical**: Solves coarse level first (BlockGroup_1), then refines at fine level (BlockGroup_0) using hints

3. **Key Classes**:
   - `Optimizer` (Zone_Generation/Optimization/optimizer.py): Base class with `get_optimizer()` factory
   - `DesignZones` (design_zones.py): Loads student/school data, builds graph, initializes zones
   - `ZoneMetricsCalculator` (Running_Analysis/metrics/calculator.py): Modular metrics computation
   - `ZoningAgent` (LLM/exploration/zoning_agent.py): LLM-based exploration with Pareto filtering

### Data Dependencies

System expects data at (controlled by `is_local` flag):
- **Remote/HPC**: `/share/data/school_choice/` and `~/sfusd-local-data/zones/SFUSD/`
- **Local**: `~/SFUSD/` and `~/Dropbox/SFUSD/`

Required files:
- Student enrollment CSV
- School locations/capacity CSV
- Census shapefiles (Block, BlockGroup)
- Adjacency/distance matrices
- Pre-computed NetworkX graphs (optional speedup)

## Important Patterns and Conventions

### Configuration Hierarchy
- Base config in `Zone_Generation/Config/config.yaml`
- Benchmark sweeps override specific parameters
- Individual runs can further override via CLI arguments

### Graph Object Structure

The optimization uses NetworkX graphs where nodes represent geographic areas and edges represent adjacency. Understanding the graph structure is critical for working with the optimization code.

#### Graph Creation and Levels

Graphs are created by `create_larger_areas.py` at multiple hierarchical levels:
- **BlockGroup_0** - Finest granularity (base level, ~500+ nodes)
- **BlockGroup_1** - Aggregated level 1 (~100-200 nodes)
- **BlockGroup_2** - Aggregated level 2 (~30-60 nodes)

The `create_graph()` function (create_larger_areas.py:17) builds graphs from `DesignZones` objects and census shapefiles.

#### Node Attributes

Each node (representing a geographic area) has the following attributes:

```python
node_attrs = {
    'area_id': int,              # Original census area ID (BlockGroup GEOID)
    'ge_students': float,        # General education students in this area
    'ge_capacity': float,        # General education capacity (school seats)
    'all_prog_students': float,  # All program students (including special ed)
    'all_prog_capacity': float,  # All program capacity
    'num_schools': int,          # Number of schools in this area
    'FRL': float,                # Free/Reduced Lunch eligible students (count)
    'school_ids': list,          # List of school IDs in this area
    'lat': float,                # Latitude of area centroid
    'lon': float,                # Longitude of area centroid
    # Ethnicity counts (see AREA_ETHNICITIES in Constants.py):
    'Ethnicity_Black_or_African_American': float,
    'Ethnicity_Hispanic/Latinx': float,
    'Ethnicity_White': float,
    'Ethnicity_Asian': float,
    'Ethnicity_Pacific_Islander': float,
    # Additional ethnicities...
}
```

For aggregated graphs (BlockGroup_1, BlockGroup_2), nodes also have:
- `'block_ids': list` - Original BlockGroup IDs that were aggregated into this node

#### Graph-Level Attributes

The graph object itself (accessed via `G.graph`) contains global data:

```python
G.graph = {
    'distance_dict': dict,  # {area_idx: {area_idx: distance}} - pairwise distances
    'school_data': dict,    # {school_id: school_info_dict} - all school metadata
    'F': float,             # District-wide FRL proportion (0-1)
    'R': dict,              # District-wide ethnicity proportions {ethnicity: proportion}
    'partition': dict,      # (Aggregated graphs only) {original_node: new_node} mapping
}
```

The `distance_dict` is indexed by node index (0, 1, 2, ...), not by area_id. Use `area2idx` mapping from DesignZones to convert.

#### Edge Structure

Edges represent geographic adjacency:
- Simple undirected edges with no weights (by default)
- Created based on shapefile geometry touching (shapefile.geometry.touches())
- Used to enforce contiguity constraints (zones must be connected components)

#### Hierarchical Aggregation Process

The `create_larger_areas.py` workflow creates multi-level graphs:

1. **create_base_graph()** (line 287):
   - Loads DesignZones for BlockGroup level
   - Creates BlockGroup_0.pickle with ~500+ nodes
   - Each node = one Census BlockGroup

2. **recursively_split_and_save()** (line 309):
   - Uses METIS graph partitioning to create coarse zones
   - Saves intermediate zone assignments (zone_dict)
   - zone_dict format: `{node_idx: zone_id}`

3. **aggregate_zone_dict()** (line 193):
   - Takes partition (zone_dict) and base graph
   - Creates new aggregated graph where each node = one partition
   - Aggregates all node attributes (sums students, capacity, FRL, ethnicities)
   - Recomputes adjacency based on dissolved geometries
   - Recalculates distance_dict using new centroids
   - Stores original→new mapping in `G.graph['partition']`

4. **create_intermediate_graphs()** (line 334):
   - Applies aggregation to create BlockGroup_1.pickle, BlockGroup_2.pickle
   - Each level has fewer nodes but same total students/capacity

#### Key Functions in create_larger_areas.py

- **create_graph(dz, config)** - Convert DesignZones to NetworkX graph
- **partition_to_subgraphs(G, partition)** - Split graph into subgraphs by partition
- **recursively_split_with_zones(G, cur_size, depth, zone_offset)** - Hierarchical METIS partitioning
- **aggregate_zone_dict(partition, G)** - Aggregate fine-grained graph to coarse graph
- **convert_to_block_zone_dict(zone_dict, G)** - Convert node indices to original area IDs

#### Usage in Optimization

The optimizer receives a graph (loaded from pickle or created fresh) and uses:
- **Node attributes** for constraint calculations (capacity, demographics)
- **Edges** for contiguity constraints (zones must be connected)
- **distance_dict** for distance-based objectives and constraints
- **F and R** for district-wide diversity targets

Example usage in recursive optimization:
```python
# Load coarse graph
with open('BlockGroup_1.pickle', 'rb') as f:
    coarse_graph = pickle.load(f)

# Optimize at coarse level
coarse_solution = optimizer.optimize(coarse_graph)

# Use coarse solution to guide fine-level optimization
fine_graph = pickle.load('BlockGroup_0.pickle')
fine_solution = optimizer.optimize(fine_graph, hints=coarse_solution)
```

#### Converting Solutions Back to Original Geography

After optimization on aggregated graphs, use the partition mapping:
```python
# Solution is {node_idx: zone_id} on aggregated graph
# G.graph['partition'] is {original_node: aggregated_node}
original_solution = {}
for orig_node, agg_node in G.graph['partition'].items():
    original_solution[orig_node] = solution[agg_node]
```

### Optimizer Selection
```python
from Zone_Generation.Optimization.optimizer import Optimizer
optimizer = Optimizer.get_optimizer(config)  # Returns appropriate solver
solution = optimizer.optimize()
```

### Metrics System
Modular metrics in `Zone_Generation/Running_Analysis/metrics/`:
- `diversity.py` - Demographic balance (FRL, race/ethnicity)
- `distance.py` - Travel distance metrics
- `programs.py` - Educational program access
- `quality.py` - School quality metrics
- `choice.py` - Iterative choice simulation

All metrics accessed via `ZoneMetricsCalculator` which computes full metric suite.

### Recursive Optimization
Specified via `recursive_levels` and `solve_time_limits` in config:
```yaml
recursive_levels: ['BlockGroup_1', 'BlockGroup_0']
solve_time_limits: [30, 30]  # seconds per level
relative_gap_limits: [0, 0]   # optimality gaps per level
```

### LLM Agent Tool Calling
The `ZoningAgent` uses function calling to:
- Apply filters on metrics (tighten/loosen thresholds)
- Compute Pareto frontier
- Find centroid solutions
- Cluster solutions by characteristics
- Load specific solutions by path

### Solution Storage
Solutions saved with structure:
```
output_folder/
  └── zone_dicts/
      └── [centroid_type]_[timestamp]/
          ├── zone_dict.pkl
          ├── metadata.json
          └── visualization.png
```

## Common Workflows

### Workflow 1: Generate and Visualize a Single Zone Configuration
1. Edit `Zone_Generation/Config/config.yaml` (set `centroids_type`, constraints)
2. Run `python Zone_Generation/Optimization/optimizer.py`
3. Solution saved to configured output folder
4. Visualize with `ZoneVisualizer` from `Graphic_Visualization/zone_viz.py`

### Workflow 2: Run Benchmark Parameter Sweep
1. Configure sweep in `comparitive_benchmarks.py` or pass `--sweep` argument
2. Run `python -m Zone_Generation.Running_Analysis.comparitive_benchmarks run-batch --sweep recursive`
3. Results aggregated to CSV with all metrics
4. Analyze Pareto frontier with LLM agent

### Workflow 3: Explore Solutions with LLM Agent
1. Generate benchmark CSV with many solutions
2. Start agent: `python -m LLM.exploration.run_agent summary.csv`
3. Chat to express preferences (e.g., "I want lower distances but diversity is important")
4. Agent dynamically filters Pareto frontier
5. Agent returns centroid solution from filtered set
6. Can request specific solution paths or cluster explorations

### Workflow 4: Interactive Web Dashboard
1. Generate benchmark results with many solutions
2. Start backend: `cd website/backend && python app.py`
3. Open browser to http://localhost:8000
4. Chat with agent to explore solutions
5. Map auto-updates with selected solution
6. View demographics charts and zone statistics

### Workflow 5: Debug Optimization Issues
1. Check solver status in output logs
2. Reduce `solve_time_limit` for faster iteration
3. Relax constraints (`frl_dev`, `racial_dev`, `max_distance`)
4. Try different optimizer: `optimizer: 'cp_bool'` instead of `'cp_int'`
5. Enable hints: `use_hints: True` for recursive optimization
6. Check feasibility with simpler centroid configuration

## Code Locations for Common Tasks

### Modify optimization constraints
- Add constraint type: `Zone_Generation/Optimization/optimizer.py` (base class methods)
- Implement in solver: `constraint_program_integer.py` or `constraint_program_boolean.py`

### Add new metric
1. Create metric class in `Zone_Generation/Running_Analysis/metrics/[category].py`
2. Register in `calculator.py` (ZoneMetricsCalculator)
3. Add to `LLM/exploration/metrics_config.py` for LLM agent awareness
4. Update benchmark CSV columns

### Modify LLM agent behavior
- Core logic: `LLM/exploration/zoning_agent.py`
- System prompt and tools: Inside `ZoningAgent.__init__()` and `_get_tools()`
- Filter state management: `filters.py`
- Pareto computation: `pareto.py`

### Customize web dashboard
- Backend API: `website/backend/app.py`
- Frontend UI: `website/frontend/index.html`, `app.js`, `style.css`
- Data loading: `website/backend/data_loader.py`

### Modify graph construction
- Graph building: `Helper_Functions/Graph.py`
- Adjacency logic: `Zone_Generation/Optimization/graph_utils.py`
- Distance calculations: `Helper_Functions/util.py`

## Environment Variables

Create `.env` file in project root:
```bash
OPENAI_API_KEY=your_gemini_api_key  # Used for LLM agent (configured for Gemini)
```

## Important Notes

- **Python version**: Requires Python 3.13+
- **Solvers**: OR-Tools CP-SAT is primary solver (free); Gurobi MIP requires license
- **Data paths**: Set `is_local: True` in config for local development paths
- **Solver performance**: CP-SAT generally faster than MIP for this problem
- **Recursive optimization**: Almost always better than single-level for quality
- **LLM API**: Currently configured to use Google Gemini via OpenAI SDK compatibility
- **Time estimates**: Recursive runs with 4-minute budgets typical for production quality
- **Solution feasibility**: If solver fails, relax diversity constraints or increase distance limits
