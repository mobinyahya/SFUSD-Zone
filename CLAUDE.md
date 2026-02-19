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

---

## Administrator AI Interaction System (Redesign 2026)

### Overview
The administrator interaction system has been redesigned as a **state-based, versioned system** with:
- Concise, natural responses (80-100 words max for normal responses)
- Flexible word limit: Can exceed when user explicitly requests comprehensive details
- Zone-level data access via tool calls
- Solution versioning with undo capability
- Explicit confirmation/exit handling
- Clear text-based metric direction indicators (not arrows)

### State Schema

```python
@dataclass
class ProposalVersion:
    """Versioned snapshot of zoning proposal state."""
    version_id: int
    timestamp: str
    filter_state: FilterState
    solution_path: Optional[str]
    solution_count: int
    description: str

@dataclass
class AgentState:
    """Complete session state."""
    versions: list[ProposalVersion]
    current_version: int
    cluster_labels: Optional[list]
    cluster_centers: Optional[list]
    cluster_directions: Optional[dict]
    clustered_solutions: Optional[object]
    clustered_vectors: Optional[object]
    awaiting_confirmation: bool
    pending_action: Optional[dict]
    last_action: str
```

### New Tools

#### Zone-Level Queries
- **`query_zone_data(zone_ids, metrics)`** - Get demographics, programs, quality, distance for specific zones
- **`compare_zones(zone_ids)`** - Side-by-side comparison of 2+ zones

#### Versioning
- **`undo_action(steps)`** - Revert to previous filter state
- **`show_version_history()`** - Display version timeline

### State Diagram

```
┌─────────────┐
│   Initial   │ v0: No filters, all Pareto solutions
└──────┬──────┘
       │
       │ User: "Tighten diversity"
       ▼
┌─────────────┐
│  Filtered   │ v1: Diversity tightened → 89 solutions
└──────┬──────┘
       │
       │ User: "Show zones 1 and 3"
       ▼
┌─────────────┐
│ Zone Query  │ (No version change - read-only)
└──────┬──────┘
       │
       │ User: "Loosen distance"
       ▼
┌─────────────┐
│  Adjusted   │ v2: Distance loosened → 134 solutions
└──────┬──────┘
       │
       │ User: "Undo"
       ▼
┌─────────────┐
│  Restored   │ v1: Back to 89 solutions (diversity tight)
└─────────────┘
```

### Example Interaction Flow

**Before (Verbose):**
```
User: What does the current solution look like?

Agent: The current solution shows a Free/Reduced Lunch deviation 
of 0.045 which represents how much the zones vary from the district 
average. The racial diversity deviation is 0.082. The average distance 
students travel is 1.23 miles. This solution includes 5 zones with 
varying demographic compositions...

[200+ words]
```

**After (Concise & Natural, 80-100 words):**
```
User: What does the current solution look like?

Agent: v0: Initial balanced solution with 156 options available.

Key metrics for the current centroid:
• FRL deviation: 0.045 (lower is better) - strong equity
• Racial diversity: 0.082 (lower is better) - good balance
• Average distance: 1.23 miles (lower is better)
• Total programs: 41 (higher is better)

This represents a balanced approach between diversity and geographic access. Would you like to prioritize a specific metric or explore different trade-offs?

---

User: Show me zone 1 and zone 3 demographics

Agent: **Zone 1:**
• Students: 2,340, FRL: 45.2%
• Ethnicity: Hispanic 52.1%, Asian 23.4%
• Programs: 8, Language immersion: 2

**Zone 3:**
• Students: 1,890, FRL: 28.7%
• Ethnicity: White 48.2%, Asian 31.1%
• Programs: 12, Language immersion: 4

---

User: Make diversity tighter

Agent: To tighten diversity (FRL deviation), I'll filter for zones with more similar economic composition.

Expected outcomes:
• Gains: More equitable distribution of students eligible for free/reduced lunch
• Trade-offs: Typically reduces solutions by 40-50%, may slightly increase average distance

How much would you like to tighten this - mildly, moderately, or aggressively?

---

User: moderately

Agent: ✓ v1: Tightened FRL deviation (moderately)
• Solutions: 156 → 67
• FRL deviation improved: 0.098 → 0.045

This focuses on more economically balanced zones. Would you like to continue exploring these solutions or adjust other metrics?

---

User: Compare zones 1 and 3 now

Agent: **Comparing Zones 1, 3:**
FRL %: 45.2% vs 28.7%
Students: 2340 vs 1890
Programs: 8 vs 12
Rating: 6.8 vs 7.4
Avg Dist: 1.1mi vs 0.9mi

---

User: Actually, undo that

Agent: ✓ Reverted to v0 (initial balanced solution)

You now have all 156 solutions available again with the original filter settings. We're back to the baseline with low FRL deviation (0.045). Would you like to try adjusting a different metric this time?

---

User: Show version history

Agent: **Version History:**
→ v0: Initial state (156 solutions)
  v1: Tightened FRL deviation (67 solutions)
```

### Backend Implementation

#### Key Changes in `zoning_agent.py`:

1. **State Management** (lines 50-110)
   - `AgentState` class tracks versions
   - `save_version()` creates snapshots
   - `undo()` restores previous states

2. **New Tools** (lines 120-170)
   - `query_zone_data` - loads zone_data from result.json
   - `compare_zones` - side-by-side zone comparison
   - `undo_action` - restore previous version
   - `show_version_history` - display timeline

3. **Auto-Versioning** (filter modifications)
   - `tighten_filter` → saves v{N}
   - `loosen_filter` → saves v{N}
   - `select_cluster` → saves v{N}

4. **Concise System Prompt** (lines 357-410)
   - Max 50 words per response
   - Bullet-point format
   - Action-first structure
   - Version tracking emphasis

### Code Recommendations

#### For Frontend Integration:
```javascript
// Track version in UI
let currentVersion = 0;

// Show undo button when version > 0
if (response.version_id > 0) {
  showUndoButton();
}

// Display version timeline
function renderVersionHistory(versions) {
  return versions.map(v => 
    `<li class="${v.is_current ? 'active' : ''}">
      v${v.version_id}: ${v.description} (${v.solution_count} solutions)
    </li>`
  );
}
```

#### For Confirmation Flow:
```python
# In zoning_agent.py
if tool_name == "tighten_filter":
    # ... existing logic ...
    
    # Check if major change (>50% reduction)
    if len(actual_filtered) < len(filtered) * 0.5:
        self.state.awaiting_confirmation = True
        self.state.pending_action = {
            "type": "tighten",
            "metric": metric_name,
            "before": len(filtered),
            "after": len(actual_filtered)
        }
        return f"⚠️ Major change: {len(filtered)}→{len(actual_filtered)} solutions. Confirm?"
```

#### For Session Persistence:
```python
# Save state to database/file
def save_session(agent: ZoningAgent, session_id: str):
    state_dict = {
        "versions": [asdict(v) for v in agent.state.versions],
        "current_version": agent.state.current_version,
        "filter_state": asdict(agent.filter_state),
    }
    with open(f"sessions/{session_id}.json", "w") as f:
        json.dump(state_dict, f)

# Restore from database/file
def load_session(agent: ZoningAgent, session_id: str):
    with open(f"sessions/{session_id}.json", "r") as f:
        state_dict = json.load(f)
    
    agent.state.versions = [ProposalVersion(**v) for v in state_dict["versions"]]
    agent.state.current_version = state_dict["current_version"]
    # Restore filter_state...
```

### Response Format Standards

**Action Confirmation:**
```
✓ v{N}: {action_description}
• {before}→{after} solutions
• Trade-off: {key_impact}
{prompt_for_next}
```

**Current Solution Summary (80-100 words, natural tone):**
```
v{N}: {count} solutions

Current solution has strong diversity metrics with low FRL deviation (0.045) 
and low racial diversity deviation (0.039). Average distance is 1.23 miles 
with 41 programs available. This balances equity with access. Would you like 
to prioritize a specific metric?
```

**Zone Data:**
```
**Zone {id}:**
• {key_metric_1}
• {key_metric_2}
• {key_metric_3}
```

**Error/Infeasibility:**
```
⚠️ {problem_description}
• Suggest: {action_to_fix}
```

### Critical Rules for Metric Display

**ALWAYS use clear text for metric directions, NOT arrows:**
- ❌ AVOID: `FRL: 0.045↓, programs: 41↑` (arrows are confusing)
- ✅ USE: `FRL deviation: 0.045 (lower is better), 41 programs (higher is better)`
- ✅ BETTER: `low FRL deviation (0.045)`, `41 programs available`
- For changes: Use "increased", "decreased", "higher", "lower", "more", "fewer"

**Why no arrows?** Arrows combined with qualitative words ("worse Hispanic/Latinx Representation↑") create confusion. Clear text is always better.

**NEVER:**
- List every metric individually (max 3-4 key ones)
- Exceed 100 words per response
- Use arrows (↑↓) for metric directions
- Mix arrows with "worse" or "better" - extremely confusing
- Use excessive qualitative ratings - occasional "strong" or "good" is fine, but don't rate everything as "excellent"
- Show program breakdowns (SA, AF, CB, etc.) without context
- Include ANY code-like references: function calls, syntax, programming terminology
- Example: ❌ "select_cluster(2)" ✅ "Please tell me which cluster number (1, 2, or 3)"

**WHY:** The verbose response example shows the problem - listing 20+ metrics with qualitative ratings ("very good", "excellent") but inconsistent direction info. Some metrics have "lower is better", others don't. The specific program counts (SA: 1.50, AF: 2.50, etc.) lack direction because they're component details, not standalone metrics.

**FIX:** 
- Keep responses to 80-100 words with natural phrasing
- Tools provide data, agent extracts 2-4 KEY metrics with clear text
- Use occasional qualitative language ("strong diversity", "good balance") but always pair with actual values
- Users can ask for details if needed

### No Code References - Critical for Administrator UX

**PROBLEM:** Agent was including programming references in responses:
- "You can select a cluster by its number (e.g., select_cluster(2))."
- "Use query_zone_data() to see more details"

**WHY WRONG:** Administrators are policy experts, not developers. Code syntax is:
- Confusing and intimidating
- Breaks the natural conversation flow
- Makes the tool feel technical rather than accessible

**SOLUTION:**
❌ BAD: "select_cluster(2)" or "query_zone_data()"
✅ GOOD: "Which cluster would you like to explore? Just tell me the cluster number (1, 2, or 3)."

❌ BAD: "Call the compare_zones function"
✅ GOOD: "Would you like to compare specific zones side-by-side?"

**Rule:** Speak in plain administrative language. Never mention function names, syntax, or programming concepts.

### Consultative Adjustment Workflow - Critical for UX

**PROBLEM:** Agent was immediately applying adjustments without explanation or user control:
```
User: "Could we prioritize math scores more"
Agent: "✓ v3: Prioritized higher Math Scores. • 289→260 solutions"
```

**WHY WRONG:** 
- No explanation of what will be gained
- No discussion of trade-offs
- No user control over degree of change
- Feels abrupt and non-consultative

**SOLUTION - 3-Step Flow:**

**Step 1: Explain Trade-offs**
```
User: "Could we prioritize math scores more"

Agent: "To prioritize math scores, I'll filter for solutions with higher average math proficiency.

Expected outcomes:
• Gains: Better math performance across zones
• Trade-offs: Typically reduces solutions by 30-40%, may increase average distance slightly

How aggressively would you like to prioritize this? (mild / moderate / aggressive)"
```

**Step 2: User Chooses Strength**
```
User: "moderate"
```

**Step 3: Apply & Show Results**
```
Agent: "✓ v3: Tightened Math Scores (moderately)
• Solutions: 289 → 183
• Math scores improved: 2,409 → 2,543

Would you like to adjust another metric or explore these solutions?"
```

**Key Principles:**
- ALWAYS explain gains and trade-offs BEFORE applying changes
- ALWAYS ask for strength level (mild/moderate/aggressive)
- ALWAYS show before/after metrics in results
- Maintain conversational flow with clear next steps

### Word Limit Flexibility

**Standard Response:** 80-100 words (concise, focused on key points)

**Flexible Expansion:** Agent should intelligently recognize when more detail is needed based on user intent, not trigger phrases.

**Use judgment - provide more detail when:**
- User asks "what about...", "tell me more...", "explain..."
- User seeks understanding: "why", "how", "what's the difference"
- User asks about trade-offs or implications
- User compares multiple things
- Context suggests comprehensive answer is helpful

**Examples:**

**Concise (80 words):**
```
User: "What's the current solution?"

Agent: "v0: Balanced solution with 156 options.

Key metrics:
• FRL deviation: 0.045 (lower better) - strong equity
• Racial diversity: 0.082 (lower better)
• Average distance: 1.23 miles (lower better)
• Programs: 41 (higher better)

Would you like to prioritize a specific metric?"
```

**Expanded (150 words) - User seeking detail:**
```
User: "What metrics are we looking at here?"

Agent: "We're evaluating solutions across several categories:

**Demographics:**
• Racial diversity deviation: 0.039 - how evenly racial/ethnic groups are distributed
• FRL representation: 0.110 - economic balance across zones
• Individual group deviations for Black, Hispanic/Latinx, White, Asian populations

**Geographic Access:**
• Average distance to schools: 0.316 miles
• Number of schools per zone: 17
• Boundary compactness: 101 (lower = more compact)

**Programs:**
• Total programs: 41 per zone
• Language immersion: 7
• Special education: 8

**School Quality:**
• Math scores: 2,409
• English scores: 2,397

Each metric has trade-offs - improving one often affects others. What aspect interests you most?"
```

**Expanded (140 words) - User asking comparison:**
```
User: "How do zones 1 and 3 differ?"

Agent: "Key differences between Zone 1 and Zone 3:

**Demographics:**
• Zone 1: 45% FRL, majority Hispanic (52%) and Asian (23%)
• Zone 3: 29% FRL, majority White (48%) and Asian (31%)
Zone 1 serves more economically disadvantaged students.

**Program Access:**
• Zone 1: 8 programs total, 2 language immersion
• Zone 3: 12 programs total, 4 language immersion
Zone 3 has more diverse program options.

**School Quality:**
• Zone 1: Math 6.2, Rating 6.8
• Zone 3: Math 7.1, Rating 7.4
Zone 3 has slightly higher performance metrics.

**Distance:**
• Zone 1: 1.1 miles average
• Zone 3: 0.9 miles average

Would you like to explore solutions that balance these differences?"
```

The key: Be concise by default, expand naturally when user intent calls for it.

### File Paths - NEVER Show to Users

**Problem:** Agent was showing internal file paths like:
```
/home/kumarc/sfusd-local-data/zones/SFUSD/local_runs/new_benchmarks_test/zone-rec-4/seed42/...
```

**Why Wrong:**
- These are implementation details
- Not helpful for administrators
- Exposes system structure
- Clutters the response

**Solution:** 
- Paths used internally to load solution data
- NEVER mentioned in responses to users
- System prompt explicitly forbids showing paths
- Solutions identified by version # and metrics, not file paths

### Testing Checklist

- [ ] Version history maintains correct order
- [ ] Undo restores exact filter state
- [ ] Zone queries return valid data
- [ ] Response length ≤50 words (measured)
- [ ] All tool calls save versions appropriately
- [ ] State persists across conversation turns
- [ ] Cluster selection updates versions
- [ ] Confirmation flow blocks major changes
