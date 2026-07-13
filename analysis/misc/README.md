# Manual Block Edge Review

Generate every Block_0 node/school case that lacks an adjacent node strictly
closer to the school centroid. By default, each plot contains only the focal
block and blocks connected to it by existing graph edges:

```bash
uv run python analysis/misc/manual_block_edge_cases.py generate
```

Plots are written to `analysis/plots/manual_cases/`. Each plot has local numeric
labels backed by stable Census Block GEOIDs in
`analysis/misc/manual_case_manifest.json`.

To include non-neighbor blocks as blue manual-edge candidates, enable the
radius-based review explicitly:

```bash
uv run python analysis/misc/manual_block_edge_cases.py generate \
  --include-nearby-non-neighbors --base-radius-miles 0.25 --overwrite
```

The radius expands per case when needed to display at least one strictly closer
candidate. Every plot uses one local panel, fits the complete geometry of every
existing neighbor, and draws an arrow toward the school centroid.

Record reviewed missing neighbors in
`analysis/misc/manual_case_selections.yaml`:

```yaml
1: [4, 7]
2: [3]
```

The key is the case number and each list value is a local node label from that
case's plot. Multiple labels are allowed. Compile the selections after editing:

```bash
uv run python analysis/misc/manual_block_edge_cases.py compile
```

Compilation validates that every selected node is displayed, is not already a
neighbor, and is strictly closer to the case's school centroid. It writes stable
Block GEOID pairs to `Config/manual_block_edges.yaml`. The graph cache namespace
includes those edges, so the next Block graph request rebuilds Block_0 and every
higher Block level from the reviewed topology.
