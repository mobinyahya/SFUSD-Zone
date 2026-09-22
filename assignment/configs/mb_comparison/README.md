# Mission Bay comparison study

Five assignment run configs split into two families, because Mission Bay ES is
a different kind of object in each school year and one series cannot hold both.

Mission Bay ES opened **17 August 2026**, the first day of SY26-27 and SFUSD's
first new school in two decades. That single fact drives the whole split:

| Year | Mission Bay in the data | What it is |
|---|---|---|
| 2023-24 | `999-GE-KG`, 66 seats | **Counterfactual.** Hard-coded in `programs_statusQuo_2324.csv`. No 2023-24 student ranks it, enrols at it, or is zoned to it |
| 2024-25 | absent | **Nothing.** No 909/999/1731 row in the capacity file, demographics or post-run |
| 2025-26 | absent | **Nothing.** Same |
| 2026-27 | `999-GE-KG` 69, `999-MM-KG` 5, `999-AF-KG` 0 | **Record.** District Main Round capacities; 233 kindergarteners ranked it, 74 enrolled |

The district code is `idSchool 1731`, remapped to the repository's placeholder
`999` by `RAW_SCHOOL_ID_ALIASES` in
`analysis/data_prep/sfusd_transfer_schema.py`. `909` and `999` are both
placeholder IDs and appear as two identical school rows.

## Family A — Mission Bay excluded, 2023-24 → 2025-26

`nomb_2324.yaml`, `nomb_2425.yaml`, `nomb_2526.yaml`

Every cell runs `capacity_profile: default` with `include_mission_bay: false`.
Each year contributes its own real program table and **all three share one
school table**, so the only thing that moves across the series is the year.

| cell | programs source | programs | students | capacity |
|---|---|---:|---:|---:|
| `nomb_2324` | `programs_2324.csv` | 164 | 3,953 | 4,268 |
| `nomb_2425` | `programs_2425.csv` | 166 | 3,852 | 4,099 |
| `nomb_2526` | `programs_2526.csv` | 166 | 3,717 | 4,202 |

School table for all three: `schools_rehauled_2324.csv` (72 schools, no
Mission Bay).

## Family B — Mission Bay included, 2023-24 vs 2026-27

`mb_2324.yaml`, `mb_2627.yaml`

| cell | programs source | programs | students | capacity | Mission Bay |
|---|---|---:|---:|---:|---|
| `mb_2324` | `programs_statusQuo_2324.csv` | 165 | 3,953 | 4,298 | `999-GE-KG` 66 |
| `mb_2627` | `programs_withMissionBay_2627.csv` | 162 | 3,855 | 4,306 | 69 / 5 / 0 |

School table for both: `schools_rehauled_withMissionBay_2324.csv` (73 schools).

## The one trap

**The 2023-24 baseline is not the same in the two families.** `status_quo` is
the only 2023-24 bundle that carries Mission Bay at all, and it is a different
roster from the `default` table Family A uses: 169 rows vs 164, 32 programs
with different capacity, 7 programs only in `status_quo`, 2 only in `default`.

So `nomb_2324` and `mb_2324` differ by more than Mission Bay, and differencing
a Family A number against a Family B one attributes that whole roster change to
Mission Bay. Compare **within** a family.

If you want to isolate the Mission Bay effect cleanly, add a sixth cell: 2026-27
without Mission Bay. That bundle exists (`programs_2627.csv`, 159 programs) and
is one filter flip — copy `mb_2627.yaml` and set `include_mission_bay: false`.
`mb_2627` vs that cell is a same-year, same-students, same-preferences contrast
in which Mission Bay is the only thing that changes.

## Running

Each cell is `assignment/configs/8-18-real-pref.yaml` with four things
changed: the `data:` block, `paths.assignment-folder`, `output_dir`, and
`neighborhood_assignment_#5` appended to `subconfigs`. Same seed (2023), 25
iterations, real preferences (`utility-model.enable: false`). The extra
subconfig is in the five cells only — the shared template is untouched, so
other runs using it are unaffected.

Eight subconfigs means each cell plans to 16 Slurm jobs (8 assignment + 8
dependent metrics) at 25 CPUs each. The repository caps a plan at
`MAX_ASSIGNMENT_JOBS = 12` / `MAX_METRICS_JOBS = 8` and running jobs at
`MAX_RUNNING_SLURM_JOBS = 12` (`assignment/slurm_graph.py`), and the cluster's
own `soal` QoS is `MaxJobsPU=12, MaxSubmitPU=20`, so **submit one cell at a
time** — a second concurrent cell would be 32 submitted and rejected.

Re-submitting a cell that has already run is cheap. `reuse_assignments: true`
plus the launcher's default `--skip-existing` makes `_assignment_batches` drop
any iteration whose CSV already exists and skip a task with nothing left to do,
so only genuinely missing work runs. The metrics jobs do recompute, which is
what rewrites `aggregate_metrics/` across the full subconfig set — that is how
`neighborhood_assignment_#5` was added without rerunning the other seven.

```bash
uv run python -m assignment.run_custom_config --config-path assignment/configs/mb_comparison/nomb_2324.yaml --workers 7
```

Outputs go to `/soalnas/share/data/school_choice/local_runs/mb_comparison/<cell>/`,
one directory per cell, so runs do not collide and `reuse_assignments: true` is
safe.
