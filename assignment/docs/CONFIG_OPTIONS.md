# Configuration Reference

This document describes every configuration layer used by the simulator
(`run_custom_config.py`) and the analysis step
(`scripts/analysis/analyze_trends.py`).

## 1. Config layering

Configuration is resolved in this order (later layers override earlier ones):

1. **`configs/base_config.yaml`** — simulation defaults (grade, iterations,
   utility-model defaults, subconfig list).
2. **Path config** — `configs/local_path_config.yaml` locally, or
   `configs/cluster_path_config.yaml` when the hostname contains `soal`
   (cluster). Machine-specific paths live here, never in code.
3. **`configs/<user>.config.yaml`** — auto-created on first run by merging
   the two layers above (`<user>` = your login). Edit it for personal
   defaults. Validated against `configs/config_schema.yaml` (yamale).
4. **Custom run YAML** — passed via
   `python run_custom_config.py --config-path <file>`. Replaces the config
   wholesale. Supports `${var}` substitution from top-level keys and CLI
   overrides `--sample`, `--frac`, `--workers N` (parallel subconfigs).
5. **Policy subconfig** — for every name in `subconfigs:`, the file
   `configs/policy_configs/<name>.yaml` is merged on top (one simulation
   per entry). Validated against `configs/policy_configs/policy.schema.yaml`.

The end-to-end pipeline `scripts/run_models_estimates.sh` generates
custom run YAMLs automatically; its own knobs (paths, run matrix) come from
a sourced settings file (`scripts/settings/*.env`, see `--settings`).

## 2. Top-level simulation keys

| Key | Type | Meaning |
|-----|------|---------|
| `year` | int | 2-digit school year (e.g. `22` = 2022-23). Selects data files and year-specific logic. |
| `grade` | str | `KG`, `06`, or `09`. Drives priorities and file naming. |
| `random-seed` | int | Seed reset before each subconfig, so subconfig order does not matter. |
| `iterations.start` / `iterations.end` | int | Iteration range; one DA run (and one utility redraw) per iteration. |
| `save-assignment` | bool | Save assignment CSVs to `paths.assignment-folder` (else simulate() returns a generator). |
| `r1-only` | bool | Treat round 1 as the only round when reconstructing final assignments. |
| `remove-special-lps` | bool | Drop students who ranked special programs and drop special programs themselves. |
| `rounds-merged-options` | list | Round-merging variants to simulate: `0` (no merge), `123`, `12`, `23`. |
| `read-lotteries` | bool | Read tie-breaker lotteries from `paths.lotteries-path` instead of drawing them. |
| `subconfigs` | list(str) | Policy subconfig names to run (files in `configs/policy_configs/`). |

## 3. `paths.*`

| Key | Meaning |
|-----|---------|
| `sfusd` | SFUSD shared-data root. Relative data paths resolve against it (`Data/Cleaned/...`). |
| `student-data` | Explicit student CSV (overrides the `Cleaned/student_<year>.csv` default). |
| `program-data` | Explicit program CSV (e.g. `programs_without_specialprogs_<year>.csv`). |
| `school-data` | Explicit schools CSV (`schools_rehauled_<year>.csv`; first column `school_id`, needs `category`, `lat`, `lon`). |
| `student-save` | Cache directory: computed student-program distances and preference pickles are saved/loaded here. Delete it to force recomputation. |
| `assignment-folder` | Output directory for assignment CSVs + a copy of the config used. |
| `estimate-path` | Utility estimates. `.csv` = `studentno` (`<year>-<no>`) × `program_id` matrix of utilities (`-inf` allowed; missing rows/columns auto-filled with `-inf`). `.npy` = raw matrix aligned with student/program indices. |
| `zone-files` | Map of policy name → zone CSV. Each zone CSV row is one zone: comma-separated geounit ids (attendance areas, block groups, or blocks depending on `zone-building-blocks`). |
| `citywide-or-lp-zones` | Map of supplemental zone name → file; only loaded when a policy sets `citywide-or-lp`. |
| `lotteries-path` | Tie-breaker lottery files (used with `read-lotteries: true`). |
| `student-codex` / `program-codex` | `.npy` codex files for `read-precomuted-umodel-prefs`. |
| `new-ctip-path` / `new-ctip-blockgroup-path` | `.npy` block lists for the `new_ctip` / `new_ctip_blockgroup` equity tie-breakers (defaults to the legacy cluster paths). |

## 4. `utility-model.*`

| Key | Meaning |
|-----|---------|
| `enable` | Use the utility model to generate preferences (else historical round-1 lists are used). |
| `list-length` | How many programs each student ranks (see below). |
| `gumbel-scale` | Scale of the Gumbel noise added to utilities. `0` = deterministic ranking by utility; default `1.0` (MNL draw). |
| `designate-lp-for-all` | Include language programs in everyone's designation ordering (not only requesters). |
| `save-path` | Where to save the drawn utility matrix (`.csv` or `.npy`). |
| `read-precomuted-umodel-prefs` | Read precomputed preference draws (`.npy` + codex files) instead of drawing. |

`list-length` options (from `PreferenceGenerator.set_number_programs_ranked`):

- `"7"` (any numeric string) — everyone ranks that many programs.
- `real_length` — each student's historical list length (`num_ranked`).
- `real_length_x2` — twice the historical length.
- `0.8*round(real_length)`, `0.7*round(real_length)`, `0.6*round(real_length)`
  — scaled historical length, floored at 3.
- `0.5*round(real_length)` — half length (ceil, no floor).
- `length_by_ethn` / `length_by_ctip` / `length_by_frl` / `length_by_income`
  — group-average historical length by ethnicity / CTIP1 / FRL score / income
  (95292 threshold).
- `all_eligible` — rank every eligible program.

## 5. Policy subconfigs (`configs/policy_configs/*.yaml`)

Validated against `configs/policy_configs/policy.schema.yaml`.

| Key | Meaning |
|-----|---------|
| `assignment-algorithm` | `DA` (deferred acceptance) or `TTC` (top trading cycles). |
| `policies` | List of zone policies to simulate; each must be a key of `paths.zone-files`, or `real_match` to read the historical assignment instead of running DA. |
| `zone-building-blocks` | Geounit type of the zone files: `attendance_area`, `block_group`, `block`, or `home_based` (JSON studentno → program list). |
| `ctip-options` | Equity tie-breaker variants: `0` (none), `1` (CTIP1), `5` (5-level CTIP types), `new_ctip`, `new_ctip_blockgroup`, `"<n>D"` (HOCidx1 quantile categories), or a map (`column`, `num_categories`/`thresholds`, `lower_disadvantaged`) for a custom tiebreaker. One simulation per option. |
| `ties-options` | Lottery variants: `STB` (single), `MTB` (multiple), `STB_REAL` / `MTB_REAL` (historical round-1 random numbers), `STBcoordinated` (shared per block group). One simulation per option. |
| `restrict-zone` | `false` = zones grant priority only; `true` = students can only access in-zone programs; `CTIP_access` = CTIP students keep citywide access. `restrict-zone-options` lists several variants. |
| `citywide-or-lp` | Supplemental zone names (keys of `paths.citywide-or-lp-zones`) granting extra program access. |
| `sibling-access` | Siblings grant zone eligibility at the sibling's school. |
| `priority-weights` | Map of priority component → weight: `ctip`, `sibling`, `zone`, `prek`, `distance`, `peng`, plus a `language-programs` sub-map (`lp-sibling`, `lp`, `sibling`, `ctip`) applied at citywide language programs. For grade 06/09, categories like `brown-ms`, `bayview-students`, `remaining`, `brown-ms-to-hs`, `msf` apply. |
| `distance-priority` | How the `distance` weight is computed: `step-size: <miles>`, `thresholds: [...]` (+ optional `weights: [...]`), or `continuous: x` / `1_over_x_sqaure`. |
| `distance-boost` | Income-based distance boost: `income_threshold` (default 95292), `low_income_boost` (default 0.2). |
| `guard-rails` | `-1` = off; `0`/`1` = soft/strict reserve guardrails (uses `reserve-settings`). `guard-rails-reserve-options` lists several variants. |
| `reserve-settings` | Reserve definition map (e.g. FRL reserve ratios) used when guardrails are on. |
| `citywide-separate-reserves` / `citywide-reserve-ratios` | Separate reserve handling at citywide schools (default `true`, `[0.57, 0.43]`). |
| `designate` | Append designation programs (closest eligible GE/LP) to preference lists. |
| `designation-ordering-type` | `in_zone` (default; in-zone programs first) or `simple` (pure distance order). |
| `non_designation_boost` / `soft_reserve_boost` | Priority boosts for ranked (non-designation) programs / soft reserves. |
| `add_aa_schools` | Append each student's grade-specific attendance-area GE program to their ranked list when it is not already present. |
| `remove_non_aa_or_citywide` | Keep only programs at each student's attendance-area school or schools categorized as citywide, including designation options. Utility-model lists select their top allowed programs before list-length truncation. |
| `aa_boost` | Add this priority boost for a student at their grade-specific attendance-area GE program. |
| `truncate-at-AA-GE` | Truncate utility-model lists at the student's attendance-area GE program. |

## 6. `list-augmentation` (alternative programs, `run_augmented_da.py`)

| Key | Meaning |
|-----|---------|
| `enable` | Turn on preference-list augmentation. |
| `targeting-method` | Which students get augmented lists: `ctip_x_ethnicity` (CTIP1 × AALPI ethnicities) or `short_list_threshold`. |
| `short-list-threshold` | Max list length to qualify under `short_list_threshold`. |
| `oversubscribed-method` | How oversubscribed programs are found: `first_choice_per_seat`, `apps_per_seat`, or `fixed_list`. |
| `oversubscribed-ratio-threshold` | Demand/capacity ratio above which a program counts as oversubscribed (default 1.5). |
| `oversubscribed-fixed-schools` | School ids to use with `fixed_list`. |
| `max-augmented-programs` | Max programs appended per student (default 1). |

Example config: `configs/custom_configs/augmented_da_2324.yaml`.

## 7. analyze_trends config (`scripts/analysis/analyze_trends.py --config <yaml>`)

```yaml
output_dir: metrics/my_experiment
schools_data: /path/to/schools_rehauled_<year>.csv   # evaluator lat/lon file
new_ctip_path: /path/to/ETB_2024.npy                 # optional equity blocks
runs:
  - label: "my_run"               # column name in the Excel
    folder: path/to/run/subconfig # evaluated recursively (all CSVs), or:
    # run_csv: path/to/single.csv
    year: 22                      # 2-digit year
    program_data: path/to/programs_without_specialprogs_2223.csv
    student_data: path/to/student_2223_filtered.csv
    # schools_data / new_ctip_path may also be set per run
row_order: ["Distance Av (All Assigned)", ...]   # metric row ordering
single_metrics: [...]    # optional: per-metric error-bar plots
group_metrics: [...]     # optional: "Metric ({group})" grouped plots
```

Output: `<output_dir>/metrics_comparison.xlsx` with sheets `Mean Values`,
`Std Values`, `Mean ± Std` (+ per-year sheets when run via the pipeline
script) and diagnostic plots under `<output_dir>/diagnostics/`.

Metric definitions and the distinction between `eval_assignment_basic()` and
`eval_assignment_full()` are documented in `ASSIGNMENT_METRICS.md`.

## 8. Pipeline settings files (`scripts/settings/*.env`)

`scripts/run_models_estimates.sh --settings <file>` sources a bash
settings file. Scalars use `: "${VAR:=default}"`, so environment variables
always win; the run matrix uses bash arrays. See
`scripts/settings/models_cluster.env` (production values, including all
`/share/data` paths) and `scripts/settings/models_test.env` (committed
fake dataset in `tests/fixtures/fake_2223/`, exercised by
`tests/test_full_pipeline.py`).
