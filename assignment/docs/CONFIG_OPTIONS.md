# Configuration Reference

This document describes every configuration layer used by the simulator
(`run_custom_config.py`) and the analysis step
(`scripts/analysis/analyze_trends.py`).

## 1. Config layering

Configuration is resolved in this order (later layers override earlier ones):

1. **`configs/base_config.yaml`** — simulation defaults (data scenario, iterations,
   utility-model defaults, subconfig list).
2. **Output path config** — `configs/local_path_config.yaml` is used when a
   personal config is created. Local/cluster path files contain outputs only.
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
| `data` | map | Strict scenario selection described below. Required in every executable run config. |
| `random-seed` | int | Seed reset before each subconfig, so subconfig order does not matter. |
| `iterations.start` / `iterations.end` | int | Iteration range; one DA run (and one utility redraw) per iteration. |
| `save-assignment` | `true` | Required. Assignment CSVs are saved to `paths.assignment-folder`. |
| `r1-only` | bool | Treat round 1 as the only round when reconstructing final assignments. |
| `rounds-merged-options` | list | Round-merging variants over chronological ordinals: `0` (no merge), `all`, or legacy three-round codes `123`, `12`, `23`. Legacy codes are rejected when more than three rounds are selected. |
| `read-lotteries` | bool | Read tie-breaker lotteries from the `assignment.lotteries` scenario role instead of drawing them. |
| `subconfigs` | list(str) | Policy subconfig names to run (files in `configs/policy_configs/`). |

## 3. `data`

Executable configs select a bundled scenario name (for example, `legacy` or
`mission-bay-2324`) or a custom scenario YAML, plus an `overrides` map.
`loaders/configs/base.yaml` schema 2 is the single file catalog and central
`school_years` registry. Scenarios define invariant source roles and complete
selector defaults. A normal run changes selectors under
`data.overrides.filters`:

```yaml
data:
  scenario: historical-2324
  overrides:
    filters:
      assignment:
        year: "2324"
        grades: [KG]
        student_population: applicant  # applicant | enrolled
        rounds: [1]                    # or all
        special_programs: include     # include | exclude_only_special | exclude_any_special
        capacity_profile: default
        capacity_scenario: programs   # programs | A | B | C | D
        include_mission_bay: false
        geography_vintage: "2020"
        outside_district_students: ignore  # ignore | include
```

Assignment execution requires exactly one canonical year and one grade per
market, even though `grades` is a list and optimization supports multiple years
and grades where available. The registry must contain the requested student
population and the year/grade/capacity-profile/Mission Bay program-school
bundle. Unsupported combinations fail and never fall back.

`capacity_profile` selects the program table registered for the assignment
market. `capacity_scenario: programs` uses its capacities directly and is the
default. An explicit scenario overlays matching school/program/grade values from
the central capacity-scenario table; programs absent from that table retain
their selected-table capacities.

Student coordinates outside the selected district Census geometry have blank
Block, BlockGroup, and Tract values. The default `ignore` policy removes those
students; `include` keeps them in the assignment market without geographic-zone
priority.

Direct source objects are reserved for exceptional experimental inputs. They
override registry-derived roles:

```yaml
data:
  scenario: historical-2324
  overrides:
    sources:
      assignment.students:
        path: /absolute/path/experimental_students.csv
        classification: restricted
```

Assignment roles include `students`, `programs`, `schools`,
`school_coordinates`, `program_codes`, `estimate`, `block_data`, `new_ctip`,
`new_ctip_blockgroup`, `zones`, `citywide_zones`, and optional `lotteries`.
Named zone maps can be extended with
`data.overrides.sources.assignment.zones`.

Relative custom scenario paths, root overrides, and rootless direct source
paths are resolved from the YAML file that declares them. APIs receiving only
an in-memory mapping have no declaring file, so they resolve those paths from
the current working directory. Saved run configs contain the anchored `data`
map and can therefore be reloaded from another working directory.

Assignment filters own the canonical school year, grades list,
applicant/enrolled population, selected round labels, special-program mode,
capacity profile, and Mission Bay inclusion. `include_mission_bay` centrally
derives the `909 -> 999` alias across tables; users do not configure an alias
map.

Participation is always any selected round. The loader sorts selected rounds,
applies Mission Bay and special-program filtering, removes students with no
remaining selected-round choice, and returns one row per unique student.
`first_participating_round` is the earliest remaining round label and its
ordinal is zero-based within the selected rounds. Choices, historical lottery
values, cohort metadata, and choice-derived eligibility all come only from that
round's aligned `selected_*` fields.

The special-program modes are exact:

- `include` keeps all students and alternatives.
- `exclude_only_special` removes special alternatives and keeps students with
  any eligible selected-round choice.
- `exclude_any_special` removes a student if any selected-round choice is
  special.

As a non-contractual smoke check, the real 2023-24 KG applicant data with all
rounds and `include` produced 4,304 unique students: 3,955 first participated in
round 1, 212 in round 2, and 137 in round 4. Counts depend on the selected data
and filters and are not universal.

Student-program distances use CacheStore artifact
`student_program_distances/v4` under
`/share/data/school_choice/Data/caches` by default. Cache identity includes
source contents, filters, an opaque ordered student-identity fingerprint and
count, and the algorithm version. The restricted-derived cache manifest and
reference never contain raw student IDs.

Saved `config.json`/`config.yaml` files contain the anchored strict external
configuration and can be replayed from another working directory. Runtime-only
resolved input keys and `data-provenance` are excluded. Runtime provenance and
market reuse fingerprints include assignment filters and checksummed immutable
table/estimate sources; zone files are excluded so policy-only changes can
reuse loaded tables. Moving only the cache root does not change that identity.

## 4. `paths.*`

`paths` is output-only. `assignment-folder` is the directory for assignment
CSVs and the replayable config snapshot. Legacy input entries are rejected.

## 5. `utility-model.*`

| Key | Meaning |
|-----|---------|
| `enable` | Use the utility model to generate preferences (else each student's first-participating selected list is used). |
| `list-length` | How many programs each student ranks (see below). |
| `gumbel-scale` | Scale of the Gumbel noise added to utilities. `0` = deterministic ranking by utility; default `1.0` (MNL draw). |
| `designate-lp-for-all` | Include language programs in everyone's designation ordering (not only requesters). |
| `save-path` | Where to save the drawn utility matrix (`.csv` or `.npy`). |

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

## 6. Policy subconfigs (`configs/policy_configs/*.yaml`)

Validated against `configs/policy_configs/policy.schema.yaml`.

| Key | Meaning |
|-----|---------|
| `assignment-algorithm` | `DA` (deferred acceptance). |
| `policies` | List of zone policies to simulate; each must be a key of `assignment.zones`, or `real_match` to read the historical assignment instead of running DA. |
| `zone-building-blocks` | Geounit type of the zone files: `attendance_area`, `block_group`, `block`, `tract`, or `home_based` (JSON studentno → program list). |
| `ctip-options` | Equity tie-breaker variants: `0` (none), `1` (CTIP1), `5` (5-level CTIP types), `new_ctip`, `new_ctip_blockgroup`, `"<n>D"` (HOCidx1 quantile categories), or a map (`column`, `num_categories`/`thresholds`, `lower_disadvantaged`) for a custom tiebreaker. One simulation per option. |
| `ties-options` | Lottery variants: `STB` (single), `MTB` (multiple), `STB_REAL` / `MTB_REAL` (historical selected first-participating random numbers), `STBcoordinated` (shared per block group). One simulation per option. |
| `restrict-zone` | `false` = zones grant priority only; `true` = students can only access in-zone programs; `CTIP_access` = CTIP students keep citywide access. `restrict-zone-options` lists several variants. |
| `citywide-or-lp` | Supplemental zone names (keys of `assignment.citywide_zones`) granting extra program access. |
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
| `drop_below_aa` | Remove programs ranked after the student's attendance-area GE program (default `false`). Runs after `add_aa_schools` when both are enabled. |
| `remove_non_aa_or_citywide` | Keep only programs at each student's attendance-area school or schools categorized as citywide, including designation options. Utility-model lists select their top allowed programs before list-length truncation. |
| `aa_boost` | Add this priority boost for a student at their grade-specific attendance-area GE program. |
| `overscribe_aa` | Assign otherwise-unassigned students to their attendance-area GE program even when it is at capacity (default `false`). |
| `truncate-at-AA-GE` | Truncate utility-model lists at the student's attendance-area GE program. |

## 7. `list-augmentation` (alternative programs, `run_augmented_da.py`)

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

## 8. analyze_trends config (`scripts/analysis/analyze_trends.py --config <yaml>`)

```yaml
output_dir: metrics/my_experiment
schools_data: /path/to/schools_rehauled_<year>.csv   # evaluator lat/lon file
new_ctip_path: /path/to/ETB_2024.npy                 # optional equity blocks
runs:
  - label: "my_run"               # column name in the Excel
    folder: path/to/run/subconfig # evaluated recursively (all CSVs), or:
    # run_csv: path/to/single.csv
    data:                         # preferred: shared normalized tables
      scenario: mission-bay-2324
      overrides: {}
    # Historical analysis configs may instead provide explicit standalone tables.
row_order: ["Distance Av (All Assigned)", ...]   # metric row ordering
single_metrics: [...]    # optional: per-metric error-bar plots
group_metrics: [...]     # optional: "Metric ({group})" grouped plots
```

Output: `<output_dir>/metrics_comparison.xlsx` with sheets `Mean Values`,
`Std Values`, `Mean ± Std` (+ per-year sheets when run via the pipeline
script) and diagnostic plots under `<output_dir>/diagnostics/`.

Metric definitions and the distinction between `eval_assignment_basic()` and
`eval_assignment_full()` are documented in `ASSIGNMENT_METRICS.md`.

## 9. Pipeline settings files (`scripts/settings/*.env`)

`scripts/run_models_estimates.sh --settings <file>` sources a bash
settings file. Scalars use `: "${VAR:=default}"`, so environment variables
always win; the run matrix uses bash arrays. See
`scripts/settings/models_cluster.env` (production values, including all
`/share/data` paths) and `scripts/settings/models_test.env` (committed
fake dataset in `tests/fixtures/fake_2223/`, exercised by
`tests/test_full_pipeline.py`).
