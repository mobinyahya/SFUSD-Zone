# Data Setup Guide

Assignment runs use the repository-level `loaders/` scenario framework. Input
selection is independent of the current working directory and hostname.

## Run Data Block

Every executable assignment YAML must contain exactly this outer shape:

```yaml
data:
  scenario: legacy
  overrides: {}
```

`Configerator` validates this block for both YAML-loaded and in-memory configs
and calls `load_scenario` immediately. Top-level data selectors and all input
keys under `paths` are errors.

`paths.assignment-folder` remains an output setting. Local and cluster path
configs contain output paths only; there is no hostname-based source selection.

## Registry and Source Roles

`loaders/configs/base.yaml` schema 2 is the single catalog and central
`school_years` registry. Its default source root is
`/soalnas/share/data/school_choice`; its default cache root is
`/soalnas/share/data/school_choice/Data/caches`. Scenarios select invariant roles and
provide complete default selectors. The merged run filters derive annual
student, program, and school roles from the registry.

Use a direct source override only for an exceptional experiment. It takes
precedence over the registry source:

```yaml
data:
  scenario: legacy
  overrides:
    sources:
      assignment.students:
        path: /absolute/path/student_2324.csv
        classification: restricted
```

Other assignment roles are `program_codes`, `block_data`, `new_ctip`,
`new_ctip_blockgroup`, and optional `lotteries`. Dynamic generated plans use
`data.overrides.sources.assignment.zones` with the same named-map shape.

Direct paths in checked-in experiment configs preserve their original source
choices. Placeholder tokens such as `<STUDENT_ASSIGNMENT_PATH>` must be replaced
with absolute paths before execution.

## Assignment Filters

Canonical year, execution grade, student population, preference rounds,
special-program handling, capacity profile, and Mission Bay inclusion belong
to the assignment filter:

```yaml
data:
  scenario: legacy
  overrides:
    filters:
      assignment:
        year: "2324"
        grades: [KG]             # assignment execution requires exactly one
        student_population: applicant  # applicant | enrolled
        rounds: [1]                    # or all
        special_programs: include     # include | exclude_only_special | exclude_any_special
        capacity_profile: default
        capacity_scenario: programs   # programs | A | B | C | D
        include_mission_bay: false
        geography_vintage: "2020"
        outside_district_students: ignore  # ignore | include
```

Assignment currently executes one year and one grade per market. Unsupported
registry combinations fail with the available years, grades, profiles, or
Mission Bay variants and never fall back.

The default `capacity_scenario: programs` uses capacities from the program table
selected by `capacity_profile`. Explicit scenarios overlay matching
school/program/grade rows from the central scenario table.

Points outside the selected district Census geometry are not snapped to nearby
Blocks. Their Census geography remains blank; `outside_district_students: ignore`
filters them by default, while `include` retains them without geographic-zone
eligibility.

Participation always means a nonempty choice in any selected round after
filtering. Selected rounds are sorted and each unique student is returned once.
Choices, historical lotteries, cohort metadata, and choice-derived eligibility
come only from the earliest remaining selected round and are exposed through
`first_participating_round` and aligned `selected_*` fields.

- `include` keeps all students and alternatives.
- `exclude_only_special` removes special alternatives and keeps students with
  any eligible selected-round choice.
- `exclude_any_special` removes a student if any selected-round choice is
  special.

`include_mission_bay` centrally applies Mission Bay inclusion and derives
`909 -> 999` across student, program, and school tables. Do not provide an alias
map.

## Prepared KG Inputs

Prepared round-one and no-special files remain catalogued for historical
reproduction and exceptional experiments. They are not standard runtime
student inputs; normal runs select the annual source and apply `rounds` and
`special_programs` at load time. To reproduce the old exports:

```bash
uv run python -m assignment.scripts.preprocessing.prepare_kg_r1_inputs \
  --output-dir /soalnas/share/data/school_choice/Data/Cleaned/choice_inputs_2324
```

Only an experiment that specifically requires those materialized files should
override `assignment.students` or `assignment.programs` to use them.

## Replay and Identity

Saved `config.json`/`config.yaml` files retain the anchored strict external
configuration and replay from another working directory. Runtime-only resolved
paths and provenance fields are not written into that snapshot. Immutable
assignment source identity includes assignment filters and checksummed
student/program/school/estimate inputs but excludes zone files, allowing loaded
tables to be reused across policy-only changes. Cache-root relocation alone does
not change identity.

## Distance Cache

Student-program distances are CacheStore artifact
`student_program_distances`, schema v4. The default root is:

```text
/soalnas/share/data/school_choice/Data/caches
```

Cache identity includes student/program/school-coordinate source contents,
assignment filters, an opaque fingerprint and count of active ordered student
identities, and the distance algorithm version. Raw student IDs occur only in
the restricted-derived payload, never in cache paths, manifests, parameters,
or references. Moving only the cache root does not change identity. Synthetic
runs should set `data.overrides.roots.cache` to a temporary directory.

## Troubleshooting

- **Legacy config field rejected**: move the filter to
  `data.overrides.filters.assignment` or the input to
  `data.overrides.sources`.
- **Unknown zone**: the policy name must match a key in `assignment.zones`;
  supplemental names must match `assignment.citywide_zones`.
- **Missing estimate**: override `assignment.estimate` with the concrete model
  output used by the run.
- **Unexpected Mission Bay ID**: verify `include_mission_bay`; enabled runs
  centrally map school `909` to `999`, while disabled runs exclude both IDs.
