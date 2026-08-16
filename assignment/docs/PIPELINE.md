# From Clone to First Simulation

End-to-end runbook: everything needed to go from a fresh `git clone` to a
running simulation. Each step links to the detailed guide where relevant.

> **TL;DR.** A bare clone is **not** runnable on its own: the confidential data
> is not in the repo. The normal order is: **env → shared data → choose registry
> selectors → provide zones only for zone-restricted policies → run.** Prepared
> student/program files are optional experimental inputs, not a standard setup
> step.

---

## Step 0 — Python environment

```bash
uv sync            # install pinned deps into .venv (see README for uv setup)
# prefix commands below with `uv run`, or `source .venv/bin/activate` first
```

## Step 1 — Get the confidential SFUSD data

The data is **not** redistributable and is **not** in the repo. Every
environment must provide the central catalog's data root, which defaults to
`/share/data/school_choice/`. Local and cluster path configs contain outputs
only. There is no home-directory data fallback.

See **[DATA_SETUP.md](DATA_SETUP.md)** for the per-file path reference.

## Step 2 — Substitute the placeholder tokens

Some exceptional experiment and generated-zone configs use placeholder tokens
instead of developer home directories. Replace those tokens with **absolute**
paths; ordinary registry-backed inputs need no substitution.

| Token | Replace with |
|-------|--------------|
| `<STUDENT_ASSIGNMENT_PATH>` | checkout containing generated zones or experiment outputs |
| `<SFUSD_CHOICE_PATH>` | your `SFUSD-Choice` checkout (MNL estimates) |
| `<RA_SFUSD_PATH>` | your `RA_SFUSD` checkout (permuted-students configs only) |

```bash
# Example: point everything at the current checkout.
grep -rl '<STUDENT_ASSIGNMENT_PATH>' configs/ \
  | xargs sed -i "s#<STUDENT_ASSIGNMENT_PATH>#$PWD#g"
```

Details and the source-role contract: **[DATA_SETUP.md](DATA_SETUP.md)**.

## Step 3 — Select Registry Data

A standard run selects a canonical year, one execution grade, student
population, rounds, special-program mode, capacity profile, and Mission Bay
policy under `data.overrides.filters.assignment`. The central `school_years`
registry resolves the annual student/program/school bundle. Unsupported
combinations fail rather than falling back; see **[DATA_SETUP.md](DATA_SETUP.md)**
for the exact block.

### Optional Experimental Preprocessing

Prepared round-one, no-special, and distance-filtered files are retained only
for historical reproduction or experiments that explicitly override registry
sources. Runtime selectors perform standard round and special-program filtering
without materializing these files.

The historical program export can be reproduced with:

```bash
python scripts/preprocessing/filter_programs.py
# reads  /share/data/school_choice/Data/Cleaned/programs_{YY}.csv  (years 2013–2023)
# writes local-data/program_filter/programs_without_specialprogs_{YY}.csv
```

The optional student experiment drops GE choices that are too far away (or
outside the N closest schools). It is driven by Hydra config
[`configs/custom_configs/distance_filter.yaml`](../configs/custom_configs/distance_filter.yaml)
— edit its paths / filter mode, then:

```bash
python filter_student_choices.py
# or override keys on the CLI:
python filter_student_choices.py distance=2.0 \
  output_csv=$PWD/local-data/student_filter/student_2324_filtered.csv
# writes a file an experiment can select with an explicit source override.
```

Only experiments that require materialized filtered files should point
`data.overrides.sources.assignment.students` or `assignment.programs` at these
outputs. Source overrides take precedence over registry roles.

## Step 4 — Zones (only for zone-restricted policies)

Skip this for baselines like `status_quo_real` (no zone restriction → no zone
file is ever opened). Two kinds of zone files appear in the configs:

- **Shared cluster zones** — `/share/data/school_choice/simulation-files/zones/*.csv`.
  Already exist; nothing to generate.
- **Locally-generated zones** — `<STUDENT_ASSIGNMENT_PATH>/data/zones/Zones_*.csv`
  (e.g. `Zones_10_FRL_Dev_0.15_Objective_2250.0_10-zone-3.csv`). These come from
  `.pkl` files produced by an **external upstream zone-optimization pipeline**
  (`Zone_Generation`), which is **provided separately and is not part of this
  repo**. Once you have the `.pkl` files, convert them to CSV:

  ```bash
  python scripts/generators/generate_zone_from_pickle.py \
      --input-dir /path/to/Generated_Zones/Zones_10/FRL_Dev_0.15/ \
      --output-dir data/zones/ \
      --building-blocks block_group
  ```

Full zone workflow (register, policy config, building blocks):
**[GENERATE_ZONES.md](GENERATE_ZONES.md)**.

## Step 5 — Run a simulation

| Goal | Command | Needs |
|------|---------|-------|
| A registry-backed config | `uv run python run_custom_config.py --config-path <config.yaml>` | Steps 0, 1, and 3; Step 4 only for a zoned policy. |
| Historical custom config | `uv run python run_custom_config.py --config-path configs/custom_configs/status_quo_real_2324.yaml` | Its explicit experimental sources and any referenced zones. |
| Augmented DA | `uv run python run_augmented_da.py --config-path configs/custom_configs/augmented_da_2324.yaml` | Its explicit experimental sources and local zones. |
| Full pipeline (generate → simulate → analyze) | `bash scripts/run_models_estimates.sh --settings scripts/settings/models_cluster.env` | Steps 0–4 + MNL estimates. |

`--config_path` is also accepted. Use `--help` for the argparse scripts.

---

## Dependency cheat-sheet

| Config family | Registry sources | Experimental overrides | Zones | MNL estimates |
|---------------|:---:|:---:|:---:|:---:|
| Standard assignment run | Required | No | Policy-dependent | Model-dependent |
| Historical custom configs | Required | As declared | Policy-dependent | As declared |
| Generated-zone sweeps | Required | Generated-zone root | Required | Model-dependent |

## See also

- **[DATA_SETUP.md](DATA_SETUP.md)** — which config key points to which file, placeholder tokens.
- **[GENERATE_ZONES.md](GENERATE_ZONES.md)** — `.pkl` → CSV zone conversion and registration.
