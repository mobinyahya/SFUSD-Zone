# From Clone to First Simulation

End-to-end runbook: everything needed to go from a fresh `git clone` to a
running simulation. Each step links to the detailed guide where relevant.

> **TL;DR.** A bare clone is **not** runnable on its own: the confidential data
> is not in the repo, and the filtered inputs under `local-data/` are
> git-ignored. The order is: **env → data → substitute placeholders →
> generate filtered inputs → (zones, only for zone-restricted policies) → run.**
> Generating zones is *not* the first step, and most baselines don't need zones
> at all.

---

## Step 0 — Python environment

```bash
uv sync            # install pinned deps into .venv (see README for uv setup)
# prefix commands below with `uv run`, or `source .venv/bin/activate` first
```

## Step 1 — Get the confidential SFUSD data

The data is **not** redistributable and is **not** in the repo.

- **On the Stanford cluster** (hostname contains `soal`): nothing to do — the
  data already lives at `/share/data/school_choice/`, and
  `configs/cluster_path_config.yaml` points there.
- **Off the cluster**: copy the data tree (and the shared zone files under
  `simulation-files/zones/`) from the cluster, then point
  `configs/local_path_config.yaml` (or your generated
  `configs/<username>.config.yaml`) at your local copy.

See **[DATA_SETUP.md](DATA_SETUP.md)** for the per-file path reference.

## Step 2 — Substitute the placeholder tokens

Committed configs use explicit placeholder tokens instead of anyone's home
directory. Replace them with your **absolute** paths (cluster paths
`/share/...` are already concrete and need no change):

| Token | Replace with |
|-------|--------------|
| `<STUDENT_ASSIGNMENT_PATH>` | your `student-assignment` checkout |
| `<SFUSD_CHOICE_PATH>` | your `SFUSD-Choice` checkout (MNL estimates) |
| `<SFUSD_DATA_PATH>` | your local copy of the SFUSD data tree (off-cluster `policy_configs` variant only) |
| `<RA_SFUSD_PATH>` | your `RA_SFUSD` checkout (permuted-students configs only) |

```bash
# Example: point everything at the current checkout.
grep -rl '<STUDENT_ASSIGNMENT_PATH>' configs/ \
  | xargs sed -i "s#<STUDENT_ASSIGNMENT_PATH>#$PWD#g"
```

Details and the full key→file table: **[DATA_SETUP.md](DATA_SETUP.md)**.

## Step 3 — Generate the filtered inputs (`local-data/`)

`local-data/` is git-ignored, so a fresh clone has none of it. The
`custom_configs` simulators read two filtered inputs that you must produce
(or copy):

### 3a. Program filter → `local-data/program_filter/`

Removes special programs from each year's cleaned program file. No arguments;
run from the repo root:

```bash
python scripts/preprocessing/filter_programs.py
# reads  /share/data/school_choice/Data/Cleaned/programs_{YY}.csv  (years 2013–2023)
# writes local-data/program_filter/programs_without_specialprogs_{YY}.csv
```

### 3b. Student filter → `local-data/student_filter/`

For each student, drops the GE choices that are too far away (or outside the N
closest schools). Driven by Hydra config
[`configs/custom_configs/distance_filter.yaml`](../configs/custom_configs/distance_filter.yaml)
— edit its paths / filter mode, then:

```bash
python filter_student_choices.py
# or override keys on the CLI:
python filter_student_choices.py distance=2.0 \
  output_csv=$PWD/local-data/student_filter/student_2324_filtered.csv
# writes the file the simulators read back as `student-data`.
```

> Tip: the simulator config's `student-data` / `program-data` keys must point at
> the files produced here. If you'd rather skip filtering, repoint those keys at
> the cluster's cleaned files instead.

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
| A custom config | `uv run python run_custom_config.py --config-path configs/custom_configs/status_quo_real_2324.yaml` | Steps 0–3 (Step 4 only if the policy is zoned). |
| Augmented DA | `uv run python run_augmented_da.py --config-path configs/custom_configs/augmented_da_2324.yaml` | Steps 0–4 (this config uses local zones). |
| Full pipeline (generate → simulate → analyze) | `bash scripts/run_models_estimates.sh --settings scripts/settings/models_cluster.env` | Steps 0–4 + MNL estimates. |

`--config_path` is also accepted. Use `--help` for the argparse scripts.

---

## Dependency cheat-sheet

| Config family | Cleaned data | Filtered inputs (Step 3) | Zones (Step 4) | MNL estimates |
|---------------|:---:|:---:|:---:|:---:|
| `status_quo_real_*` | ✅ | ✅ | — | — (`utility-model.enable: false`) |
| `augmented_da_2324` | ✅ | ✅ | ✅ (local) | — |
| `all_zones*`, `selected*` (zoned) | ✅ | ✅ | ✅ (local) | depends |
| utility-model configs (`enable: true`) | ✅ | ✅ | depends | ✅ `<SFUSD_CHOICE_PATH>` |

## See also

- **[DATA_SETUP.md](DATA_SETUP.md)** — which config key points to which file, placeholder tokens.
- **[GENERATE_ZONES.md](GENERATE_ZONES.md)** — `.pkl` → CSV zone conversion and registration.
