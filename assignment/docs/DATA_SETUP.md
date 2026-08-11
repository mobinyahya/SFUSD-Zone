# Data Setup Guide

This guide explains **which files the simulator needs and which config key
points to each one**, so a new user can clone the repo and run it.

## How paths are resolved

At first run, `Configerator` (`student_assignment/configerator/configerator.py`)
builds your personal config `configs/<username>.config.yaml` (git-ignored) by
merging `configs/base_config.yaml` with an environment-specific *path config*:

| Environment | Path config used | How it's detected |
|-------------|------------------|-------------------|
| Cluster | `configs/cluster_path_config.yaml` | hostname contains `soal` |
| Anywhere else | `configs/local_path_config.yaml` | default |

Both path configs are committed as templates. **You override paths by editing
your generated `configs/<username>.config.yaml`** (preferred) or the path
config template itself.

All paths live under the top-level `paths:` key of the config.

## Placeholder tokens in example / custom configs

The committed configs under `configs/custom_configs/`, `configs/examples/`, and
the analysis configs do **not** hardcode anyone's home directory, and — by
design — **no path is relative to another path**. Every value is one of:

- a **shared cluster path** (`/share/data/school_choice/...`), used verbatim; or
- a **placeholder token** you replace with your own **absolute** path.

There are no CWD-relative (`./`, `../`) values and nothing relies on a file
sitting "under" the `sfusd` root: each file is named by its exact, full path.

| Token | Replace with (absolute path) |
|-------|------------------------------|
| `<STUDENT_ASSIGNMENT_PATH>` | Your `student-assignment` checkout (e.g. `/path/to/student-assignment`). Covers filtered inputs **and** run outputs under `local-data/`. |
| `<SFUSD_CHOICE_PATH>` | Your `SFUSD-Choice` checkout, which holds the MNL `estimates_*.csv`. |
| `<SFUSD_DATA_PATH>` | Your local copy of the confidential SFUSD data tree (root that contains `cleaned/`, `zones/`, …). Only the off-cluster `policy_configs/config_08082025_06.yaml` variant uses it; the `_clusterpaths` twin uses `/share/...` directly. |
| `<RA_SFUSD_PATH>` | Your `RA_SFUSD` checkout (only the permuted-students experiment configs). |

> **Why absolute everywhere?** Several keys (`student-data`, `program-data`,
> `school-data`) are passed through `os.path.join(<sfusd>, value)`. With an
> absolute value the join is a no-op, so the file you name is the file that's
> read — regardless of the `sfusd` setting or the working directory. Output keys
> (`assignment-folder`, `student-save`, `save-path`) are read directly, so they
> are absolute placeholders too rather than CWD-relative `./local-data/...`.

---

## Quick start by environment

### On the Stanford cluster (`soal-*`)

Nothing to set up — the shared data already lives at
`/share/data/school_choice/`, and `cluster_path_config.yaml` points there.

```bash
git clone <repo> && cd student-assignment
uv sync
# auto-creates configs/<user>.config.yaml on first run:
uv run python run_custom_config.py --config-path <config>.yaml
```

### Off the cluster

The confidential SFUSD data is **not** in the repo and cannot be redistributed.
You must copy it from the cluster (or another authorized source), then point
`local_path_config.yaml` (or your `<username>.config.yaml`) at your local copy.
See the per-file table below.

---

## Path reference: which key points to which file

### Required keys (under `paths:`)

| Config key | What it points to | Notes |
|------------|-------------------|-------|
| `sfusd` | **Root folder** of the confidential SFUSD data tree | All relative files below are resolved against this root. Sanity check: it contains a `Data/` subdirectory. |
| `student-save` | Precomputed-data folder (distances, etc.) | Written/read during runs. |
| `assignment-folder` | Folder where assignment CSVs are written | Created if missing. |
| `estimate-path` | MNL choice-model estimates (`.npy` or `estimates_*.csv`) | Required when `utility-model.enable: true`. Produced by the **SFUSD-Choice** repo — only this file is needed, not that repo's code. |

### Files resolved automatically under `sfusd`

When you set `sfusd`, these are found automatically (year `{yy}` = config `year`,
e.g. `18` → `1819`). Defined in `student_assignment/definitions/sfusd_files.py`:

| File (relative to `sfusd`) | Purpose |
|----------------------------|---------|
| `Data/program_codes.csv` | Program code lookup |
| `Data/SF 2010 blks ... .xlsx` | Census block attributes |
| `Data/Cleaned/student_{yy}{yy+1}.csv` | Student records |
| `Data/Cleaned/programs_{yy}{yy+1}.csv` | Program records |
| `Data/Cleaned/schools_rehauled_{yy}{yy+1}.csv` | School records |
| `Data/Precomputed/student_program_distances_{yy}{yy+1}.csv` | Student↔program distances |
| `Data/Student Location Data/out_...cbeds20{yy}.dta` | Student locations (CBEDS) |
| `Census 2010_ Blocks .../*.shp` | Census block shapefile |

### Optional override keys

Set any of these to an **absolute path** to bypass the `sfusd`-relative default
above (used by experiment configs in `configs/custom_configs/`):

| Config key | Overrides | Default if omitted |
|------------|-----------|--------------------|
| `student-data` | student records CSV | `<sfusd>/Data/Cleaned/student_{yy}{yy+1}.csv` |
| `program-data` | programs records CSV | `<sfusd>/Data/Cleaned/programs_{yy}{yy+1}.csv` |
| `school-data` | schools records CSV | `<sfusd>/Data/Cleaned/schools_rehauled_{yy}{yy+1}.csv` |

### Prepared 2023-24 KG round-one inputs

Generate student and program inputs directly from the unmodified source files:

```bash
uv run python -m assignment.scripts.preprocessing.prepare_kg_r1_inputs \
  --output-dir /share/data/school_choice/Data/Cleaned/choice_inputs_2324
```

The command restricts students to KG and a non-empty round-one school list. It
does not apply a distance filter or modify any submitted choices. It writes:

```text
student_2324_kg_r1.csv
student_2324_kg_r1_no_special.csv
programs_2324_kg_r1.csv
programs_2324_kg_r1_no_special.csv
```

For a config that can switch modes without changing paths, use the full student
and program files and set `remove-special-lps: true` to exclude special-program
students and alternatives or `false` to retain them:

```yaml
paths:
  student-data: /share/data/school_choice/Data/Cleaned/choice_inputs_2324/student_2324_kg_r1.csv
  program-data: /share/data/school_choice/Data/Cleaned/choice_inputs_2324/programs_2324_kg_r1.csv
remove-special-lps: true
```

The explicit `_no_special.csv` pair is for consumers that do not apply the
runtime option. Schools that also offer GE or language programs remain present;
only their special-program alternatives are removed. The current MNL estimate
file has no special-program utility columns, so `utility-model.enable: true`
still requires the no-special mode unless new estimates are generated.

### Zone keys (needed only for zone-restricted policies)

| Config key | What it points to | Notes |
|------------|-------------------|-------|
| `zone-files` | Mapping `<name>: <zone CSV>` | Referenced by `policies:` in policy configs. See `docs/GENERATE_ZONES.md` to create new ones. |
| `citywide-or-lp-zones` | Mapping `<name>: <zone .txt>` | Language / special-education / citywide zones. |

### Keys for precomputed-preference mode (rarely needed)

| Config key | What it points to | Required when |
|------------|-------------------|---------------|
| `student-codex` | Student index codex | `utility-model.read-precomuted-umodel-prefs: true` |
| `program-codex` | Program index codex | same |
| `lotteries-path` | Precomputed lottery numbers | using precomputed lotteries |

---

## Minimal off-cluster `paths` block

A working `local_path_config.yaml` (or `<username>.config.yaml`) typically needs:

```yaml
paths:
  sfusd: /path/to/your/SFUSD/                 # contains Data/...
  student-save: /path/to/precomputed/
  assignment-folder: /path/to/assignments/
  estimate-path: /path/to/estimates_2324.csv  # MNL estimates from SFUSD-Choice
  zone-files:                                 # only if running zone policies
    my-zone: /path/to/zones/my_zone.csv
```

If your config relies on `custom_configs/` overrides, also set `student-data`,
`program-data`, and `estimate-path` to your own copies.

---

## Troubleshooting

- **`FileNotFoundError` on a `Data/Cleaned/...` path** → `sfusd` is wrong, or
  you're missing the cleaned data for that `year`.
- **`estimate-path` not found** → an experiment config points at someone else's
  SFUSD-Choice output; repoint it to your own `estimates_*.csv`.
- **Zone key not found** → the name under `policies:` must match a key in
  `zone-files` / `citywide-or-lp-zones`.
- **Wrong path config picked** → detection keys on `soal` in the hostname; off
  the cluster it always uses `local_path_config.yaml`.
