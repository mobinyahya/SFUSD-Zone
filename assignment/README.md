# Student Assignment Simulator

Simulation tools for the **San Francisco Unified School District (SFUSD)**
school-choice system. The simulator runs Deferred Acceptance (and variants)
under different zone, priority, and tie-breaking policies, then compares a
**Status Quo** baseline against optimized assignments on choice attainment
(top-k), travel distance, and equity (racial / socioeconomic composition).

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

---

## Requirements

- **Python 3.10+** (pinned to 3.10 via [`.python-version`](.python-version);
  `uv` installs a matching interpreter for you).
- **[uv](https://docs.astral.sh/uv/)** for dependency management.

## Installing uv

`uv` is a single static binary — no root required, works on the cluster too.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via pip / pipx / Homebrew
pip install uv     #  •  pipx install uv  •  brew install uv
```

The installer adds `~/.local/bin` to your `PATH`; restart your shell (or
`source ~/.bashrc`) if `uv` is not found.

## Installation

From a fresh clone, **one command** provisions the exact pinned environment
(Python interpreter included) from the tracked `pyproject.toml` + `uv.lock`:

```bash
git clone <repo> && cd student-assignment
uv sync
```

This creates a `.venv/` with every locked dependency. Run project commands with
`uv run <cmd>` (no activation needed), or `source .venv/bin/activate` for a
classic shell.

## Quick start

```bash
# Run the test suite — including the end-to-end pipeline on the committed
# fake dataset. Works on a bare clone, with no confidential data.
uv run python -m pytest tests -q

# Prove the checkout is self-sufficient (tracked files only, fresh env):
bash scripts/test_clean_checkout.sh
```

> A bare clone runs the tests and the **fake-data** pipeline out of the box.
> **Real** simulations need the confidential SFUSD data and the filtered
> `local-data/` inputs (not committed) — see
> **[docs/PIPELINE.md](docs/PIPELINE.md)** for the clone-to-first-run guide.

```bash
# Single simulation from one YAML config (needs real data; see Data setup)
uv run python run_custom_config.py \
    --config-path configs/custom_configs/status_quo_real_2324.yaml

# DA with preference-list augmentation (alternative programs)
uv run python run_augmented_da.py \
    --config-path configs/custom_configs/augmented_da_2324.yaml
```

## Entry points

Everything that drives a simulation or analysis is **config-file driven**;
preprocessing utilities take plain CLI flags. Prefix each with `uv run`.

| Script | Invocation | Purpose |
|---|---|---|
| `run_custom_config.py` | `--config-path <yaml>` | Run one simulation from a YAML config. |
| `run_augmented_da.py` | `--config-path <yaml>` | Run DA with preference-list augmentation. |
| `scripts/run_models_estimates.sh` | `--settings <env>` | **Full pipeline**: generate run YAMLs → simulate → analyze → `metrics_comparison.xlsx`. |
| `scripts/analysis/analyze_trends.py` | `--config <yaml>` | Aggregate runs into a metrics workbook + plots. |
| `scripts/analysis/plot_simulation_frontier.py` | `--config <yaml>` | Score a folder of simulations and plot their **Pareto frontier** (e.g. distance vs. dissimilarity). |
| `filter_student_choices.py` | (Hydra) | Filter students by choice-model confidence (`configs/custom_configs/distance_filter.yaml`; override `key=value` on the CLI). |
| `create_simulator_input.py` | CLI flags | Build simulator input tables. |
| `recompute_lottery_number.py` | `--students --schools --output` | Recompute tie-breaker lotteries. |
| `scripts/preprocessing/filter_programs.py` | `--data-dir --output-dir` | Drop special programs from program CSVs. |
| `scripts/preprocessing/prepare_kg_r1_inputs.py` | `--students --programs --output-dir` | Build paired KG round-one inputs with and without special programs. |
| `scripts/generators/generate_zone_from_pickle.py` | CLI flags | Build a zone CSV from a pickled plan. |
| `scripts/generators/generate_fake_dataset.py` | `--out-dir --num-students --seed` | Regenerate the committed fake test dataset. |

Use `--help` on any script for its options. Full config reference:
**[docs/CONFIG_OPTIONS.md](docs/CONFIG_OPTIONS.md)**.

## Repository layout

```
student_assignment/         Core library (installed as a package by uv sync)
  da/                       Deferred-acceptance variants (vanilla, guardrails, quotas)
  market_generator/         Preference/utility generation, list augmentation
  data_interfaces/          Student, program, zone loaders
  evaluation/               Match evaluation and metrics
  configerator/             Layered config loading + schema validation
  utils/plotting.py         Shared matplotlib/seaborn styling

scripts/
  run_models_estimates.sh   End-to-end pipeline (generate → simulate → analyze)
  settings/                 Pipeline settings files (cluster, test)
  analysis/analyze_trends.py    Aggregate runs into metrics_comparison.xlsx
  preprocessing/            Data filtering and extraction
  generators/               Zone + fake-dataset generators
  test_clean_checkout.sh    Verify the repo runs from tracked files only

configs/
  base_config.yaml          Simulation defaults
  local_path_config.yaml    Off-cluster paths    cluster_path_config.yaml  Cluster paths
  config_schema.yaml        yamale schema for the user config
  custom_configs/           Representative run configs (status quo, augmented, …)
  policy_configs/           Policy definitions (zones, distance bands, reserves)
  examples/                 One canonical sample per generated config family

tests/                      pytest suite (incl. end-to-end test_full_pipeline.py)
tests/fixtures/fake_2223/   Committed fake dataset the pipeline test runs on
docs/                       CONFIG_OPTIONS.md, PIPELINE.md, DATA_SETUP.md, Sphinx sources
```

## Configuration

Config is resolved in layers (each overrides the previous): `base_config.yaml`
→ environment path config → auto-created `configs/<user>.config.yaml` → custom
run YAML → policy subconfig. The first time you run any entry point, the
`Configerator` writes your personal `configs/<user>.config.yaml` automatically.

Every option (top-level keys, `paths.*`, `utility-model.*`, policy subconfigs,
list-augmentation, the analysis config, and pipeline settings files) is
documented in **[docs/CONFIG_OPTIONS.md](docs/CONFIG_OPTIONS.md)**.

## Data setup

Paths resolve automatically by environment: on the cluster (hostname contains
`soal`) from `configs/cluster_path_config.yaml` (data already at
`/share/data/school_choice/`); elsewhere from `configs/local_path_config.yaml`,
which you point at your own copy of the confidential SFUSD data.

Paths in the example configs are **explicit** — either a shared cluster path
(`/share/data/school_choice/...`, used as-is) or a placeholder token you replace
with your own **absolute** path:

| Token | Replace with (absolute path) |
|-------|------------------------------|
| `<STUDENT_ASSIGNMENT_PATH>` | your `student-assignment` checkout (inputs and run outputs under `local-data/`) |
| `<SFUSD_CHOICE_PATH>` | your `SFUSD-Choice` checkout (MNL `estimates_*.csv`) |
| `<SFUSD_DATA_PATH>` | your local copy of the confidential SFUSD data tree |
| `<RA_SFUSD_PATH>` | your `RA_SFUSD` checkout (permuted-students experiment configs) |

Apply them quickly, e.g. `sed -i "s#<STUDENT_ASSIGNMENT_PATH>#$PWD#g" configs/<your-config>.yaml`.
Full guide (which key points to which file, what runs out-of-the-box):
**[docs/DATA_SETUP.md](docs/DATA_SETUP.md)**.

## Development

```bash
make install         # uv sync
make test            # uv run pytest tests -q
make lint            # uv run ruff check .
make format          # uv run ruff format . && uv run ruff check --fix .
make clean-checkout  # bash scripts/test_clean_checkout.sh
```

- **Lint + format:** [Ruff](https://docs.astral.sh/ruff/) (line length 80;
  `ruff format` for layout, `ruff check` for lint). Config in `pyproject.toml`.
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html).
- **Dependencies:** `uv add <pkg>` / `uv remove <pkg>` (edits `pyproject.toml`
  and re-locks `uv.lock`); commit both.

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/CONFIG_OPTIONS.md](docs/CONFIG_OPTIONS.md) | Every config key, layer by layer |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Clone → first real simulation runbook |
| [docs/DATA_SETUP.md](docs/DATA_SETUP.md) | Data files, path config, placeholder tokens |
| [docs/GENERATE_ZONES.md](docs/GENERATE_ZONES.md) | Building zone files |

Project onboarding documentation is maintained internally by the SFUSD
research group and is available to collaborators on request.

This work builds on the original `sfusd-project` codebase by Kaleigh Mentzer
and collaborators.
