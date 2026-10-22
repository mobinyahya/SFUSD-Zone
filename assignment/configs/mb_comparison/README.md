# Mission Bay and Webster capacity comparison study

The active study has eight assignment cells. All use the same ten policies,
seed 2023, 25 iterations, real preferences, and the high-income and FRL metric
reports. The historical `mb_2324.yaml` cell is excluded from the study and run
plan. Its Mission Bay seats came from a different, hypothetical 2023-24 program
roster, so it is not a clean counterpart to the no-Mission-Bay baseline.

## 2023-24 through 2025-26: no Mission Bay

These three baseline cells use each year's real program table, with Mission Bay
excluded. Their school table is `schools_rehauled_2324.csv` (72 schools).

| Baseline config | Program source | Programs | Students | Total capacity |
|---|---|---:|---:|---:|
| `nomb_2324.yaml` | `programs_2324.csv` | 164 | 3,953 | 4,268 |
| `nomb_2425.yaml` | `programs_2425.csv` | 166 | 3,852 | 4,099 |
| `nomb_2526.yaml` | `programs_2526.csv` | 166 | 3,717 | 4,202 |

Each has a separate Webster capacity sensitivity cell:

| Sensitivity config | Capacity scenario | Webster GE seats added | New total capacity |
|---|---|---:|---:|
| `nomb_2324_webster_seats.yaml` | `webster_plus_66_ge_2324` | 66 | 4,334 |
| `nomb_2425_webster_seats.yaml` | `webster_plus_69_ge_2425` | 69 | 4,168 |
| `nomb_2526_webster_seats.yaml` | `webster_plus_69_ge_2526` | 69 | 4,271 |

The 66 seats match the hypothetical 2023-24 Mission Bay GE count; the 69 seats
in each later year match Mission Bay's actual 2026-27 GE count. **These are
capacity additions, not transfers from a program in those years.** The three
baseline tables have no Mission Bay seats to remove. Only Webster GE capacity
changes within each baseline/sensitivity pair; student data, preferences,
schools, and policies stay the same. The original source tables are unchanged.

## 2026-27: real Mission Bay and seat transfer

`mb_2627.yaml` uses `programs_withMissionBay_2627.csv` (162 programs, 3,855
students, 4,306 seats). Mission Bay has 69 GE, five MM, and zero AF seats. Its
school table is `schools_rehauled_withMissionBay_2324.csv` (73 schools).

`mb_2627_webster_seats.yaml` uses the same inputs with
`capacity_scenario: mission_bay_ge_to_webster_2627`. The loader moves Mission
Bay's 69 GE kindergarten seats to Daniel Webster GE: Mission Bay GE becomes
zero, Webster GE goes from 27 to 96, Mission Bay MM remains five, and total
capacity stays 4,306. This is an actual transfer between programs in the
selected 2026-27 table. The district source table is unchanged.

## Policies and metrics

Each cell includes `no_distance+reserves_05frl_#3` and `#4` alongside
`distance_05_1_2+reserves_05frl_#3` and `#4`. Within each pair, the sole
policy change is removal of the distance priority weight and thresholds. Both
retain the 50/50 FRL soft reserves, zone access, attendance-area rules, and
lottery. The existing `status_quo+reserves_05frl` policy also has no distance
priority but changes other rules, so it is not the isolated comparison.

All eight configs set `export-aggregate-metrics: true` and
`export-local-metrics: true`. The metrics pass recomputes the high-income and
FRL citywide measures, plus program, ZIP, and attendance-area reports from
both new and reused assignment CSVs. See
`assignment/docs/ASSIGNMENT_METRICS.md` for metric definitions.

## Running on SOAL

Sync the current study code and configs to the cluster, then run `uv sync`.
Submit the following eight configs **one at a time**, waiting for the previous
cell's metrics finalizer to finish successfully before submitting the next:

1. `nomb_2324.yaml`
2. `nomb_2324_webster_seats.yaml`
3. `nomb_2425.yaml`
4. `nomb_2425_webster_seats.yaml`
5. `nomb_2526.yaml`
6. `nomb_2526_webster_seats.yaml`
7. `mb_2627.yaml`
8. `mb_2627_webster_seats.yaml`

For each file, use the Slurm launcher from the repository root, for example:

```bash
uv run python -m assignment.slurm submit --config assignment/configs/mb_comparison/nomb_2324.yaml
```

Each ten-policy cell plans up to 18 Slurm jobs (ten assignment jobs and eight
dependent metrics jobs). The cluster's `soal` QoS is `MaxJobsPU=12,
MaxSubmitPU=20`, so concurrent cell submissions can exceed the limit.

The launcher defaults to `--skip-existing`, and all configs use
`reuse_assignments: true`. If the earlier eight-policy baseline runs are
complete and correspond to these configs, the three no-Mission-Bay baselines
and `mb_2627` each need only the two new no-distance policies (200 new
assignment iterations total). All four Webster sensitivity cells are new (1,000
assignment iterations total). Metrics run across all ten policies in all eight
cells (2,000 policy iterations), including reused assignments. The excluded
`mb_2324.yaml` is neither submitted nor included in these counts.

Check each final `assignments/aggregate_metrics/manifest.json`, the four
aggregate CSV reports, and 25 saved assignment iterations per policy. The
2023-24 through 2025-26 Webster program rows should gain 66, 69, and 69 GE
seats respectively; the 2026-27 report should show Mission Bay GE at zero and
Webster GE at 96. Outputs live under
`/soalnas/share/data/school_choice/local_runs/mb_comparison/<cell>/`, one
directory per cell.
