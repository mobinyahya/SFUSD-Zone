# SFUSD transfer conversion

`convert_sfusd_transfer.py` turns a district pre-run/post-run/demographics
transfer into the cleaned student, program, and school tables that
`loaders/configs/base.yaml` registers per school year. It is not a runtime data
path: run it once when a transfer lands, check the report it writes, then use
the registry year and the matching `sfusd-*` scenario.

```bash
uv run python analysis/data_prep/convert_sfusd_transfer.py \
    --transfer "/soalnas/share/data/school_choice/Data/raw_SFUSD_data_downloads/Sep 14 2026 data transfer" \
    --gaps fill-and-report
```

## What it writes

Per year, below `--out` (default `<data root>/Data/Cleaned`):

| File | Contents |
|---|---|
| `student_<year>.csv` | Every applicant, every grade, one preference round |
| `enrolled_<year>.csv` | The kindergarten subset, matching the existing `enrolled_*` convention |
| `programs_<year>.csv`, `programs_withMissionBay_<year>.csv` | Kindergarten programs observed that year |
| `programs_06_<year>.csv`, `programs_09_<year>.csv` | Grade 6 and 9 programs observed that year |
| `sfusd_transfer_report_<year>.md` / `.json` | Every absent column, blank output, substituted value, and derivation |

No school table is written: the transfer has no school coordinates, so the
registry reuses the 2023-24 school tables.

## What is missing, and how it is resolved

[`TRANSFER_2425_2627_GAPS.md`](TRANSFER_2425_2627_GAPS.md) is the standing
account for the September 2026 transfer: a severity-ranked ledger of every gap,
the per-year missingness counts, the capacity substitutions, and the decisions
that were judgement rather than transcription. Read it before quoting a number
from `sfusd-2425`, `sfusd-2526`, `sfusd-2627` or `sfusd-2425-2627`.

## Missing data

The default `--gaps fail` aborts and names the gap rather than substituting a
value. `--gaps fill-and-report` proceeds and records what it substituted, both
in the report and — for capacities — in a `capacity_source` column on the
affected program rows. Columns the transfer simply does not carry are written
blank, never zero.

Two gaps are structural to these transfers and will reappear with every new
one:

- **No capacity file.** Program existence is derived from requests; capacity
  comes from the 2023-24 district tables, or from the program's observed
  assignment count when 2023-24 has no such program.
- **One preference list per student, of unknown round.** The pre-run has no
  round column and no student has a repeated rank, so only `r1_*` preference
  columns are emitted. But the demographics extract's per-student
  `rounds_applied` field records a later round for a minority of applicants
  (9% of 2024-25 KG applicants, 38% of 2025-26, 27% of 2026-27), and nothing in
  the pre-run separates those rows: `idRequest` spans the same range for tagged
  and untagged students. So the `r1_` label is an assumption. The report's
  `round_participation` section quantifies it per year.

## Tests

```bash
uv run python -m pytest analysis/data_prep loaders/tests/test_transfer_years.py
```

The converter tests run on a synthetic transfer and need no shared data.
