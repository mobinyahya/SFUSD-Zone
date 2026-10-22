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
| `student_<year>.csv` | Every applicant, every grade, one preference round. At kindergarten that includes the students SFUSD promotes in from TK, who filed no request; `mr_applicant` says which rows did |
| `enrolled_<year>.csv` | The kindergarten subset of `student_<year>.csv` the post-run seats |
| `programs_<year>.csv`, `programs_withMissionBay_<year>.csv` | Every kindergarten program in that year's Main Round capacity file, at gross `TotalSeats` |
| `programs_06_<year>.csv`, `programs_09_<year>.csv` | Grade 6 and 9 programs observed that year, at borrowed capacity |
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

Three gaps are structural to these transfers and will reappear with every
new one:

- **No capacity file for grades 6 and 9.** The auxiliary capacity file covers
  every grade, but only kindergarten has been reconciled against the district's
  own promotion counts, so grades 6 and 9 still derive program existence from
  requests and capacity from the 2022-23 district tables, or from the program's
  observed assignment count when those tables have no such program.
- **One preference list per student: the main round.** The pre-run has no round
  column and no student has a repeated rank, so only `r1_*` preference columns
  are emitted. The district confirmed the extract is the main-round request
  file, and the post-run corroborates it. What is missing is the later-round
  requests: the demographics extract's per-student `rounds_applied` field
  records a later round for a minority of applicants (9% of 2024-25 KG
  applicants, 38% of 2025-26, 27% of 2026-27) with no request rows to match,
  and the field is too incomplete to identify the main round in the other
  direction either. The report's `round_participation` section quantifies it
  per year.
- **Auto-promotion from 2024-25.** The main run seats TK students in
  kindergarten before any applicant is placed, and they filed no request, so
  they appear only in the post-run (40 in 2024-25, 580 in 2025-26, 847 in
  2026-27). They hold a claim on a seat all the same, so the kindergarten
  applicant pool includes them: they are added to `student_<year>.csv` with
  `mr_applicant = 0`, and `enrolled_<year>.csv` is the subset of that pool the
  run seated. Their preference lists are not in the transfer either, and are
  reconstructed from the TK requests they filed the year before, or from their
  feeder and attendance-area programs; `tk_promotion.py` holds the rules and
  [`TK_PROMOTION_SPEC.md`](TK_PROMOTION_SPEC.md) the specification. Capacity is
  never netted against those seats: the run releases a promote's held seat back
  into the same market when they win elsewhere. The claim itself rides in the
  `promote` column and is **not yet a priority** — see
  [`assignment/README.md`](../../assignment/README.md#tk-to-k-auto-promotion-and-what-is-still-to-be-built)
  for what remains to be built.

## Tests

```bash
uv run python -m pytest analysis/data_prep loaders/tests/test_transfer_years.py
```

`test_convert_sfusd_transfer.py` runs on a synthetic transfer and needs no
shared data: it pins the rules. `test_tk_promotion_real.py` pins the numbers
those rules were reconciled against — the program counts, the market size, and
the promotion counts program by program — so it reads the real transfer and
carries the `real_data` marker. Deselect it with `-m "not real_data"` on a
machine without the shared data root.
