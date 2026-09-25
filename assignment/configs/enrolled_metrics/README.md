# Enrolled-student metrics

Assignment metrics, local reports, and heatmaps for the kindergarteners who
actually enrolled in 2023-24, 2024-25, 2025-26, and 2026-27. Every cell runs
two subconfigs on the same enrolled students:

- **`enrolled_match`** treats enrollment as the outcome, and no mechanism
  runs. It is `real_match` with `real-match-source: enrollment`: it reads
  each student's enrolled school and program and hands that match to the same
  evaluator every simulated policy uses. It produces one iteration.
- **`neighborhood_assignment_#5`** is policy 5 (every student to their
  attendance-area GE program), run for iterations 0-24 with seed 2023 as in
  the Mission Bay study.

```bash
uv run python -m assignment.slurm submit --config assignment/configs/enrolled_metrics/enrolled_2425.yaml
```

Each cell plans four jobs: two assignment jobs, one metrics job, and
`metrics-finalize`. The cluster's `MaxSubmitPU=20` fits four cells at a
time, not all seven.

The three `*_webster_seats` cells rerun 2023-24 through 2025-26 with Webster's
GE capacity raised by Mission Bay's GE seats. They use the built-in capacity
scenarios from the Mission Bay study (`webster_plus_66_ge_2324`,
`webster_plus_69_ge_2425`, `webster_plus_69_ge_2526`; see
`../mb_comparison/README.md`). Under `enrolled_match` they hold exactly the
same students at the same schools as their baselines, so only
capacity-dependent outputs move. Under policy 5 the extra seats are live
capacity. 2026-27 has no such cell, because its Mission Bay is real.

## Who is counted

The population is `student_population: enrolled`. For the transfer years
that is the students the post-run seated **and** the district's fall
enrolment record puts at a kindergarten school. A record at 899 ("Central
Enrollment"), no record at all, or a record at another grade means the
student did not enroll. For 2023-24 it is the KG applicants with an enrolled
school. See "Assigned is not enrolled" in `loaders/README.md`.

| Cell | Scenario | Mission Bay | Enrolled | Evaluated |
|---|---|---|---:|---:|
| `enrolled_2324` | `assignment-generated-zones-2324` | excluded | 3,510 | 3,226 |
| `enrolled_2425` | `assignment-generated-zones-2425` | excluded | 3,149 | ≤ 3,136 |
| `enrolled_2526` | `assignment-generated-zones-2526` | excluded | 3,278 | ≤ 3,050 |
| `enrolled_2627` | `assignment-generated-zones-2627` | included (real that year) | 3,233 | 3,117 |

The 2425 and 2526 upper bounds only account for missing Census Blocks.

The `*_webster_seats` cells count the same students as their baselines.

Evaluated is below enrolled for two reasons. First,
`outside_district_students: ignore` drops students with no Census Block:
13 / 228 / 116 in the transfer years, most of them without coordinates in the
post-run. Second, every cell uses round-1 preferences, and a student with no
round-1 list cannot enter the match. In 2023-24, 283 enrolled students applied
only in later rounds and are excluded for that reason. `rounds: all` would
keep them, scored against their first participating round's list. The
transfer years carry only a round-1 list.

## Reading the numbers

- **"Designated"** students are included. Under `enrolled_match` the term
  means a student who enrolled at a school absent from their ranked list.
  Rank is scored against the first participating round's list, and such a
  student gets `list length + 1`, exactly as `real_match` scores them.
- **Programs.** The transfer years record the enrolled pathway. For 2023-24
  the program is inferred: first the round whose outcome is the enrolled
  school, then the first program ranked at that school, then GE. Each run
  warns with the counts.
- **Unassigned** under `enrolled_match` means the student enrolled at a
  school with no kindergarten program in the program table, mostly 809 and
  517. Each run warns with the list.
- The policy knobs in `enrolled_match.yaml` (priorities, ties, CTIP) are
  copied from `real_match.yaml`. Nothing reads them for this match.
- Policy 5's inner signature directory
  (`aa_noRestrict_softReserve_ctip1_..._0b67c6a46480`) is shared with other
  policies in other studies. Group by the outer `config_name`.
