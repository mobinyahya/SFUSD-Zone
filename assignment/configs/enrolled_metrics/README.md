# Enrolled-student metrics

Assignment metrics, local reports, and heatmaps for the kindergarteners who
actually enrolled in 2023-24, 2024-25, 2025-26, and 2026-27. Enrollment is
treated as the outcome. No mechanism or policy runs. Each cell has one
subconfig, `enrolled_match`, which is `real_match` with
`real-match-source: enrollment`. It reads each student's enrolled school and
program and hands that match to the same evaluator every simulated policy uses.

```bash
uv run python -m assignment.slurm submit --config assignment/configs/enrolled_metrics/enrolled_2425.yaml
```

Each cell plans two jobs: one assignment job and `metrics-finalize`.

The three `*_webster_seats` cells rerun 2023-24 through 2025-26 with Webster's
GE capacity raised by Mission Bay's GE seats. They use the built-in capacity
scenarios from the Mission Bay study (`webster_plus_66_ge_2324`,
`webster_plus_69_ge_2425`, `webster_plus_69_ge_2526`; see
`../mb_comparison/README.md`). Enrollment is the outcome, so they hold exactly
the same students at exactly the same schools as their baselines. Only
capacity-dependent outputs move, such as unfilled seats, over-enrollment, and
the heatmaps. 2026-27 has no such cell, because its Mission Bay is real.

## Who is counted

The population is `student_population: enrolled`, and the enrolled table
drops anyone recorded at school 899 or with no enrolled school. See
"Assigned is not enrolled" in `loaders/README.md`.

| Cell | Scenario | Mission Bay | Students evaluated |
|---|---|---|---:|
| `enrolled_2324` | `assignment-generated-zones-2324` | excluded | 3,226 |
| `enrolled_2425` | `assignment-generated-zones-2425` | excluded | 3,809 |
| `enrolled_2526` | `assignment-generated-zones-2526` | excluded | 3,259 |
| `enrolled_2627` | `assignment-generated-zones-2627` | included (real that year) | 3,837 |

The `*_webster_seats` cells count the same students as their baselines.

The counts are below the enrolled tables' 3,510 / 3,830 / 3,515 / 3,977 for
three reasons. First, `outside_district_students: ignore` drops students with no
Census Block. That is 254 in 2025-26, whose post-run lacks coordinates for 197
students. Second, the tables' own blank-school and no-list students are
dropped, which is a handful per year.

Third, every cell uses round-1 preferences, and a student with no round-1 list
cannot enter the match. In 2023-24, 283 enrolled students applied only in
later rounds and are excluded for that reason. `rounds: all` would keep them,
scored against their first participating round's list. The transfer years
carry only a round-1 list.

## Reading the numbers

- **"Designated"** students are included. The term here means a student who
  enrolled at a school absent from their ranked list: 138 / 282 / 155 / 212
  by year. Rank is scored against the first participating round's list,
  and such a student gets `list length + 1`, exactly as `real_match` scores
  them.
- **Programs.** The transfer years record the enrolled pathway. For 2023-24
  the program is inferred: first the round whose outcome is the enrolled
  school (3,113), then the first program ranked at that school (163), then
  GE (96).
- **Unassigned** students (49 / 2 / 12 / 23) enrolled at a school with no
  kindergarten program in the program table, mostly 809 and 517. Each run
  warns with the list.
- The policy knobs in `enrolled_match.yaml` (priorities, ties, CTIP) are
  copied from `real_match.yaml`. Nothing reads them for this match.
