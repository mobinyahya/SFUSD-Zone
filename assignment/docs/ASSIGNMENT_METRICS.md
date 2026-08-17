# Assignment Metrics

`student_assignment.evaluation.match_evaluator.MatchEvaluator` is the single
assignment evaluator. It has two reports:

- `eval_assignment_basic()` preserves the compact metric contract consumed by
  `choice/assignment_metrics.py` and the benchmark suite.
- `eval_assignment_full()` produces the standalone report used by
  `scripts/analysis/analyze_trends.py` and the runner's optional aggregate
  exports. It does not require a zoning solution.

The trend-analysis YAML continues to accept either `run_csv` or `folder` for
each run. Folder discovery is recursive and only reads CSVs containing the
assignment columns `studentno`, `programno`, `programcodes`, and `rank`.

## Population Definitions

- Assigned means `programno > 0`.
- Designated means an assigned row with `designation == 1` unless a metric name
  explicitly refers to all first-round students.
- Non-designated assignment outcomes use assigned rows with
  `designation == 0`.
- AALPI is Black, Hispanic, or Pacific Islander after ethnicity normalization.
- High FRL means `freelunch_prob + reducedlunch_prob > 0.5`; Low FRL means the
  sum is at most `0.5`.
- CTIP uses `ctip1`. ET (2024) uses membership in the optional equity-block
  `.npy`; without that file every student is non-ET.
- `first_round: true` filters to students with a non-null
  `r1_ranked_idschool`. `no_special_program: true` excludes students whose
  first-round program list intersects `SPECIAL_PROGRAMS`.

Empty cohort proportions are `NaN` in the evaluator. `analyze_trends.py`
retains its historical behavior of replacing those values with zero before
aggregating saved runs.

## Distance And Choice

The full report calculates straight-line Haversine miles from student latitude
and longitude to the assigned school's coordinates. Distance and choice
proportions use assigned students as their denominator unless the name says
`Non-Designated` or `All Students`.

`In-Zone Rank` is the rank stored by the assignment process. It may reflect
broader eligibility rules such as siblings, citywide access, or CTIP and must
not be interpreted as proof that the assigned school is geographically in the
student's zone. Historical assignment files may also set it equal to ordinary
rank. If the column is absent, the evaluator uses ordinary rank. No separate,
reliable in-zone assignment indicator exists in these files.

The `Proportion of students in top N (All Students)` family uses the complete
filtered first-round cohort. Missing ranks, including unassigned students,
compare false.

## FRL And Income Concentration

School FRL is the mean student FRL probability among assigned students at that
school. District FRL is the corresponding mean across assigned students.
`+10%`, `+15%`, `-10%`, and `-15%` are absolute percentage-point differences
from that district mean. Non-designated school metrics recalculate school
composition from non-designated assigned students while retaining the assigned
district mean as the reference.

The High Income (95292) metrics do not use a school's mean household income.
They preserve the migrated methodology: each school's share of students with
household income at or above the threshold is compared with the district share
plus or minus 10 or 15 percentage points. Low Income metrics analogously use
the share at or below the threshold.

`Prop AALPI in ... Schools` is the pooled AALPI share among students in the
selected schools. `Avg Prop AALPI ...` and `Avg Prop FRL ...` are unweighted
means of the selected schools' composition proportions.

## Segregation And Exposure

The full dissimilarity family preserves the existing one-sided formula:

```text
(1 / (2 * group_total)) *
sum_school(abs(group_school - school_total * group_total / total_students))
```

This is not the classic two-group dissimilarity index. The basic report also
preserves the benchmark's historical implementation, including its compact
school iteration, so existing benchmark values do not change.

Absolute exposure of group `G` to group/property `P` is:

```text
sum_school((G_school / G_total) * (P_school / school_total))
```

`Black/White exposure to ...` and `Hispanic/White exposure to ...` are signed
differences between the two groups' exposures, not absolute exposure levels.
`exposure to FRL prob` instead averages each attended school's mean FRL
probability over students in the named group.

## Programs And Utilization

GE isolation metrics count GE program IDs from the program file, including GE
programs with zero Black or Pacific Islander assignments. Program assignment
counts are student counts by assigned program type and designation status.

`utilization_<type>` is total assigned enrollment divided by total capacity for
that program type. `utilization_rate_avg` preserves the migrated unweighted
mean of individual program utilization rates; it is not district-wide filled
seats divided by district-wide capacity.

When `export-aggregate-metrics: true`, an assignment run writes four combined
reports under `paths.assignment-folder/aggregate_metrics/`:

- `metrics_by_school.csv` has one row per school. Enrolled means all simulated
  placements (`programno > 0`), assigned means a non-designated placement, and
  designated means `designation == 1`. School utilization is simulated
  enrollment divided by the sum of the school's selected program capacities.
- `metrics_by_zip_code.csv` has one row per non-missing student `zipcode`.
- `metrics_by_attendance_area.csv` has one row per non-missing student
  `idschoolattendance`.
- `metrics_citywide.csv` has one row per config and the complete citywide metric
  set as columns.

The two residential-geography reports contain the complete full metric set,
recomputed from the students residing in each geography. Program inventory and
capacity diagnostics retain the evaluator's selected district program table.
Every report has `config_name`, which identifies the subconfig and full policy
variant but excludes the iteration suffix. Numeric metrics are averaged across
iterations for each config and school, ZIP code, attendance area, or citywide
row. No per-assignment metric CSVs are written. The option defaults to `false`.

The basic metric `Programs with 1-4 AA` is a legacy benchmark field whose
historical calculation counts schools with one to four Black students. The
full report's explicitly named GE-program metric is the correctly scoped count
of Black or Pacific Islander students by GE program.
