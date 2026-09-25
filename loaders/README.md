# Loader Configuration Reference

This document is the complete field reference for YAML consumed by the shared
`loaders` package. The executable configuration, base catalog, and scenario
files are separate schemas:

| Layer | Location | Purpose |
|---|---|---|
| Run | The `data` block in an optimization, benchmark, or assignment YAML | Select a scenario and provide partial overrides |
| Base | `loaders/configs/base.yaml` | Declare roots, catalog files, geography bundles, FRL estimates, and school-year source bundles |
| Scenario | `loaders/configs/scenarios/*.yaml` | Declare invariant source roles and complete selector defaults |

`loaders/config.py` is the authoritative validator. Unknown fields are rejected
unless this reference explicitly describes a map as open-ended.

## Run Configuration

Executables place the loader configuration under `data`:

```yaml
data:
  scenario: legacy
  overrides:
    roots: {}
    sources: {}
    filters: {}
```

The `data` wrapper belongs to the executable. The map passed to
`load_scenario()` contains exactly these fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `scenario` | String | Yes | Bundled scenario name or path to a scenario YAML. A relative custom path is resolved from the run YAML. |
| `overrides` | Map | Yes | Partial run-specific overrides. May be empty. |
| `overrides.roots` | Map of root name to path | No | Replaces base root paths for this run. Relative paths are resolved from the run YAML. |
| `overrides.sources` | Map of dotted role to source reference | No | Replaces or structurally patches exceptional sources. Final merged sources must be complete. |
| `overrides.filters` | Map | No | Partial `optimization` and/or `assignment` selector overrides. |

`roots`, `sources`, and `filters` default to empty maps when omitted. The
`overrides` field itself is still required. Bundled names resolve to
`loaders/configs/scenarios/<name>.yaml`; otherwise `scenario` is interpreted as
a path.

Filter maps merge recursively. Lists and scalar values replace scenario values
rather than being appended. A source override has highest source precedence and
is intended for an exceptional experimental input, not normal year or grade
selection.

## Source References

Entries in scenario `sources`, run `overrides.sources`, and geography
`manual_edges` can use any of these forms:

```yaml
# Catalog ID from base.yaml
optimization.programs: assignment.programs.2324

# Absolute scalar path. A relative scalar is treated as a catalog ID.
choice.estimate: /absolute/path/to/estimates.csv

# Direct source object
assignment.estimate:
  path: simulation-files/choice-model/estimates.csv
  root: data
  companions:
    - simulation-files/choice-model/estimates.metadata.json
  classification: restricted
  geography_vintage: "2010"

# Ordered source collection
optimization.students:
  - optimization.students.enrolled.2122
  - optimization.students.enrolled.2223

# Named source map
assignment.zones:
  Medium1: zones.common.medium1
  Large1: zones.common.large1
```

Lists and named maps may contain any other source-reference form recursively.
`DataScenario.sources(role)` flattens them in configured order, while
`DataScenario.source_map(role)` preserves a top-level named map.

Named-map keys must be non-empty strings. The five direct-source field names
(`path`, `root`, `companions`, `classification`, and `geography_vintage`) are
reserved: any map containing one of them is parsed as a direct source object,
not a named map. Empty lists and maps pass configuration validation but resolve
to no files and are not usable by consumers that require one or more sources.

### Direct Source Fields

| Field | Type | Required | Default | Description |
|---|---|---:|---|---|
| `path` | Path string | Yes | None | File path. It may be omitted only in a partial run patch; the final merged source must contain it. |
| `root` | String | No | Declaring YAML directory | Named base root or built-in `package` or `repository` root. |
| `companions` | List of path strings | No | `[]` | Files that participate in source identity with `path`. Paths use the same root as `path`. |
| `classification` | String | No | `unspecified` | Data-classification metadata included in manifests and cache identity. Values such as `public`, `internal`, and `restricted` are conventions, not a validated enum. |
| `geography_vintage` | Four-digit string | No | None | Census vintage already represented by a location-bearing source, such as `"2010"`. |

For a Shapefile `path`, `.dbf`, `.shx`, and `.prj` files are added to
`companions` automatically. Rootless direct paths are relative to the YAML that
declares them. Catalog paths are relative to `base.yaml` unless their direct
source object names a root. A scalar string must be a catalog ID unless it is an
absolute path.

One base-registry exception requires an explicit path base: a direct source
object placed at `geographies.<vintage>.manual_edges` is generated into a
scenario role before resolution and is not anchored to `base.yaml`. Use a
catalog ID, an absolute path, or an explicit `root` for that field.

Source `geography_vintage` is different from the filter field with the same
name. The source field describes the input table; the filter selects the target
geography. Same-vintage Census IDs are retained. A different source vintage is
remapped from coordinates when the shared table reader supports that role. No
conversion is requested when every source in a role lacks vintage metadata.
Vintage is evaluated for the entire flattened role: different non-null vintages
within one role fail, while one non-null vintage applies to the concatenated
table even if some sources in that role are untagged.

## Base Catalog

`loaders/configs/base.yaml` has this top-level shape:

```yaml
schema_version: 2
roots: {}
files: {}
geographies: {}
student_frl_estimates: {}
school_years: {}
```

All six fields are required.

| Field | Type | Description |
|---|---|---|
| `schema_version` | Integer | Must be exactly `2`. |
| `roots` | Map | Named path roots available to direct source objects. |
| `files` | Map | Catalog ID to complete direct source object. |
| `geographies` | Non-empty map | Census-vintage geography source bundles. |
| `student_frl_estimates` | Non-empty map | Selectable student FRL count sources. |
| `school_years` | Non-empty map | Annual optimization and assignment source registry. |

### Roots

`roots` is open-ended, but it must define non-null `data` and `cache` paths.
Additional roots may be paths or `null`; a null root must be supplied before a
source can use it. The names `package` and `repository` are reserved built-ins
and cannot be declared or overridden.

| Root | Current default | Description |
|---|---|---|
| `data` | `/soalnas/share/data/school_choice` | External source files. |
| `cache` | `/soalnas/share/data/school_choice/Data/caches` | Content-addressed derived artifacts. |
| `package` | `loaders/configs/` | Built-in root for packaged configuration files. |
| `repository` | Repository root | Built-in root for checked-in files outside `loaders/configs/`. |

Root precedence, from lowest to highest, is base `roots`,
`SFUSD_DATA_ROOT`/`SFUSD_CACHE_ROOT`, then run `overrides.roots`. Run overrides
may name only roots declared by the base and cannot replace a built-in root.
Relative base roots are resolved from the directory containing `base.yaml`;
relative environment-variable roots are resolved from the repository root; and
relative run overrides are resolved from the run YAML.

### Files

Each key below `files` is an arbitrary non-empty catalog ID. Each value must be
a complete direct source object using only `path`, `root`, `companions`,
`classification`, and `geography_vintage`. A catalog entry cannot alias another
catalog ID or contain a list or named map.

Source files do not have to exist while configuration is resolved. Missing
files are recorded as `missing` in source manifests and fail only when a
consumer requires them.

### Geography Bundles

Each `geographies` key is a four-digit Census vintage:

```yaml
geographies:
  "2020":
    blocks: census.blocks.2020
    blockgroups: census.blockgroups.2020
    tracts: census.tracts.2020
    crosswalk: census.block_crosswalk.2020
    adjacency:
      block: census.adjacency.block.2020
      blockgroup: census.adjacency.blockgroup.2020
      tract: census.adjacency.tract.2020
    manual_edges: bundled.manual_edges.2020
```

| Field | Type | Required | Description |
|---|---|---:|---|
| `<vintage>` | Four-digit string key | Yes | Available value for filter `geography_vintage`. |
| `blocks` | Catalog ID | Yes | Census Block geometry. |
| `blockgroups` | Catalog ID | No | Direct Census Block Group geometry. It may instead be derived from Blocks and the crosswalk. |
| `tracts` | Catalog ID | No | Direct Census Tract geometry. It may instead be derived from Blocks and the crosswalk. |
| `crosswalk` | Catalog ID | Yes | Mapping from Block to parent Block Group and Tract IDs. |
| `adjacency` | Map | Yes | Adjacency catalog IDs for all supported geography units. |
| `adjacency.block` | Catalog ID | Yes | Block adjacency. |
| `adjacency.blockgroup` | Catalog ID | Yes | Block Group adjacency. |
| `adjacency.tract` | Catalog ID | Yes | Tract adjacency. |
| `manual_edges` | Source reference | Yes | One or more manual Block-edge YAML files. May be a catalog ID, direct object, list, or named map. |

All scalar source leaves except `manual_edges` reference catalog IDs from
`files`. `adjacency` must contain exactly `block`, `blockgroup`, and `tract`.
The checked-in catalog currently provides `"2010"` and `"2020"` bundles.

### Student FRL Estimates

`student_frl_estimates` maps selector names to catalog IDs:

```yaml
student_frl_estimates:
  updated_2526: student_frl.updated_2526
```

Each source must contain `BlockID`, `Not FRL`, `FRLunch`, and `Students`.
Counts must be non-negative integers and `Students` must equal `Not FRL +
FRLunch`. The loader derives the exact student probability as `FRLunch /
Students`; zero-student blocks and blocks absent from the estimate retain the
student file's FRL value. A selected source's `geography_vintage` metadata must
match the filter's `geography_vintage`. The registered `updated_2526` source
therefore requires `geography_vintage: "2020"`.

For an exceptional input, retain the registered selector name and replace the
consumer-specific role through the normal source override mechanism:

```yaml
overrides:
  sources:
    optimization.frl_estimate:
      path: /absolute/path/to/frl-counts.csv
      geography_vintage: "2020"
  filters:
    optimization:
      geography_vintage: "2020"
      frl_estimate: updated_2526
```

Assignment uses the equivalent `assignment.frl_estimate` role.

### School-Year Registry

`school_years` selects annual student files and assignment program/school
bundles without embedding those choices in every scenario:

```yaml
school_years:
  "2324":
    optimization:
      students:
        applicant: optimization.students.applicant.2324
        enrolled: optimization.students.enrolled.2324
    assignment:
      students:
        applicant: assignment.students.2324
        enrolled: optimization.students.enrolled.2324
      grades:
        KG:
          profiles:
            default:
              standard:
                programs: assignment.programs.2324
                schools: assignment.schools.2324
            status_quo:
              mission_bay:
                programs: assignment.programs.2324.status_quo
                programs_catalog: assignment.programs.2324.mission_bay
                schools: assignment.schools.current_mission_bay
```

Every leaf in this registry is a catalog ID from `files`.

| Field | Type | Required | Description |
|---|---|---:|---|
| `<year>` | Four-digit school-year key | Yes | Registered year, for example `"2324"`. |
| `<year>.optimization` | Map | Yes | Optimization sources for the year. |
| `<year>.optimization.students` | Map | Yes | Must contain exactly `applicant` and `enrolled`. |
| `<year>.optimization.students.applicant` | Catalog ID | Yes | Applicant student table. |
| `<year>.optimization.students.enrolled` | Catalog ID | Yes | Enrolled student table. |
| `<year>.assignment` | Map | No | Assignment sources available for the year. If present, both `students` and `grades` are required. |
| `<year>.assignment.students` | Map | Yes when assignment exists | Must contain exactly `applicant` and `enrolled`. |
| `<year>.assignment.students.applicant` | Catalog ID | Yes when assignment exists | Applicant student table. |
| `<year>.assignment.students.enrolled` | Catalog ID | Yes when assignment exists | Enrolled student table. |
| `<year>.assignment.grades` | Non-empty map | Yes when assignment exists | Canonical grade to capacity-profile registry. |
| `<grade>` | Canonical grade key | Yes | `PK`, `TK`, `KG`, or zero-padded `"01"` through `"12"`. |
| `<grade>.profiles` | Non-empty map | Yes | Available assignment capacity profiles. |
| `<profile>` | Non-empty string key | Yes | Value accepted by assignment filter `capacity_profile`. |
| `<profile>.standard` | Source bundle | One variant required | Bundle selected when `include_mission_bay` is false. |
| `<profile>.mission_bay` | Source bundle | One variant required | Bundle selected when `include_mission_bay` is true. |
| `<variant>.programs` | Catalog ID | Yes | Program/capacity table. |
| `<variant>.programs_catalog` | Catalog ID | No | Richer program catalog onto which capacities from `programs` are mapped by `program_id`. |
| `<variant>.schools` | Catalog ID | Yes | School and school-coordinate table. |

Only `standard` and `mission_bay` are valid variant names. A profile may provide
one or both, but a requested Mission Bay policy must have the corresponding
variant. There is no fallback to another year, grade, profile, or variant.

When `programs_catalog` is present, both tables are filtered by grade, special
program policy, and Mission Bay policy before merging. The `programs` table must
contain unique `program_id` values and both `program_id` and `capacity` columns.
Its capacities replace matching catalog capacities. Unmatched catalog rows keep
their existing capacity, or remain missing if the catalog has no capacity
column.

The current registry supports optimization years `"1415"` through `"2627"`.
Assignment years `"1516"` through `"2223"` provide grades `KG`, `"06"`, and
`"09"` with profile `default` and variant `standard`. Assignment year `"2324"`
provides `KG/default/standard` and `KG/status_quo/mission_bay`. Assignment
years `"2425"` through `"2627"` provide grades `KG`, `"06"`, and `"09"` with
profile `default`; the `KG` profile carries both `standard` and `mission_bay`
variants, while `"06"` and `"09"` carry `standard` only. The checked-in
`base.yaml` remains authoritative when this inventory changes.

### The 2024-25 through 2026-27 transfer years

These three years come from the September 2026 SFUSD transfer and are built by
`analysis/data_prep/convert_sfusd_transfer.py`. The three per-year extracts
hold only students; a folder of auxiliary files that arrived on 2026-09-20 adds
the district's Main Round capacities and TK-to-K auto-promotion lists. What is
still borrowed, a reader has to know about:

| What | Where it comes from |
|---|---|
| Students, grades, ranked preferences, tie-breakers | That year's pre-run, post-run, and demographics extracts |
| Kindergarten programs and capacities | That year's Main Round capacity file, at gross `TotalSeats` |
| Grade 6 and 9 programs and capacities | That year's observed requests, with capacities from the 2022-23 district tables or the program's observed assignment count |
| School coordinates, category, ratings | The 2023-24 school tables |
| Block-level equity indices (`FRL Score`, `AALPI Score`, `HOCidx1`, …) | The 2021-22 through 2023-24 cleaned student files, joined on the 2010 Census Block |
| Preference rounds | One list per student, emitted as `r1_*`, so `rounds: all` resolves to `[1]`. That list is the main round; the later rounds are absent &mdash; see below |
| The kindergarten `enrolled` population | That year's post-run, not the KG rows of the applicant table &mdash; see below |
| A kindergarten student's auto-promotion claim | The post-run's current TK program, checked against the capacity file &mdash; see below |

Every emitted program row carries a `capacity_source` column naming its
provenance, and the converter writes
`sfusd_transfer_report_<year>.md` beside the tables listing every field the
transfer does not contain. Read that report before comparing a number from one
of these years against a checked-in year.

#### `rounds: [1]` is the main round, and the later rounds are absent

Each transfer holds exactly one preference list per student: the pre-run has no
round column and no student has a repeated rank. That list is the **main
round**. The district confirmed the extract is the main-round request file, and
the post-run corroborates it: every student it seats at a grade either holds a
pre-run request for that grade or is flagged `byPromote`.

What the transfer omits is the later-round requests themselves. The
demographics extract's per-student `rounds_applied` field records a later round
(`4` is the amendment round; `50`, `666`, `888`, `902`, `905`, `999` are
administrative codes) for a substantial minority of applicants &mdash; 9% of
2024-25 kindergarten applicants, 38% of 2025-26, and 27% of 2026-27 &mdash; but
no request rows accompany those tags, and `idRequest` spans the same range for
tagged and untagged students. The field is also unreliable in the other
direction: among 2026-27 enrolled kindergarteners who *do* hold a main-round
request, 1,598 have it blank and 793 are tagged round 4, so no value of it
identifies the main round. Use `rounds: [1]`, read it as "the main round", and
do not attempt a per-round comparison against the checked-in years, which carry
genuine separate `r1_`/`r2_`/`r4_` blocks.

#### Auto-promotion into kindergarten

From 2024-25 SFUSD auto-promotes TK students into kindergarten. A student in TK
at a school that also runs their pathway at kindergarten keeps that seat
without applying; a family that wants something else files an ordinary
main-round application and keeps the seat if nothing they asked for comes
through. So a kindergarten market is the main-round applicants *plus* the
promoted students, and some students are in both groups: 40 students took a
kindergarten seat without applying in 2024-25, 580 in 2025-26, and 847 in
2026-27, alongside 575, 278 and 341 promotion-eligible students who did apply.
Three things follow.

**The applicant pool is not the set of people who applied.** A promoted
student holds a claim on a kindergarten seat without filing anything, so
`student_<year>.csv` holds them too: its kindergarten rows are the Main Round
applicants *plus* the promotion-eligible students *plus* anyone else the run
seated at kindergarten without a request. Every row carries `mr_applicant`
&mdash; 1 for a Main Round applicant, 0 otherwise &mdash; and a promoted
student's preference list is reconstructed rather than transcribed, as below.
No other grade is treated this way: grades 6 and 9 are out of scope, so their
rows are Main Round applicants only.

`enrolled_<year>.csv` is then the subset of those rows the post-run seats,
less anyone the district records at school **899**, so `enrolled` is a subset
of `applicant` by construction. In all three transfer years every market
student takes a seat, so without the 899 rule the two would coincide at
`grades: [KG]`; that is a fact about these years rather than an invariant, and
a market student the run seated nowhere would appear in `applicant` alone.

**Assigned is not enrolled, and 899 means enrolled nowhere.** The post-run's
`idNextSchool` is the Main Round *assignment*. Where a student actually
enrolled for the fall is the demographics extract's `SCHOOL_CODE`, `GRADE` and
`ENR_PATHWAY`, and `SCHOOL_CODE` 899 ("Central Enrollment",
`loaders.tables.NOT_ENROLLED_SCHOOL_ID`) is the district's placeholder for a
student with no school. The converter therefore builds the enrolled table from
the enrolment record:

| Year | Seated at KG | At 899, dropped | Enrolled | From the record | Moved from their seat | No record, seat kept |
|---|---:|---:|---:|---:|---:|---:|
| 2024-25 | 3,875 | 45 | 3,830 | 3,149 | 485 | 675 (+6 recorded at another grade) |
| 2025-26 | 3,980 | 465 | 3,515 | 3,278 | 477 | 226 (+11) |
| 2026-27 | 3,996 | 19 | 3,977 | 3,233 | 447 | 734 (+10) |

In `enrolled_<year>.csv`, `enrolled_idschool` and `enrolled_programcode` are
where the student enrolled and `enrollment_source` says where that came from
(`fall_record`, `postrun_no_record`, `postrun_other_grade`). Everywhere else,
the Main Round outcome is unchanged: `final_school` and the `r1_*` outcome
columns in both tables, and `enrolled_idschool` in `student_<year>.csv`, are
still the post-run seat. So the two tables' `enrolled_idschool` differ for
the ~450-490 students a year who moved. 899 also appears as the post-run's
`idCurrentSchool` for applicants with no current SFUSD school. That is the same
placeholder at an earlier date, not a school.

The 2023-24 `enrolled_2324.csv` predates the converter. It is every KG
applicant, and a blank `enrolled_idschool` (794 of 4,304) marks one who did
not enroll. So the **assignment** group's enrolled population drops, at load
time, every row whose `enrolled_idschool` is blank or 899, in any year.
Optimization's enrolled population keeps those rows, as it always has, and
weights them out of the GE population through `enrolled_students` instead.

**Capacity is gross, and no seat is reserved anywhere in the data.** The
capacity file publishes both `TotalSeats` and `OpenSeatsPreRun`, the latter net
of the seats held for promotion. The tables use `TotalSeats`. A promote who
wins a school elsewhere releases the held seat back into the same run, so the
seats never leave the market: 413-GE had 52 open seats before the SY26-27 run
and made 53 choice assignments. Each program row carries the district's
`total_promote_before_run`, `total_promote_with_request_before_run` and
`free_seats` as provenance, and none of them touches `capacity`.

An earlier `capacity_profile: post_promotion` did net those seats out. It is
gone &mdash; profile and files &mdash; because it modelled a market the
district never ran. See
[`TK_PROMOTION_SPEC.md`](../analysis/data_prep/TK_PROMOTION_SPEC.md).

**The claim lives on the student, in four columns.** Every kindergarten row of
both tables carries:

| Column | Meaning |
|---|---|
| `promote_eligible` | 1 when the post-run puts the student in TK in a program that also exists at kindergarten that year |
| `feeder_school`, `feeder_program` | the program they are entitled to |
| `pref_source` | which source supplied their preference list |

`pref_source` takes four values. `k_list` is the student's own main-round list.
`tk_imputed` is the TK requests they filed the year before, mapped onto this
year's kindergarten programs and dropping any with no counterpart. `feeder_only`
and `aa_only` mean no source held them at all, split by whether they have a
feeder; those students get their feeder and their attendance-area GE program,
which is the *only* case in which the attendance area enters the data &mdash;
for everyone else appending it is the policy config's job (`add_aa_schools`).
An eligible student's feeder is appended to whatever list they have, unless
they already rank it.

`promote_eligible` is not the post-run's `byPromote` flag. That flag is
overloaded &mdash; TK promotion at kindergarten, Lowell and SOTA admission at
grade 9, K-8 continuation at grade 6 &mdash; and it is 0 for every 2024-25
student even though the capacity file holds seats. The rule used instead
reproduces the district's `TotalPromoteWithReqBeforeRun` exactly, program by
program, for 2026-27.

Two discrepancies are recorded rather than reconciled, both in the conversion
report's `promote_counts_vs_district_KG` section. For 2026-27 the rule
identifies 1,178 eligible students against 1,188 seats held, and for 2025-26
854 against 859. For 2024-25 the three sources disagree three ways: the
auto-promotion list says nobody was promoted, the capacity file holds 20 seats,
and the post-run seats 40 students with no application, only one of them at
their current school. The rule is applied uniformly all the same, so it marks
607 students eligible that year.

Finally, one limit of the loader: it drops any student who ranks no school in
the selected rounds, because every market indexes students by their ranked
list. The construction above gives every promoted student a list, so they
survive that filter &mdash; but a student with no list, no feeder, and no
usable attendance area does not, and the conversion report counts them under
`market_KG_students_with_no_list` (one student, in 2024-25). That student is
in both files and in neither loaded table.

The Early Education Schools are a case the data carries but does not yet use.
From the 2026-27 TK cohort, a TK student at an EES or the Mission Education
Center is promoted to a designated feeder elementary rather than to their own
site. The 24 rows describing that are parsed out of the SY26-27 list and kept,
but the first kindergarten class they place enters in SY2027-28, so no
converted year applies them. `EES_FEEDER_FIRST_YEAR` in
`analysis/data_prep/sfusd_transfer_schema.py` is the switch.

SY26-27 introduces one identity change worth knowing: Mission Bay ES appears in
real requests under school ID `1731`, which none of this repository's Mission
Bay handling recognises. The converter rewrites it to the canonical `999`, so
`include_mission_bay` keeps working as documented.

## Scenario Configuration

A scenario YAML contains exactly these required fields:

```yaml
id: mission-bay-2324
sources: {}
filters: {}
```

| Field | Type | Description |
|---|---|---|
| `id` | Non-empty string | Stable scenario identity included in manifests and semantic fingerprints. |
| `sources` | Map | Dotted source role to source reference. May be empty. |
| `filters` | Map | Zero or more complete `optimization` and `assignment` filter groups. |

Source role names must be strings containing a dot, such as
`optimization.programs` or `assignment.zones`. The loader does not otherwise
enumerate role names; consumers define which roles they need. Rootless direct
paths in this map are resolved from the scenario YAML.

If a scenario includes a filter group, every field for that group is required
except the four fields with loader defaults listed below. A run override may
specify only the fields it changes when the scenario already defines that
group. An override that introduces a new group must provide every non-defaulted
field because the final merged group is validated as complete.

## Filter Fields

The only defaults injected by the loader are:

| Field | Default |
|---|---|
| `capacity_scenario` | `programs` |
| `geography_vintage` | `"2010"` |
| `frl_estimate` | `null` |
| `outside_district_students` | `ignore` |

These defaults are applied within a configured group. They do not create an
`optimization` or `assignment` group when the scenario omits it.

Fields shared by both groups follow these rules:

| Field | Accepted values and behavior |
|---|---|
| `grades` | Non-empty, duplicate-free list containing `PK`, `TK`, `KG`, or zero-padded `"01"` through `"12"`. Student and program rows are restricted to these grades. |
| `student_population` | `applicant` or `enrolled`; selects that population from the school-year registry. |
| `rounds` | `all` or a non-empty, duplicate-free list of positive integers. Explicit lists are sorted and every requested round must exist in the student table. |
| `special_programs` | `include`, `exclude_only_special`, or `exclude_any_special`. `exclude_only_special` removes special-program preferences; `exclude_any_special` removes students listing a special program in a selected round. Both exclusion modes remove special program rows. |
| `capacity_scenario` | Non-empty string. `programs` retains capacities in the selected program table; another value selects an external capacity overlay. |
| `include_mission_bay` | Boolean. False removes school IDs 909 and 999 from school-keyed records and preferences. True normalizes the legacy ID `909` to `999`. |
| `geography_vintage` | Four-digit string registered in base `geographies`; currently `"2010"` or `"2020"`. |
| `frl_estimate` | Name registered in base `student_frl_estimates`, or `null` to retain FRL fields from the selected student table. |
| `outside_district_students` | `ignore` or `include`. When normalized student data has a `census_block` column, `ignore` removes rows with no selected-geography Block and `include` retains them. If that column is absent, this filter leaves the table unchanged. Optimization graph construction still rejects retained students lacking its selected geography. |

Student data must provide matching ranked-school and ranked-program columns for
every discovered round, including rounds not selected. `all` requires at least
one paired round. After round, Mission Bay, and special-program filtering,
students with no remaining preference in any selected round are removed.

The special program codes are the case-sensitive values `AF`, `DA`, `DT`, `ED`,
`MM`, `MS`, `SA`, `TC`, and `AO`. Either exclusion mode requires program tables
and optional program catalogs to contain `program_type`. Every requested grade
must occur in each selected program table and optional program catalog; a
missing requested grade fails rather than producing a partial result.

### Optimization Filters

```yaml
filters:
  optimization:
    years: ["2122", "2223", "2324"]
    grades: [KG]
    student_population: enrolled
    rounds: all
    special_programs: include
    program_population: GE
    capacity_scenario: programs
    include_k8: false
    include_citywide_zoning: false
    include_citywide_choice_opt: true
    include_mission_bay: true
    geography_vintage: "2020"
    frl_estimate: updated_2526
    outside_district_students: ignore
```

| Field | Type / values | Required | Description |
|---|---|---:|---|
| `years` | Non-empty list of unique four-digit strings | Yes | Selects one annual optimization student source per year, in configured order. Every year/population combination must exist in `school_years`. |
| `grades` | Canonical grade list | Yes | Shared grade selector described above. Multiple grades are schema-valid. |
| `student_population` | `applicant` or `enrolled` | Yes | Shared population selector described above. |
| `rounds` | `all` or positive-integer list | Yes | Shared preference-round selector described above. |
| `special_programs` | Supported special-program mode | Yes | Shared special-program selector described above. |
| `program_population` | Non-empty string | Yes | `GE` selects GE participants and GE population weights; `All` retains all program participants. Another exact program code filters students to that program. Non-`GE` graph weighting uses all-program totals. |
| `capacity_scenario` | Non-empty string | No | Capacity behavior described above. Defaults to `programs`. |
| `include_k8` | Boolean | Yes | Retains K-8 schools when true. This filter is applied only when `program_population` is not `All`. |
| `include_citywide_zoning` | Boolean | Yes | Citywide schools exist for the **base zoning problem** only when true (and `program_population` is `All`). See below. |
| `include_citywide_choice_opt` | Boolean | Yes | Citywide programs are alternatives in the **welfare markets and choice optimization** when true. See below. |
| `include_mission_bay` | Boolean | Yes | Shared Mission Bay policy described above. |
| `geography_vintage` | Four-digit registered vintage | No | Target Census geography. Defaults to `"2010"`. |
| `frl_estimate` | Registered estimate name or `null` | No | Block-count FRL source. Defaults to `null`, which retains optimization's source `FRL Score`. |
| `outside_district_students` | `ignore` or `include` | No | Outside-district policy. Defaults to `ignore`. |

### The two citywide selectors

A citywide school enrols from the whole district, so it does not belong to any
one zone. The two questions that follow are independent, and there is one
selector for each.

`include_citywide_zoning` decides whether a citywide school **exists for the
partitioning problem**. It is applied in `optimization.data.loaders.load_schools`,
which feeds the graph, so with it on the school occupies a node and its seats
and its count join the capacity balance (`overage`/`shortage`) and the
school-count constraints of whichever zone contains it. With it off the school
has no node, contributes no capacity, and is counted by no zone. It is off by
default: assigning a district-wide school to one zone's balance sheet
overstates that zone's supply to students who were never restricted to it.

`include_citywide_choice_opt` decides whether a citywide program is an
**alternative in the welfare markets** — `optimization.data.mid`, the SAA market
built on top of it, and the MNL zoning utility in `choice.mnl`. It is on by
default, because students really can attend these schools and a welfare number
that omits them understates what students get. In the markets such a program
carries `school_node=None`, which every oracle reads as *reachable from every
zone*; no zoning can grant or remove it.

The default pairing is therefore `zoning: false, choice_opt: true`: no zone owns
a citywide school, but every student may still choose one. The other
combinations are legal. `zoning: true` with `choice_opt: false` describes a
district that sites the school in a zone but scores welfare without it;
`true/true` treats it as an ordinary school on both sides.

Two consequences worth knowing:

- **Welfare scale.** On `summer-26-zoning` at Block level, citywide is 24 of 130
  programs and 723 of 4,189 seats — 17% of capacity, and about 22% of reported
  MID welfare. Turning `include_citywide_choice_opt` on or off moves every
  welfare number, so numbers computed under different settings are not
  comparable.
- **Zone-welfare decomposition.** Zone values are additive only when each
  program belongs to exactly one zone. A citywide program is held by every zone
  at once, so `dantzig_wolfe` requires `include_citywide_choice_opt: false` and
  rejects the config otherwise. The MID, SAA, iterative-choice, priced-access
  and short-bursts paths all support it.

Adding or renaming any selector changes every derived cache key, because
`DataScenario.source_manifest` hashes the whole filter block into each cache
namespace. The first run after this split rebuilds graphs and student tables.

`program_population` and `capacity_scenario` accept any non-empty string at
schema validation. Values other than the special cases above must be supported
by the selected source data.

### Assignment Filters

```yaml
filters:
  assignment:
    year: "2324"
    grades: [KG]
    student_population: applicant
    rounds: [1]
    special_programs: include
    capacity_profile: status_quo
    capacity_scenario: programs
    include_mission_bay: true
    geography_vintage: "2020"
    frl_estimate: updated_2526
    outside_district_students: ignore
```

| Field | Type / values | Required | Description |
|---|---|---:|---|
| `year` | Four-digit school-year string | Yes | Selects one assignment registry year. |
| `grades` | Canonical grade list | Yes | The schema accepts a list, but assignment source resolution requires exactly one grade per market. |
| `student_population` | `applicant` or `enrolled` | Yes | Shared population selector described above. |
| `rounds` | `all` or positive-integer list | Yes | Shared preference-round selector described above. |
| `special_programs` | Supported special-program mode | Yes | Shared special-program selector described above. |
| `capacity_profile` | Non-empty string | Yes | Selects a profile registered for the chosen year and grade. |
| `capacity_scenario` | Non-empty string | No | Capacity behavior described above. Defaults to `programs`. |
| `include_mission_bay` | Boolean | Yes | Selects the registry's `mission_bay` variant when true or `standard` when false, then applies the shared school-ID policy. |
| `geography_vintage` | Four-digit registered vintage | No | Target Census geography. Defaults to `"2010"`. |
| `frl_estimate` | Registered estimate name or `null` | No | Block-count FRL source. Defaults to `null`, which retains free- and reduced-lunch values from the student table. |
| `outside_district_students` | `ignore` or `include` | No | Outside-district policy. Defaults to `ignore`. |

### Capacity Overlays

When `capacity_scenario` is `programs`, the loader uses the `capacity` values in
the selected program table and does not read a capacity-overlay source. The
assignment study also has built-in scenarios `webster_plus_66_ge_2324`,
`webster_plus_69_ge_2425`, `webster_plus_69_ge_2526`, and
`mission_bay_ge_to_webster_2627`; these adjust the selected program table in
memory. The first three add Webster GE seats because Mission Bay is absent in
those years. The last moves 69 GE seats from Mission Bay to Webster in 2026-27.
See `assignment/configs/mb_comparison/README.md` for the study design.

Other explicit values, such as `A`, require the source role `<group>.capacity`
and read:

| Source column | Meaning |
|---|---|
| `SchNum` | School ID match key |
| `PathwayCode` | Program type match key |
| `Scenario_<value>_Capacity` | Capacity selected by `capacity_scenario` |
| `Grade` | Optional grade match key |

Selected overlay capacities must be numeric, finite, and non-negative. Matching
uses school and program, plus grade when `Grade` is present. At least one
selected program must match. Unmatched programs retain their program-table
capacities. Conflicting duplicate overlay rows fail after Mission Bay
normalization; when both alias ID `909` and canonical ID `999` are present, the
canonical row is preferred.

## Generated Source Roles and Precedence

Selectors generate these roles from the base registries:

| Filter group | Generated roles |
|---|---|
| `optimization` | `optimization.students`, `optimization.census`, `optimization.crosswalk`, `optimization.adjacency`, `optimization.manual_edges`, and optional `optimization.geography.blockgroups` / `optimization.geography.tracts` |
| `assignment` | `assignment.students`, `assignment.programs`, optional `assignment.programs.catalog`, `assignment.schools`, `assignment.school_coordinates`, and available `assignment.geography.blocks`, `assignment.geography.blockgroups`, `assignment.geography.tracts`, `assignment.geography.crosswalk` |

Source precedence, from lowest to highest, is:

1. Scenario `sources`.
2. Registry-generated annual and geography sources.
3. Run `overrides.sources`.

Map-valued sources merge recursively at each precedence layer. A named-map
override can replace one child while retaining the others, and a partial direct
source object can patch an existing direct source object. Catalog IDs are still
scalar strings at merge time, so patching a catalog-backed role requires either
a replacement catalog ID or a complete direct source object; catalog metadata
cannot be patched before expansion.

When the corresponding filter group exists, scenario-declared annual roles are
removed before registry generation. This prevents scenario files from bypassing
year, population, grade, profile, or Mission Bay selection. Explicit run source
overrides remain the escape hatch and take final precedence.

## Manual Block-Edge YAML

Files referenced by geography `manual_edges` can be a top-level list of edge
pairs or a map:

```yaml
edges:
  - [60750105001001, 60750179021001]
source_cases:
  60750105001001:60750179021001: [1]
review_graph_fingerprint: f57c47f462f897e3
```

| Field | Type | Required | Description |
|---|---|---:|---|
| `edges` | List of two-item GEOID lists | No | Runtime edge additions. Defaults to `[]` for a map payload. Values must convert to integers; booleans and self-edges are rejected. |
| `source_cases` | Map by convention | No | Compilation provenance from reviewed case numbers. Ignored and not type-validated by the runtime loader. |
| `review_graph_fingerprint` | String by convention | No | Graph identity recorded by review tooling. Ignored and not type-validated by the runtime loader. |

Edges are normalized as undirected pairs, deduplicated, and sorted. An empty or
YAML-null file is also treated as no edges. The parser checks integer
conversion, pair length, booleans, and self-edges; it does not validate GEOID
length or positivity. Applying the edges later fails if an endpoint is absent
from the target base graph. Additional map fields are ignored by the runtime
loader.

## Validation

The primary contract tests are in `loaders/tests/test_config.py`. Run the full
loader suite from the repository root with:

```bash
uv run python -m pytest loaders/tests
```

See `../DATA_FOLDERS.md` for external data layout, cache namespaces, migration
notes, and the broader data-management contract.
