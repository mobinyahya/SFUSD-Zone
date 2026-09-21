# TK-to-K promotion in the assignment code: implementation plan

The data layer is done. This is the sketch for the half that is not: turning a
promoted student's claim into something the lottery respects.

Read [`analysis/data_prep/TK_PROMOTION_SPEC.md`](../../analysis/data_prep/TK_PROMOTION_SPEC.md)
first — it is the specification the data was built to, and its §9 is the
seed of this document. §4b of
[`TRANSFER_2425_2627_GAPS.md`](../../analysis/data_prep/TRANSFER_2425_2627_GAPS.md)
records what the conversion actually produced.

---

## 1. What already exists

For registry years `2425`, `2526` and `2627`, every kindergarten row of
`student_<year>.csv` and `enrolled_<year>.csv` carries:

| Column | Type | Meaning |
|---|---|---|
| `mr_applicant` | 0/1 | 1 when the student filed a Main Round request |
| `promote_eligible` | 0/1 | 1 when the student was in TK in a program that also runs at kindergarten |
| `promote` | list of program IDs | that program, e.g. `['664-AO-KG']` — **the column this plan keys on** |
| `feeder_school`, `feeder_program` | int, str | the same program, unfiltered, as provenance |
| `pref_source` | enum | `k_list` / `tk_imputed` / `feeder_only` / `aa_only` |

and the feeder is already **on the student's ranked list** — appended at the
end if they did not rank it themselves.

The kindergarten applicant pool is the whole market, promoted non-applicants
included:

| Run | market | Main Round applicants | promoted, no request | promotion-eligible |
|---|---|---|---|---|
| 2425 | 3,875 | 3,835 | 40 | 607 |
| 2526 | 3,980 | 3,400 | 580 | 854 |
| 2627 | 3,996 | 3,149 | 847 | 1,178 |

`promote` is a list of program IDs on purpose: it is the same shape as
`currentlpsibling`, which `loaders/tables.py` already filters and aliases for
`include_mission_bay`. Key the priority on `promote`, never on
`feeder_school` — in a run that excludes Mission Bay the former is correctly
empty and the latter is blanked separately.

**Nothing turns any of this into a priority.** A promoted student currently
competes for their own feeder on exactly the same footing as a stranger. That
is the gap this plan closes.

---

## 2. The change, in four pieces

### 2.1 `Students.promotion()` — the claim as a matrix

`assignment/student_assignment/data_interfaces/students.py`

A near-copy of `language_pathway_sibling` (same file, ~line 702), because both
read a list of program IDs:

```python
def promotion(self, program_id2index):
    """Return a (students x programs) 0/1 matrix of TK-to-K promotion claims.

    A student promoted from TK holds a claim on exactly one program -- the
    same pathway at the same school -- so at most one cell per row is set.
    Keyed on the ``promote`` column rather than on ``feeder_school`` because
    the loader filters ``promote`` for ``include_mission_bay`` and does not
    reconstruct the program ID.
    """
    promotion = np.zeros((self.n, self.num_programs), dtype=int)
    if "promote" not in self.student_data.columns:
        return promotion
    for i, value in enumerate(self.student_data["promote"]):
        indices = [
            program_id2index[program_id] - 1
            for program_id in self._programs_to_list(value)
            if program_id in program_id2index
        ]
        promotion[i, indices] = 1
    return promotion
```

The `if "promote" not in ...` guard matters: years 1516–2324 have no such
column and must keep working untouched.

Cache it on `self._promotion` beside `self._sibling` and `self._prek` if the
profile says it is worth it; it is O(n) with a dict lookup, so probably not.

### 2.2 The priority weight

`assignment/student_assignment/market_generator/priority_generator.py`, inside
`_set_policy_priorities` (the weights loop at ~line 227), beside the
existing `sibling` branch:

```python
elif k == "promote":
    priorities += v * self.market.students.promotion(self.market.programs.indices)
```

**Trap.** The kindergarten weights loop has no `else: raise` — unlike the
grade-6 loop at ~line 725, which raises `ValueError(f"Unknown priority
category '{category}'.")`. So adding `promote: 1024` to a policy config
*before* this branch exists is **silently ignored**: no error, no priority, a
run that looks fine and models nothing. Land the branch and the config in the
same change, and see §4.1 for the test that would have caught it.

Consider also adding the `else: raise` to the KG loop while you are here. It
is a one-line fix to a class of silent typo, and it is the reason this trap
exists at all. Across every checked-in policy config the KG keys are `ctip`,
`sibling`, `zone`, `distance`, `prek` and `language-programs`; the loop
branches on the first five and handles `language-programs` after it, so
`language-programs` is the only key an `else: raise` has to exempt.
(`brown-ms`, `bayview-students`, `remaining`, `lp`, `lp-sibling` and
`brown-ms-to-hs` belong to grade 6 and 9, which return earlier from
`_set_policy_priorities` and never reach this loop.)

### 2.3 The weight value

`promote: 1024` in the policy config's `priority-weights`.

Existing weights are zone 256, non_designation 128, soft_reserve 64,
sibling 16, ctip 8, distance 4, so 1024 dominates all of them including the
FRL reserves. A large value is safe here in a way it would not be for a
school-level priority: the claim is to **one program**, so it cannot help the
student anywhere else on their list.

### 2.4 Placement relative to the zone mask — the whole ballgame

Add the term **inside** `priorities`, i.e. before `_set_policy_priorities`
returns. Do **not** add it to the return value of `get_priorities_with_lottery`
or `get_priorities_without_lottery`, which apply the zone restriction
afterwards (~line 1068):

```python
if self.market.config["restrict-zone"]:
    zone_mask = self.market.zones.zone_eligibility_matrix
    return np.multiply(final, zone_mask) - (1 - zone_mask) * 500
```

The mask is multiplicative, so an out-of-zone program is zeroed and *then*
penalised to −500 however large the boost was. Putting the boost inside
`priorities` therefore gives the intended behaviour with no magnitude tuning
and no list surgery:

| Policy | Feeder in zone | Feeder out of zone |
|---|---|---|
| `restrict-zone: false` | promote outranks everyone at the feeder | same — no zone restriction applies |
| `restrict-zone: true` | promote outranks everyone at the feeder | feeder collapses to −500; the promote is blocked and falls through to the AA append and designation |

Under the policies this work is for, **"zone" means attendance area**: both
`#3` and `#4` set `zone-building-blocks: 'attendance_area'`, and
`Zones.zone_priority_matrix` keys on `row.idschoolattendance`. So the right
column is not a corner case, it is the main event — see §5.1 for how many
students it moves.

Do **not** add promotion to `Zones.zone_eligibility_matrix`. That matrix
already carries the sibling and CTIP exceptions (`sibling-access`,
`restrict-zone: "CTIP_access"`), and promotion is deliberately not one of
them: under a zone policy the block wins. That is the mechanism by which a
zone policy takes the feeder away from a promote whose feeder sits outside
their attendance area, which is the intended policy effect rather than a bug
to work around.

`_get_attendance_area_priorities` (~line 330) is the closest existing model
for all of this: per-student, one program, added inside `priorities`.

---

## 3. Two decisions that need a human, not a default — one now taken

### 3.1 Does landing on the feeder count as a first choice?

**Decided: yes.** Kumar, 2026-09-20: "treat promotes as applicants, and getting
your promote counts as getting your first choice." So implement the first
reading below — special-case `submitted_rank = 1` when the assigned program is
the student's `promote`.

The rest of this subsection stands as the reason to also build the second
column. The decision was taken before the magnitudes below were on the table,
and they are large enough that anyone quoting a top-choice number from these
years needs to know which convention produced it. Recommendation: implement the
decision *and* emit `rank_excluding_promotion` alongside it, so the sensitivity
is one column away rather than a re-run. Raise the schema-version question
(§3.1, last paragraph) when you do.

The data as built does not produce the decided behaviour on its own, and the
gap is not small:

| Run | promotion-eligible | feeder is their rank-1 choice | feeder is lower down |
|---|---|---|---|
| 2425 | 607 | 287 | 320 |
| 2526 | 854 | 529 | 325 |
| 2627 | 1,178 | 733 | 445 |

`choice_ranks.py` derives `submitted_rank` from the student's listed ranks
against the program they were assigned. A promote-applicant who ranked five
schools and had the feeder appended at rank 6 is recorded as receiving their
sixth choice.

Two defensible readings:

* **As written in §9, and as decided.** The promote got the thing they were
  entitled to, so the outcome is a success regardless of position. Implement by
  special-casing `submitted_rank = 1` when the assigned program is the
  student's `promote`.
* **As the data reads now.** A promote-applicant who lands on the feeder lost
  everything they asked for and fell back. Calling that a first choice
  flatters every top-choice metric these years produce, and by a lot: it would
  move 445 of 1,178 students in 2026-27.

The two give materially different headline numbers for exactly the metric
these scenarios are most likely to be quoted on. Report both, as `rank` and a
new `rank_excluding_promotion`: it costs one column and ends the argument, but
adds a column to a schema (`ASSIGNMENT_SCHEMA_VERSION`) that several consumers
validate, so confirm the bump before shipping it.

There is a related convention already settled on the data side, and the two
should not drift apart: the appended attendance-area program counts as a
submitted choice **only when the student's list would otherwise be empty**
(Kumar, 2026-09-20). That is why `pref_source = feeder_only` and `aa_only`
students have a genuine rank-1 — and those are the only students for whom the
data layer appends the AA program at all (§5.2). Where a *policy* appends it
for everyone else, as `#3`/`#4`/5 do, it sits past `pref_length` and is not a
choice they made.

Whichever is chosen: a `feeder_only` student's feeder *is* at rank 1 already,
so nothing needs doing for them. The question is only about the promotes who
also applied.

### 3.2 An imputed list lets a promote win somewhere they never applied

This one is a consequence of the data layer that the assignment layer inherits,
and it means the baseline will **not** reproduce the historical match.

In reality, a TK student who filed no Main Round request competed for nothing
and took their feeder: all 847 of them landed at `idCurrentSchool` in 2026-27.
In the converted data those students carry a list imputed from the TK requests
they filed the year before, and for most of them it is longer than the feeder
alone:

| Run | `tk_imputed` students | …whose list holds more than the feeder |
|---|---|---|
| 2425 | 21 | 14 |
| 2526 | 382 | 259 |
| 2627 | 668 | 477 |

So in simulation those 477 students will compete for, and some will win,
schools they did not apply to in the real 2026-27 run. That is deliberate —
§7 of the spec asks for the imputation, because a counterfactual policy
simulation wants their preferences rather than their non-behaviour — but it
means:

* a `real_match`-style baseline cannot be scored against the district's actual
  2026-27 placements for these students, and
* the district's `byPromote` count (915 in 2026-27) is **not** a target the
  simulation should hit.

Decide and write down which baseline these years are validated against before
anybody quotes a match rate from them. If the answer is "we need one that
reproduces the real run", the cheapest route is a policy flag that truncates
every `tk_imputed` student's list to their feeder alone, rather than a second
conversion.

---

## 4. Tests

### 4.1 Unit — the priority is real and is placed correctly

`assignment/tests/market_generator/test_priority_generator.py` has the harness
(`Configerator` + synthetic data via `configure_synthetic_assignment_data`);
`test_attendance_area_policy.py` is the closer model for a KG per-student,
per-program boost.

1. **The claim becomes a matrix.** A student with `promote = ['413-GE-KG']`
   gets a 1 at that column of `Students.promotion(...)` and nowhere else. A
   student with `promote = []` gets an all-zero row.
2. **The weight is actually read.** With `priority-weights: {promote: 1024}`,
   `_set_policy_priorities` differs from the same config without the key by
   exactly 1024 at the feeder cell. *This is the test that catches the silent
   ignore in §2.2 — write it first and watch it fail.*
3. **The boost beats every other priority.** A non-promote with sibling +
   CTIP + zone at the same program ranks below the promote.
4. **`restrict-zone` still wins.** Same market, two runs. With
   `restrict-zone: false` the promote is top at the feeder; with
   `restrict-zone: true` and the feeder out of zone, the feeder's entry is
   −500 and the promote is placed by the AA append instead. Assert on the
   assignment outcome, not on the priority matrix — reading the matrix tests
   the implementation rather than the behaviour, and the whole point of §2.4
   is that the two look different.
5. **Older years are untouched.** A 2023-24 market, which has no `promote`
   column, produces exactly the priority matrix it does today.

### 4.2 Integration — a promote keeps their seat

A small market where a promoted student ranks an oversubscribed school first
and their feeder second, and every seat at the first school is taken by
higher-priority students: the promote must land at the feeder, not unassigned.
Then the mirror: the same student wins the oversubscribed school, and the
feeder seat they released is taken by someone else — that is the released-seat
behaviour the gross-capacity decision rests on (413-GE had 52 open seats
pre-run and made 53 choice assignments).

### 4.3 Real data — a sanity band, not an equality

Mark `real_data`. After a DA run on `sfusd-2627`, `grades: [KG]`:

* every student with a non-empty `promote` who is assigned at all is assigned
  **either** to their `promote` program **or** to something they ranked above
  it — never below, and never unassigned while the feeder has a seat;
* the count landing at the feeder is at least the 68 promote-applicants the
  district recorded falling back, and at most 1,178.

Do not assert equality with `byPromote = 915`; §3.2 says why.

---

## 5. The policy configs

### 5.1 Which policies these are

Settled with Kumar on 2026-09-20. "Neighborhood policies" means the **distance**
policies — the ones built on attendance areas rather than on drawn zones.

| # | Config | `restrict-zone` | Notes |
|---|---|---|---|
| 3 | `distance_05_1_2+reserves_05frl_#3` | `false` | distance priority [0.5, 1, 2], FRL reserves 50/50, `guard-rails: 0`, `add_aa_schools: true`, `drop_below_aa: false`, `overscribe_aa: true`, MTB |
| 4 | `distance_05_1_2+reserves_05frl_#4` | `true` | identical except the zone restriction |
| 5 | new | — | neighborhood assignment, no choice (§5.3) |

Policies 1 and 2 were never named in the meeting. Do not guess them — confirm
the numbering before wiring anything that depends on it.

Both 3 and 4 set `zone-building-blocks: 'attendance_area'`, so policy 4's
"zone" is the student's attendance area. That makes §2.4's right-hand column
the common case, not the exception:

| Run | promotion-eligible | feeder **is** their AA school | feeder is elsewhere | no AA school |
|---|---|---|---|---|
| 2425 | 607 | 110 | 487 | 10 |
| 2526 | 854 | 227 | 605 | 22 |
| 2627 | 1,178 | 338 | 794 | 46 |

So under policy 4 roughly **two thirds of promotes lose their feeder** — 794 of
1,178 in 2026-27 — and land via the AA append or designation instead. That is
the policy's intended bite (a TK seat won through citywide choice a year
earlier is not reachable under neighborhood assignment), not a defect. Expect
the headline gap between policies 3 and 4 to be driven substantially by these
students, and sanity-check that it is.

### 5.2 What every policy config gains

* **`promote: 1024`** in `priority-weights` — but only once §2.2's branch
  exists, or it is silently ignored.
* **`add_aa_schools` — leave every config exactly as it is.** Kumar corrected
  an earlier instruction on 2026-09-21: the attendance-area program is appended
  only for **TK promotes who have neither a K preference list nor a TK
  preference list**, and the data layer has already done it for them
  (`pref_source = feeder_only`, whose list is `[feeder, AA GE]`; plus the
  handful of `aa_only` students who have no feeder either).

  So there is **no sweep to do**. Do not add `add_aa_schools: true` to the
  status-quo or zone configs. The `#3`/`#4` family already sets it and policy 5
  needs it, but that is those policies' own design — appending the AA program
  for everyone is part of what a neighborhood policy *is*, not a global rule.

  | Students | Where the AA program comes from |
  |---|---|
  | `feeder_only`, `aa_only` (15 + 4 / 194 + 4 / 169 + 10) | already on the list, from the data layer, under every policy |
  | everyone else, under policies 3/4/5 | appended at run time by `add_aa_schools: true`, as those configs already do |
  | everyone else, under any other policy | not appended at all |

  One residual worth knowing: appending is not guaranteeing. The AA program is
  a real seat subject to capacity unless `overscribe_aa: true`, so a
  `feeder_only` student under a policy that does not overscribe can still end
  up unassigned if both their feeder and their AA school fill. That is a small
  population and arguably the correct outcome, but it means "they always have
  the AA fallback" is not literally true outside policies 3/4/5.

### 5.3 Policy 5 (new)

Every student's list collapses to their attendance-area GE program, capacity
ignored, AA appended; language-pathway students collapse to AA GE as well
(Kumar, 2026-09-20 — explicitly a config concern, with no data-layer
counterpart). Closest template is the `#3`/`#4` family with
`overscribe_aa: true` — see
`assignment/configs/policy_configs/distance_05_1_2+reserves_05frl_#3_no_drop_below_aa.yaml`
and `remove_non_aa_or_citywide` in `preference_generator.py`.

This policy makes the promotion boost inert by construction: if the only
program on the list is the AA GE one, there is nothing for the claim to act on
unless the feeder happens to be it — which is true for 338 of 1,178 students in
2026-27 and nobody else. Policy 5 is therefore the sharpest statement of the
counterfactual: every promote whose TK seat was won outside their neighborhood
gives it up.

### 5.4 Utility model off

`utility-model.enable: false` for these years. No choice estimate covers these
cohorts — the 2023-24 estimate is a per-student matrix covering 0.9% of 2024-25
and 0% of the other two — and using it is exactly the forward-fill the
conversion avoids. `choice/mnl.py` already warns below 50% coverage.

### 5.5 Lottery and tie-breaking

Nothing to do, and nothing to add to the data. Kumar, 2026-09-20: lottery
handling stays a runtime concern. Keep the existing `ties-options: [MTB]` and
the configured iteration count; promoted non-applicants have no district
`RandomNumber`, and the fresh per-iteration draw already covers them.

---

## 6. Two things this plan assumes that are not true yet

### 6.1 Capacity is still the 2023-24 borrow

§4.2 and §4.3 rest on gross capacity with released seats re-entering the
market. The district's Main Round capacity files (received 2026-09-20,
`.../auxillary data/Main Round capacities...csv`) are what makes that correct:
they give `TotalSeats` per program per year for every grade, and it is
`TotalSeats` the spec calls for — 4,099 / 4,202 / 4,306 at kindergarten.

**They are not wired in.** The four `sfusd-*` scenarios currently resolve to
`capacity_profile: default`, which is the 2023-24 borrow — capacities from a
different year, below the observed seatings for ~65 KG programs per year. The
interim `post_promotion` profile, which netted the promoted seats out of
capacity, has been retired from every scenario precisely because the released
seats must re-enter; do not reach for it.

Consequences while this is outstanding: §4.3's band is a smoke test rather than
a measurement, the released-seat behaviour in §4.2 cannot be checked against
413-GE's real 52-open/53-assigned numbers, and no capacity-sensitive figure
from these years should leave the room. The priority work in §2 is unaffected
and can land first.

### 6.2 2024-25 had no auto-promotion, but 607 students are flagged for it

The conversion applies one identification rule across all three years, so
`student_2425.csv` marks 607 kindergarten students `promote_eligible`. The
policy did not exist that year:

* the district's own list for SY24-25 reads `NONE - All TK students had to
  reapply for K`;
* auto-promotion was adopted in **April 2025**, after the SY24-25 round ran;
* the SY24-25 capacity file reserves 20 seats total, scattered as singletons
  across 14 mostly special or language programs (SE, MS, TC, AF, AO, SN) — not
  a TK cohort;
* the SY24-25 post-run flags zero `byPromote` at kindergarten.

Give all 607 a 1024 boost and the 2024-25 run models a policy a year before it
existed — and 2024-25 is the year the design relies on as the *pre*-promotion
baseline, so the error lands exactly where it does the most damage.

Recommendation: gate the promotion weight per year and leave it off for
`2425`, keeping that year as the clean comparison. The flags stay in the data
(they are a true statement about where those students were enrolled); the
priority does not. Confirm with Kumar before implementing — this was resolved
on the data side as "honor the capacity file and note the inconsistency", which
does not by itself say what the priority layer should do.

---

## 7. Decisions already taken

Settled with Kumar on 2026-09-20 unless noted. Recorded so nobody re-derives
them from the older meeting notes, which differ in one place.

| Decision | Note |
|---|---|
| The promote **always wins** at their feeder — including over a new applicant who lives in that school's attendance area | **Supersedes** the earlier meeting TODO, which had in-neighborhood new applicants outranking out-of-neighborhood promotions. Do not reintroduce that inversion |
| Under a zone policy the promote is hard-blocked from an out-of-zone feeder | The fallbacks (AA append, designation) take over — §2.4 |
| "In-neighborhood" means **attendance area**, for every purpose | Both for the block and for the append |
| The AA program is appended **only** for TK promotes with neither a K list nor a TK list | Corrected 2026-09-21, superseding an earlier "everyone, every policy". The data layer has already done it; no config sweep — §5.2 |
| The AA append counts as a submitted choice only if the list would otherwise be empty | §3.1 |
| Promotion priority sits **above** the FRL reserves | §2.3 |
| Landing on the feeder counts as a first choice | §3.1, with the caveat recorded there |
| Seats are never modelled as reserved; capacity is gross | §6.1 |
| Utility model off | §5.4 |
| Grade K only for now | Grades 6 and 9 have no attendance-area school at all in the data (0 of 3,278 and 0 of 4,565 students in 2026-27), their reserved seats are different phenomena — grade 6's are K-8 continuers who never appear in our files, grade 9's are Lowell/SOTA admissions decided outside the lottery — and `byPromote` means something different at each grade |
| The EES feeder rule is inert for all three years | The feeder system starts with the 2026-27 TK cohort, who enter K in SY2027-28. In every year modelled here `promote` is the student's own school and pathway, so the different-school branch of the map is never exercised |

One implication of the last two rows for §4.3: `byPromote` is overloaded across
grades and years — 0 at kindergarten in 2024-25 despite seats existing,
Lowell/SOTA admission at grade 9, K-8 continuation at grade 6. It is not a
target and it is not an identification rule; the spec's §6 rule is.

---

## 8. Order of work

1. §4.1 test 2 (the weight is read) — fails.
2. §2.1 `Students.promotion` and §2.2 the branch — test 2 passes.
3. §4.1 tests 1, 3, 5.
4. §2.4's behaviour test, §4.1 test 4. This is the one worth getting right;
   everything else is bookkeeping.
5. §6.2 — gate the weight off for `2425` before any three-year run, or that
   year silently models a policy that did not exist. Cheap, and easy to forget
   once the priority works.
6. §3.2, and the `rank_excluding_promotion` column from §3.1. §3.1's headline
   convention is decided; §3.2's baseline question is not, and it blocks
   quoting a match rate.
7. §4.2, then §4.3 — but see §6.1: until the Main Round capacity files are
   wired in, §4.3 is a smoke test, not a measurement.
8. Nothing to do for `add_aa_schools` — §5.2 explains why the sweep that used
   to be listed here was withdrawn.
9. Policy 5 (§5.3) and the utility-model flag (§5.4), which are independent of
   all of the above.
