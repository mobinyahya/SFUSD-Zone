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

## 3. Two decisions that need a human, not a default

### 3.1 Does landing on the feeder count as a first choice?

§9 of the spec says "promotes are applicants, and landing on the feeder counts
as a first choice." The data as built does not produce that, and the gap is
not small:

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

* **As written in §9.** The promote got the thing they were entitled to, so
  the outcome is a success regardless of position. Implement by special-casing
  `submitted_rank = 1` when the assigned program is the student's `promote`.
* **As the data reads now.** A promote-applicant who lands on the feeder lost
  everything they asked for and fell back. Calling that a first choice
  flatters every top-choice metric these years produce, and by a lot: it would
  move 445 of 1,178 students in 2026-27.

The two give materially different headline numbers for exactly the metric
these scenarios are most likely to be quoted on. **Ask before implementing
either.** A third option — report both, as `rank` and a new
`rank_excluding_promotion` — costs one column and ends the argument, but adds
a column to a schema (`ASSIGNMENT_SCHEMA_VERSION`) that several consumers
validate.

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

## 5. Also in §9 of the spec, not covered above

* **Policy 5 (new).** Every student's list collapses to their attendance-area
  GE program, capacity ignored, AA appended; language-pathway students
  collapse to AA GE as well. Closest template is the `#3`/`#4` family with
  `overscribe_aa: true` — see
  `assignment/configs/policy_configs/distance_05_1_2+reserves_05frl_#3_no_drop_below_aa.yaml`
  and `remove_non_aa_or_citywide` in `preference_generator.py`. Note this
  policy makes the promotion boost inert by construction: if the only program
  on the list is the AA GE one, there is nothing for the claim to act on
  unless the feeder happens to be it.
* **Utility model off** (`utility-model.enable: false`) for these years. No
  choice estimate covers these cohorts — the 2023-24 estimate is a per-student
  matrix covering 0.9% of 2024-25 and 0% of the other two — and using it is
  exactly the forward-fill the conversion avoids. `choice/mnl.py` already
  warns below 50% coverage.

---

## 6. Order of work

1. §4.1 test 2 (the weight is read) — fails.
2. §2.1 `Students.promotion` and §2.2 the branch — test 2 passes.
3. §4.1 tests 1, 3, 5.
4. §2.4's behaviour test, §4.1 test 4. This is the one worth getting right;
   everything else is bookkeeping.
5. Ask about §3.1 and §3.2. Neither blocks the priority work, and both block
   quoting a number.
6. §4.2, then §4.3.
7. Policy 5 and the utility-model flag, which are independent of all of the
   above.
