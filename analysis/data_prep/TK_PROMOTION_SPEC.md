# TK-to-K promotion: data specification

Implementation spec for rebuilding the 2024-25, 2025-26 and 2026-27 kindergarten
market data around SFUSD's TK-to-K auto-promotion, using the auxiliary data that
arrived on 2026-09-20.

**This is a data-layer task.** It produces student and program tables. It does
not change the simulator, the lottery, the metrics, or the policy configs; §9
records the small config-layer follow-ups for whoever picks those up.

Ask any questions as needed and avoid making strong assumptions. Where a number
is quoted, it was verified against the files; reproduce it as a test rather than
trusting the prose.

**Stop and ask rather than guessing.** This spec is deliberately specific, but
it cannot anticipate everything the data will throw at you. Ask whenever: a
validation gate in §8 does not close and you cannot explain the difference from
the data; a rule here contradicts what the files actually contain; a case
appears that §7's edge cases do not cover; or you find yourself inventing a
placement, a capacity, or an eligibility rule that no input states. A wrong
assumption recorded silently in a student table costs far more than a question
— several facts in §1 only came to light because an earlier inference was
checked against the district's own counts and turned out to be wrong.

---

## 1. What the district actually does

Kindergarten has two entry paths. Students apply in the Main Round, *or* they
are already in TK and are auto-promoted into K without applying. Auto-promotion
was adopted in April 2025.

| TK site | K placement | Application needed |
|---|---|---|
| TK at an elementary or K-8 with K in the building | same school, same pathway | none |
| TK at an Early Education School (EES) or Mission Education Center | designated feeder elementary, same pathway | none — **but only from the 2026-27 TK cohort onward**, so the first affected class enters K in SY2027-28 |

A TK family that wants a different school files an ordinary Main Round
application. Two protections: if no requested program is assigned they keep
their current or feeder seat, and they are auto-waitlisted for their top three
choices. The waitlist is post-Main-Round and out of scope.

The 2025-26 EES cohort was *not* covered by the feeder rule and had to apply:
259 of them did so in the 26-27 run and none was promoted.

### Verified against the files

For the SY26-27 run, from the MR capacity file and the post-run:

- 1,188 K seats held for promotion (`TotalPromoteBeforeRun`)
- 341 of those students also filed a Main Round application
  (`TotalPromoteWithReqBeforeRun`)
- 847 = 1,188 − 341 took a K seat with no application at all
- 915 `byPromote` outcomes = 847 + the 68 applicants who were unsuccessful and
  fell back

All 847 landed at `idCurrentSchool`, and the program code was preserved in
847 of 847 cases. The same chain for 25-26 is 859 / 279 / 580 / 636.

**Released seats re-enter the same run.** At 413-GE there were 52 open seats
pre-run but 53 choice assignments, because promotes who won elsewhere released
their held seats. This is why the market is modelled as a single DA with a
priority boost rather than a two-stage pre-seating.

---

## 2. Scope

- Years: `2425`, `2526`, `2627`.
- Grade: **K only.** Grades 6 and 9 are out of scope for now — they have no
  attendance-area school in the data (0 of 3,278 and 0 of 4,565 students), and
  their reserved seats are a different phenomenon (grade 6's are invisible K-8
  continuers, grade 9's are Lowell/SOTA admissions).
- Do **not** treat seats as reserved anywhere in the data. Capacity is gross;
  the promotion claim is expressed as a priority boost at run time.
- Do **not** use `capacity_profile: post_promotion`. It nets out seats the real
  run released back into the market and is superseded by this spec.

---

## 3. Inputs

| Role | Path |
|---|---|
| Pre-run / post-run / demographics | `Data/raw_SFUSD_data_downloads/Sep 14 2026 data transfer/SY<YY-YY>/` |
| MR capacities | `.../auxillary data/Main Round capacities (including inflation) for SY24-25, SY25-26, SY26-27 - <YY-YY> MR Capacities.csv` |
| Autopromotion lists | `.../auxillary data/TK-to-K autopromotion lists - SY <YY-YY>.csv` |
| 2023-24 TK applications | `Data/Cleaned/student_2324.csv`, rows with `grade == "TK"` (1,320 students) |

Parsing notes:

- The 24-25 autopromotion list is a single prose line with no data rows.
- The 25-26 list has its header on line 14 → `skiprows=13`.
- The 26-27 list has its header on line 3 → `skiprows=2`.
- Capacity-file grade codes are `K`, `TK`, `PK`, `1`–`13`; normalize `K` → `KG`.
- Mission Bay is school `1731`; apply the existing 1731 → 999 alias.
- Student identity: raw `scrambledstudentno` `S000000001` → `1`, matching
  `_student_identity` in the converter. IDs are stable across years, which the
  cross-year lookups in §7 rely on.

---

## 4. Artifact 1 — the TK-to-K map

From the autopromotion lists, build
`(TK school, TK pathway) -> (K school, K pathway)`.

Ordinary elementaries map to themselves. EES and Mission Education Center rows
map to a feeder school and appear only in the 26-27 list (24 rows); their TK
pathway codes are compound and encode the destination (`GE750` → Sunset,
`SE420` → Alvarado, `GE644` → Jefferson, …).

Build the EES branch but leave it **unused** for all three years: its first
affected cohort enters K in SY2027-28. Keep it behind a flag so the SY27-28
transfer can switch it on without a rewrite.

---

## 5. Artifact 2 — program tables with real capacities

Build each year's K program table from that year's capacity-file rows:
166 / 166 / 162 programs, capacity = **`TotalSeats`** (4,099 / 4,202 / 4,306).

- Not `OpenSeatsPreRun`. Seats are not reserved in the data.
- Rows with `TotalSeats <= 0` are closed programs (13 / 10 / 2 at K).
- Carry `TotalPromoteBeforeRun`, `TotalPromoteWithReqBeforeRun` and `FreeSeats`
  through as provenance columns; they are inputs to the validation gates, not to
  capacity.

This removes every remaining 2023-24 capacity substitution for these years.

---

## 6. Artifact 3a — population and entitlement

The K market for a year is: **K Main Round applicants ∪ promotion-eligible
students.**

A student is **promotion-eligible** when their post-run row for that year has
`CurrentGrade == TK` and `(idCurrentSchool, CurrentProgramCode)` is a K program
in that year's capacity file. Their **feeder** is that same program.

This rule was validated for 26-27: it identifies 1,178 students against the
district's 1,188, and its applied subset matches the district's
`TotalPromoteWithReqBeforeRun` **exactly, program by program, at 341**.

Do not use `byPromote` for identification. It is 0 for every 24-25 student
despite seats existing, and at grade 9 it marks Lowell/SOTA admissions instead.

New student columns:

| Column | Meaning |
|---|---|
| `promote_eligible` | 1 when the rule above holds |
| `feeder_school`, `feeder_program` | the entitled K program |
| `pref_source` | `k_list`, `tk_imputed`, `feeder_only`, or `aa_only` |

---

## 7. Artifact 3b — preference construction

For each student in the market, in order:

```
base = submitted K Main Round list                    -> pref_source = k_list
     | prior-year TK list, mapped to K programs       -> pref_source = tk_imputed
     | empty

if promote_eligible:   list = base + [feeder]         (dedupe, keep earliest position)
if list is empty:      list = [feeder] + [AA GE program]   -> pref_source = feeder_only
```

TK-list mapping uses Artifact 1: same school and pathway for ordinary schools,
compound code → feeder for EES. Drop entries with no K counterpart.

Prior-year TK sources, by run year:

| Run | TK source |
|---|---|
| 2425 | `Data/Cleaned/student_2324.csv`, `grade == "TK"`, columns `r1_ranked_idschool` / `r1_programs` |
| 2526 | the 2425 pre-run, `Grade == "TK"` |
| 2627 | the 2526 pre-run, `Grade == "TK"` |

Using 2023-24 for the 24-25 run is explicitly allowed; the general prohibition
on forward-filling from 2023-24 concerns capacities, school attributes and the
choice estimate, not TK preference lists.

The attendance-area program enters the **data** only in the `feeder_only` and
`aa_only` cases — that is, only for a TK promote who has neither a K list nor a
TK list to work from (confirmed 2026-09-21, narrowing an earlier "append for
everyone under every policy").

For every other student, whether the AA program is appended is decided entirely
by the policy config's existing `add_aa_schools` setting, which the `#3`/`#4`
family sets and other policies do not. **Do not enable it anywhere it is not
already enabled**, and do not append AA in the data for anyone else.

Deduplication matters: 192 of the 341 promote-applicants in 26-27 already rank
their own feeder, 122 of them first.

### Edge cases

- **No list and no feeder.** 8 of the 40 relevant 24-25 students are not
  promotion-eligible (their current program is not a K program, e.g. an EES).
  Give them `[AA GE program]` alone, `pref_source = aa_only`.
- **No attendance area.** ~70 K students in 26-27 have no AA school, or an AA
  school with no GE K program. Emit an empty list and record them in the
  conversion report rather than inventing a placement.
- **Code 899.** ~575 K applicants report current grade TK with no current school
  on record. They are ordinary applicants: no feeder, no entitlement.

---

## 8. Validation gates

These close exactly. Fail the build if they do not.

| Run | `k_list` | `tk_imputed` | `feeder_only` + `aa_only` | Total | Post-run K seats |
|---|---|---|---|---|---|
| 2425 | 3,835 | 21 | 19 (15 + 4) | 3,875 | 3,875 |
| 2526 | 3,400 | 382 | 198 (194 + 4) | 3,980 | 3,980 |
| 2627 | 3,149 | 668 | 179 (169 + 10) | 3,996 | 3,996 |

The last column splits by whether the student has a feeder at all: `feeder_only`
is a promote with no lists, `aa_only` is the handful with no feeder either.

Of the `k_list` students, 0 / 279 / 341 are promotion-eligible and get a feeder
appended. Of the 40 students in the 24-25 non-applicant group, 32 are
promotion-eligible under the uniform rule.

Also assert, per program, for 26-27 K: identified eligible ≈ 1,188 (1,178 is the
expected shortfall — 10 students took no seat anywhere), and identified
eligible-with-application == `TotalPromoteWithReqBeforeRun` == 341 exactly.

---

## 9. Config layer — not this task

Recorded so the next person has it:

- **Promotion boost.** Implement the claim as a very strong tiebreaker at the
  feeder program only — never a global priority, or promotes would win every
  school they rank. Existing weights are zone 256, non_designation 128,
  soft_reserve 64, sibling 16, ctip 8, distance 4, so a boost of **1024**
  dominates all of them including FRL reserves. The promote outranks
  in-attendance-area new applicants at the feeder.

- **Where to add it, and why the placement is the whole ballgame.** Add the
  boost as a term in the weighted policy-priority matrix built by
  `PriorityGenerator._set_policy_priorities` — i.e. inside `priorities`, before
  the zone mask is applied. Do **not** add it after the mask, and do **not** add
  promotion to `Zones.zone_eligibility_matrix`.

  `restrict-zone` is enforced in `get_priorities` and
  `get_priorities_without_lottery` as:

  ```python
  if self.market.config["restrict-zone"]:
      zone_mask = self.market.zones.zone_eligibility_matrix
      return np.multiply(final, zone_mask) - (1 - zone_mask) * 500
  ```

  The mask is multiplicative, so an out-of-zone program is zeroed and then
  penalised to −500 **regardless of how large the boost is**. Putting the boost
  inside `priorities` therefore gives exactly the intended behaviour, with no
  magnitude tuning and no list surgery:

  | Policy | Feeder in zone | Feeder out of zone |
  |---|---|---|
  | `restrict-zone: false` (1, 2, 3, 5) | promote outranks everyone at the feeder | same — no zone restriction applies |
  | `restrict-zone: true` (4) | promote outranks everyone at the feeder | feeder collapses to −500; the promote is blocked and falls through to the AA append and designation |

  Note that `zone_eligibility_matrix` already carries the sibling and CTIP
  exceptions (`sibling-access`, `restrict-zone: "CTIP_access"`). Promotion is
  deliberately *not* one of them: under a zone policy the block wins. This is
  the mechanism by which policy 4 takes the feeder away from the ~570 promotes
  whose feeder sits outside their attendance area, which is the intended
  policy effect rather than a bug to work around.

  Verify with a unit test at both settings of `restrict-zone` rather than by
  reading the priority matrix.
- **Policy 5** (new): every student's list collapses to their AA GE program,
  capacity ignored, AA appended. Closest template is the `#3`/`#4` family with
  `overscribe_aa: true`. Language-pathway students collapse to AA GE as well.
- **Utility model off** (`utility-model.enable: false`): no choice estimate
  covers these cohorts, and the 2023-24 one is exactly the forward-fill being
  avoided.
- **Outcome accounting:** promotes are applicants, and landing on the feeder
  counts as a first choice.

---

## 10. Discrepancies to record, not fix

- **24-25.** The autopromotion list says no auto-promotion; the capacity file
  reserves 20 seats as singletons across 14 mostly special or language programs
  (SE, MS, TC, AF, AO, SN); the post-run flags zero `byPromote` yet shows 40
  students seated without an application, some from an EES. Three different
  populations. Honor the capacity file for capacity, apply the identification
  rule uniformly, and note the mismatch in the conversion report.
- **10 missing students** at 26-27: identified 1,178 against 1,188 reserved.
- **`byPromote` is overloaded**: TK promotion at K, Lowell/SOTA admission at
  grade 9, K-8 continuation at grade 6.
- **Current school 899** is a placeholder meaning no current SFUSD enrollment on
  record at run time — never a choosable school, never an assignment, never
  named. 460 of the 575 such K applicants in 26-27 held a 2025-26 TK seat they
  evidently did not keep.
