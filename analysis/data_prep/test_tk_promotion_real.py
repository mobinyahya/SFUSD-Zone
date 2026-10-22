"""TK-to-K promotion checked against the real September 2026 transfer.

The synthetic tests in ``test_convert_sfusd_transfer.py`` pin the *rules*. These
pin the *numbers*, because the rules were chosen by reconciling them against
the district's own counts and a rule that stops reproducing those counts has
stopped being the district's rule. Every figure here was read off the transfer
and cross-checked against a second source in it:

* the program counts and seat totals against the Main Round capacity file;
* the market size against the post-run's seated kindergarten cohort;
* the promotion counts against the capacity file's ``TotalPromoteBeforeRun``
  and ``TotalPromoteWithReqBeforeRun``, program by program.

They need the shared data root, so they carry the ``real_data`` marker. They
deliberately do not run the geography join or the Block-index lookup, which are
the slow parts of a conversion and have nothing to do with promotion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.data_prep import convert_sfusd_transfer as converter
from analysis.data_prep.convert_sfusd_transfer import (
    Report,
    add_market_students,
    build_market_table,
    build_student_table,
    define_market,
    discover_transfer_files,
    drop_unlocatable_requests,
)
from analysis.data_prep.sfusd_transfer_schema import TRANSFER_YEAR_FOLDERS
from analysis.data_prep.tk_promotion import (
    build_promotion_map,
    discover_auxiliary_files,
    load_mr_capacities,
)
from loaders import load_scenario

pytestmark = pytest.mark.real_data

TRANSFER_NAME = "Sep 14 2026 data transfer"
YEARS = tuple(TRANSFER_YEAR_FOLDERS)

#: Kindergarten programs the capacity file lists, the seats they hold, and how
#: many of them it opens with none.
CAPACITY_EXPECTATIONS = {
    "2425": {"programs": 166, "seats": 4099, "closed": 13},
    "2526": {"programs": 166, "seats": 4202, "closed": 10},
    "2627": {"programs": 162, "seats": 4306, "closed": 2},
}

#: The kindergarten market, by where each student's preference list came from.
#: ``feeder_only`` and ``aa_only`` are reported separately but gate together:
#: they are the one bucket "no source held this student", split by whether
#: they have a feeder to fall back on.
MARKET_EXPECTATIONS = {
    "2425": {"k_list": 3835, "tk_imputed": 21, "no_source": 19, "total": 3875},
    "2526": {"k_list": 3400, "tk_imputed": 382, "no_source": 198, "total": 3980},
    "2627": {"k_list": 3149, "tk_imputed": 668, "no_source": 179, "total": 3996},
}

#: Promotion-eligible students identified by the rule, and how many of them
#: also filed a kindergarten application. The district's own counts for the
#: same two quantities are ``TotalPromoteBeforeRun`` and
#: ``TotalPromoteWithReqBeforeRun``; they agree exactly only for 2026-27, and
#: ``test_the_identified_promotes_are_compared_with_the_district`` says why.
PROMOTION_EXPECTATIONS = {
    "2425": {"eligible": 607, "with_application": 575},
    "2526": {"eligible": 854, "with_application": 278},
    "2627": {"eligible": 1178, "with_application": 341},
}


@pytest.fixture(scope="module")
def data_root() -> Path:
    scenario = load_scenario({"scenario": "legacy", "overrides": {}})
    return Path(scenario.roots["data"])


@pytest.fixture(scope="module")
def cleaned_dir(data_root: Path) -> Path:
    return data_root / "Data" / "Cleaned"


@pytest.fixture(scope="module")
def transfer(data_root: Path) -> Path:
    folder = data_root / "Data" / "raw_SFUSD_data_downloads" / TRANSFER_NAME
    if not folder.is_dir():
        pytest.skip(f"The transfer {folder} is not on this machine.")
    return folder


@pytest.fixture(scope="module")
def markets(transfer: Path, cleaned_dir: Path) -> dict[str, dict]:
    """Build each year's kindergarten market once, without the geography join.

    ``add_market_students`` resolves Census geography and Block indices for
    the promoted students it adds. Those are the slow parts of a conversion
    and none of the assertions here touches them, so both are stubbed out.
    """
    original = converter.attach_geography

    def no_geography(students, report, *, scenario_name="legacy", report_label="x"):
        frame = students.copy()
        for column in ("census_block", "census_blockgroup", "census_tract"):
            frame[column] = pd.NA
        return frame

    converter.attach_geography = no_geography
    try:
        return _build_markets(transfer, cleaned_dir)
    finally:
        converter.attach_geography = original


def _build_markets(transfer: Path, cleaned_dir: Path) -> dict[str, dict]:
    built: dict[str, dict] = {}
    for year in YEARS:
        report = Report(year=year, transfer=str(transfer))
        paths = discover_transfer_files(transfer, year)
        auxiliary = discover_auxiliary_files(transfer, year)
        for path in (*paths.values(), *auxiliary.values()):
            converter.require_readable(path, f"{year} input")

        prerun = converter._clean_nulls(pd.read_csv(paths["prerun"], low_memory=False))
        postrun = converter._clean_nulls(
            pd.read_csv(paths["postrun"], low_memory=False)
        )
        demographics = converter._clean_nulls(
            pd.read_csv(paths["demographics"], low_memory=False)
        )
        prerun = drop_unlocatable_requests(
            prerun, cleaned_dir=cleaned_dir, gaps="fill-and-report", report=report
        )
        capacities = load_mr_capacities(auxiliary["capacities"])
        promotion = build_promotion_map(auxiliary["autopromotion"], year)
        students = build_student_table(
            prerun, postrun, demographics, report=report, gaps="fill-and-report"
        )
        market = define_market(
            students,
            prerun,
            postrun,
            grade="KG",
            capacities=capacities,
            cleaned_dir=cleaned_dir,
            report=report,
        )
        students = add_market_students(
            students,
            postrun,
            demographics,
            pd.DataFrame(index=pd.Index([], name="census_block")),
            market=market,
            report=report,
            scenario_name="legacy",
        )
        table = build_market_table(
            students,
            postrun,
            market=market,
            year=year,
            capacities=capacities,
            promotion=promotion,
            prior_tk=converter.load_prior_tk_lists(
                year, transfer=transfer, cleaned_dir=cleaned_dir, report=report
            ),
            report=report,
        )
        built[year] = {
            "market": table,
            "definition": market,
            "students": students,
            "applicants": market.applicants,
            "prerun": prerun,
            "postrun": postrun,
            "capacities": capacities,
            "promotion": promotion,
            "report": report,
        }
    return built


# --------------------------------------------------------------------------- #
# Artifact 1 -- the TK-to-K map
# --------------------------------------------------------------------------- #
def test_the_2024_25_list_says_no_auto_promotion_happened(markets):
    promotion = markets["2425"]["promotion"]
    assert promotion.empty
    assert promotion.active_feeders() == {}


def test_only_the_2026_27_list_carries_early_education_feeders(markets):
    assert markets["2526"]["promotion"].ees_feeder == {}
    ees = markets["2627"]["promotion"].ees_feeder
    assert len(ees) == 24
    # The TK pathway code is compound and encodes the destination, which is
    # why an EES needs a map at all: one site feeds several elementaries.
    assert ees[(928, "GE750")] == (750, "GE")
    assert ees[(724, "SE420")] == (420, "SE")
    assert ees[(903, "GE644")] == (644, "GE")


def test_no_transfer_year_applies_the_early_education_feeder_rule(markets):
    # The rule covers the 2026-27 TK cohort, whose first kindergarten class
    # enters in SY2027-28.
    for year in YEARS:
        promotion = markets[year]["promotion"]
        assert not promotion.ees_active
        assert promotion.active_feeders() == promotion.same_site


# --------------------------------------------------------------------------- #
# Artifact 2 -- program tables
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("year", YEARS)
def test_kindergarten_capacity_matches_the_capacity_file(markets, year):
    capacities = markets[year]["capacities"]
    kindergarten = capacities.loc[capacities["grade"] == "KG"]
    expected = CAPACITY_EXPECTATIONS[year]
    assert len(kindergarten) == expected["programs"]
    assert int(kindergarten["capacity"].sum()) == expected["seats"]
    assert int((kindergarten["capacity"] <= 0).sum()) == expected["closed"]


# --------------------------------------------------------------------------- #
# Artifact 3 -- the market
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("year", YEARS)
def test_the_market_is_the_student_table_at_the_grade(markets, year):
    # student_<year>.csv is the applicant pool, so its kindergarten rows are
    # the market -- not the subset of it that filed a request.
    market = markets[year]["market"]
    students = markets[year]["students"]
    at_grade = students.loc[students["grade"] == "KG"]
    assert len(market) == MARKET_EXPECTATIONS[year]["total"]
    assert set(market.index) == set(at_grade["studentno"])


@pytest.mark.parametrize("year", YEARS)
def test_the_market_is_also_the_post_run_seated_cohort(markets, year):
    # Every market student takes a kindergarten seat in all three years, so
    # the enrolled population coincides with the applicant one. That is a fact
    # about these years, not an invariant: a market student with no seat would
    # be in student_<year>.csv and absent from enrolled_<year>.csv.
    market = markets[year]["market"]
    postrun = markets[year]["postrun"]
    seated = converter.seated_at_grade(converter._postrun_outcomes(postrun), "KG")
    assert set(market.index) == set(seated.index)


@pytest.mark.parametrize("year", YEARS)
def test_no_promotion_eligible_student_is_left_out_of_the_applicant_pool(markets, year):
    market = markets[year]["market"]
    entitled = converter.promotion_entitlements(
        converter._postrun_outcomes(markets[year]["postrun"]),
        converter.program_keys(markets[year]["capacities"], "KG"),
    )
    assert set(entitled.index) <= set(market.index)


@pytest.mark.parametrize("year", YEARS)
def test_every_market_student_gets_a_list_from_a_named_source(markets, year):
    counts = markets[year]["market"]["pref_source"].value_counts()
    expected = MARKET_EXPECTATIONS[year]
    assert int(counts.get("k_list", 0)) == expected["k_list"]
    assert int(counts.get("tk_imputed", 0)) == expected["tk_imputed"]
    assert (
        int(counts.get("feeder_only", 0)) + int(counts.get("aa_only", 0))
        == expected["no_source"]
    )
    assert int(counts.sum()) == expected["total"]


@pytest.mark.parametrize("year", YEARS)
def test_k_list_is_exactly_the_kindergarten_applicants(markets, year):
    market = markets[year]["market"]
    applicants = markets[year]["applicants"]
    assert set(market.index[market["pref_source"] == "k_list"]) == applicants
    # mr_applicant says the same thing from the other table.
    students = markets[year]["students"]
    at_grade = students.loc[students["grade"] == "KG"]
    assert set(at_grade.loc[at_grade["mr_applicant"] == 1, "studentno"]) == applicants


@pytest.mark.parametrize("year", YEARS)
def test_the_identification_rule_reproduces_the_promote_counts(markets, year):
    market = markets[year]["market"]
    applicants = markets[year]["applicants"]
    expected = PROMOTION_EXPECTATIONS[year]
    assert int(market["promote_eligible"].sum()) == expected["eligible"]
    assert (
        int(market.loc[market.index.isin(applicants), "promote_eligible"].sum())
        == expected["with_application"]
    )


def test_the_2026_27_applied_promotes_match_the_district_program_by_program(markets):
    # The tightest check available on the identification rule: the district
    # publishes TotalPromoteWithReqBeforeRun per program, and the rule has to
    # reproduce it exactly, not just in total.
    comparison = markets["2627"]["report"].missing_values[
        "promote_counts_vs_district_KG"
    ]
    applied = comparison["eligible_with_application"]
    assert applied["identified"] == applied["district"] == 341
    assert applied["programs_differing"] == 0


def test_the_identified_promotes_are_compared_with_the_district(markets):
    # Where the rule and the district disagree, the difference is recorded
    # rather than reconciled away. Three separate facts about the transfer:
    #
    # 2026-27: 1,178 identified against 1,188 seats held. The 10 are students
    #   the file held a seat for who took none anywhere.
    # 2025-26: 854 against 859, and 278 applied against 279.
    # 2024-25: the district ran no promotion -- byPromote is 0 for every
    #   student and TotalPromoteWithReqBeforeRun sums to 0 -- yet the capacity
    #   file holds 20 seats and the post-run seats 40 students with no
    #   application, only 1 of them at their current school. The
    #   identification rule is applied uniformly all the same, so it finds 607
    #   students the district's own counts do not.
    for year, district in (("2425", 20), ("2526", 859), ("2627", 1188)):
        comparison = markets[year]["report"].missing_values[
            "promote_counts_vs_district_KG"
        ]
        assert comparison["eligible"]["district"] == district
        assert (
            comparison["eligible"]["identified"]
            == PROMOTION_EXPECTATIONS[year]["eligible"]
        )


@pytest.mark.parametrize("year", YEARS)
def test_every_promotion_eligible_student_ranks_their_feeder(markets, year):
    market = markets[year]["market"]
    eligible = market.loc[market["promote_eligible"] == 1]
    for schools, programs, school, program in zip(
        eligible["r1_ranked_idschool"],
        eligible["r1_programs"],
        eligible["feeder_school"],
        eligible["feeder_program"],
        strict=True,
    ):
        listed = list(
            zip(converter._literal(schools), converter._literal(programs), strict=True)
        )
        assert (int(school), str(program)) in listed


@pytest.mark.parametrize("year", YEARS)
def test_the_parallel_preference_columns_stay_aligned(markets, year):
    # PriorityGenerator._mtb_real rejects a lottery list that does not line up
    # with the ranked programs, and filter_student_choices rejects any of the
    # four, so appending a feeder has to extend all of them.
    market = markets[year]["market"]
    columns = [
        "r1_ranked_idschool",
        "r1_programs",
        "r1_listed_ranks",
        "r1_randomnumber",
        "r1_cohortstring",
    ]
    lengths = market[columns].map(lambda value: len(converter._literal(value)))
    assert (lengths.nunique(axis=1) == 1).all()
    assert (market["num_ranked"] == lengths["r1_ranked_idschool"]).all()


def test_the_2026_27_dedupe_leaves_the_already_ranked_feeders_alone(markets):
    """Appending a feeder must not duplicate a choice the student already made.

    192 of the 341 promote-applicants already rank their feeder *school*, 122
    of them first. The preference list is of programs, though, so the append
    is a no-op only for the 117 who rank the feeder's own pathway as well; the
    other 224 lists each grow by exactly one.
    """
    market = markets["2627"]["market"]
    students = markets["2627"]["students"]
    applicants = students.loc[
        students["studentno"].isin(markets["2627"]["applicants"])
    ].set_index("studentno")
    eligible = market.loc[
        (market["promote_eligible"] == 1) & market.index.isin(applicants.index)
    ]
    assert len(eligible) == 341

    ranks_the_school = 0
    ranks_the_school_first = 0
    ranks_the_program = 0
    grew_by_one = 0
    for studentno, school, program in zip(
        eligible.index,
        eligible["feeder_school"],
        eligible["feeder_program"],
        strict=True,
    ):
        row = applicants.loc[studentno]
        submitted = list(
            zip(
                converter._literal(row["r1_ranked_idschool"]),
                converter._literal(row["r1_programs"]),
                strict=True,
            )
        )
        submitted_ranks = converter._literal(row["r1_listed_ranks"])
        if any(one == school for one, _ in submitted):
            ranks_the_school += 1
            # A student who withdrew their first choice has no rank 1 at all,
            # so "first" is the choice listed at rank 1 or nothing.
            first = next(
                (
                    one
                    for one, rank in zip(submitted, submitted_ranks, strict=True)
                    if rank == 1
                ),
                None,
            )
            ranks_the_school_first += int(first is not None and first[0] == school)
        if (int(school), str(program)) in submitted:
            ranks_the_program += 1
        grew_by_one += int(
            int(market.loc[studentno, "num_ranked"]) == len(submitted) + 1
        )

    assert ranks_the_school == 192
    assert ranks_the_school_first == 122
    assert ranks_the_program == 117
    assert grew_by_one == 341 - ranks_the_program


def test_the_2026_27_students_with_no_attendance_area_are_recorded(markets):
    # ~70 kindergarten students have no attendance-area school, or one that
    # runs no general education kindergarten. They are counted, not given a
    # placement the transfer does not state.
    market = markets["2627"]["market"]
    postrun = markets["2627"]["postrun"]
    outcomes = converter._postrun_outcomes(postrun)
    capacities = markets["2627"]["capacities"]
    kindergarten = {
        (int(school), str(program))
        for school, program in zip(
            capacities.loc[capacities["grade"] == "KG", "school_id"],
            capacities.loc[capacities["grade"] == "KG", "program_type"],
            strict=True,
        )
    }
    area = pd.to_numeric(
        outcomes.loc[list(market.index), "idSchoolAttendance"], errors="coerce"
    )
    without = sum(
        1 for value in area if pd.isna(value) or (int(value), "GE") not in kindergarten
    )
    assert without == 70
