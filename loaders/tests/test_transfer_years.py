"""Contract tests for the 2024-25 through 2026-27 transfer school years.

These assert the registry and scenario shape only: they resolve sources and
check selector completeness without touching the shared data root, so they run
anywhere. Tests that actually read the converted CSVs belong with the
``real_data`` marker.
"""

from __future__ import annotations

import pytest
import yaml

from loaders import load_scenario
from loaders.config import BASE_CONFIG_PATH

TRANSFER_YEARS = ("2425", "2526", "2627")
PER_YEAR_SCENARIOS = tuple(f"sfusd-{year}" for year in TRANSFER_YEARS)
POOLED_SCENARIO = "sfusd-2425-2627"


@pytest.fixture(scope="module")
def base() -> dict:
    return yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("year", TRANSFER_YEARS)
def test_registry_year_offers_both_populations_and_three_grades(base, year):
    entry = base["school_years"][year]
    catalog = set(base["files"])

    optimization = entry["optimization"]["students"]
    assert set(optimization) == {"applicant", "enrolled"}
    assert optimization["applicant"] == f"optimization.students.applicant.{year}"
    assert optimization["enrolled"] == f"optimization.students.enrolled.{year}"
    assert set(optimization.values()) <= catalog

    assignment = entry["assignment"]
    assert set(assignment["students"]) == {"applicant", "enrolled"}
    assert set(assignment["students"].values()) <= catalog

    # The transfer carries requests for every grade, but only KG, 6, and 9 have
    # a district capacity table and a school-coordinate table to borrow.
    assert set(assignment["grades"]) == {"KG", "06", "09"}
    for grade, registry in assignment["grades"].items():
        for profile in registry["profiles"].values():
            assert set(profile) <= {"standard", "mission_bay"}
            for bundle in profile.values():
                assert set(bundle) <= {"programs", "programs_catalog", "schools"}
                assert set(bundle.values()) <= catalog
        # Only KG has a Mission Bay variant: Mission Bay ES is an elementary
        # school, so no middle or high school bundle needs one.
        default = registry["profiles"]["default"]
        assert ("mission_bay" in default) == (grade == "KG")


@pytest.mark.parametrize("year", TRANSFER_YEARS)
def test_registry_year_points_at_that_year_student_files(base, year):
    files = base["files"]
    assert (
        files[f"optimization.students.applicant.{year}"]["path"]
        == f"Data/Cleaned/student_{year}.csv"
    )
    assert (
        files[f"optimization.students.enrolled.{year}"]["path"]
        == f"Data/Cleaned/enrolled_{year}.csv"
    )
    # Every converted student table is tagged 2010, like the checked-in years,
    # so a 2010 run keeps its Census IDs and a 2020 run remaps from lat/lon.
    for role in (
        f"optimization.students.applicant.{year}",
        f"optimization.students.enrolled.{year}",
        f"assignment.students.{year}",
    ):
        assert files[role]["geography_vintage"] == "2010"
        assert files[role]["classification"] == "restricted"


@pytest.mark.parametrize("name", PER_YEAR_SCENARIOS + (POOLED_SCENARIO,))
def test_transfer_scenarios_resolve_every_role_both_consumers_need(name):
    scenario = load_scenario({"scenario": name, "overrides": {}}, environ={})
    assert scenario.id == name

    for role in (
        "optimization.students",
        "optimization.schools",
        "optimization.programs",
        "optimization.census",
        "optimization.crosswalk",
        "optimization.adjacency",
        "optimization.manual_edges",
        "assignment.students",
        "assignment.programs",
        "assignment.schools",
        "assignment.school_coordinates",
        "assignment.program_codes",
        "assignment.zones",
        "choice.estimate",
    ):
        assert scenario.sources(role), role


@pytest.mark.parametrize("year", TRANSFER_YEARS)
def test_per_year_scenario_selects_only_that_year(year):
    scenario = load_scenario({"scenario": f"sfusd-{year}", "overrides": {}}, environ={})
    assert list(scenario.filter("optimization", "years")) == [year]
    assert scenario.filter("assignment", "year") == year
    students = scenario.sources("optimization.students")
    assert [source.path.name for source in students] == [f"student_{year}.csv"]


def test_pooled_scenario_pools_optimization_and_assigns_on_the_newest_year():
    scenario = load_scenario({"scenario": POOLED_SCENARIO, "overrides": {}}, environ={})
    assert list(scenario.filter("optimization", "years")) == list(TRANSFER_YEARS)
    assert scenario.filter("assignment", "year") == TRANSFER_YEARS[-1]
    assert [
        source.path.name for source in scenario.sources("optimization.students")
    ] == [f"student_{year}.csv" for year in TRANSFER_YEARS]


@pytest.mark.parametrize("name", PER_YEAR_SCENARIOS + (POOLED_SCENARIO,))
def test_transfer_scenarios_select_the_single_available_round(name):
    # Each transfer holds one round of requests, so these scenarios must not
    # ask for a round the converted tables cannot have.
    scenario = load_scenario({"scenario": name, "overrides": {}}, environ={})
    assert list(scenario.filter("optimization", "rounds")) == [1]
    assert list(scenario.filter("assignment", "rounds")) == [1]


@pytest.mark.parametrize("name", PER_YEAR_SCENARIOS + (POOLED_SCENARIO,))
def test_transfer_scenarios_keep_mission_bay_consistent_across_consumers(name):
    scenario = load_scenario({"scenario": name, "overrides": {}}, environ={})
    assert scenario.filter("optimization", "include_mission_bay") is True
    assert scenario.filter("assignment", "include_mission_bay") is True
    # The zoning capacity balance and the welfare market must agree about
    # whether Mission Bay's seats exist, so the optimization program table is
    # the Mission Bay variant too.
    assignment_year = scenario.filter("assignment", "year")
    programs = scenario.sources("optimization.programs")
    assert [source.path.name for source in programs] == [
        f"programs_withMissionBay_{assignment_year}.csv"
    ]


def test_no_registry_year_nets_the_promoted_seats_out_of_capacity(base):
    # An interim post_promotion profile once held each program's capacity net
    # of the seats the run gave students who filed no request. That models a
    # market the district never ran: a promote who wins a school elsewhere
    # releases the held seat back into the same run, so the seats never leave.
    # Kindergarten capacity is now the Main Round capacity file's gross
    # TotalSeats and the promotion claim is a run-time priority instead, so
    # neither the profile nor its files may come back.
    # See analysis/data_prep/TK_PROMOTION_SPEC.md.
    for source in base["files"].values():
        assert "postPromotion" not in source["path"]
    for entry in base["school_years"].values():
        for registry in entry.get("assignment", {}).get("grades", {}).values():
            assert "post_promotion" not in registry["profiles"]


@pytest.mark.parametrize("name", PER_YEAR_SCENARIOS + (POOLED_SCENARIO,))
def test_transfer_scenarios_take_the_default_capacity_profile(name):
    scenario = load_scenario({"scenario": name, "overrides": {}}, environ={})
    assert scenario.filter("assignment", "capacity_profile") == "default"


def test_the_enrolled_population_is_registered_as_its_own_file(base):
    # enrolled_<year>.csv is the kindergarten subset of student_<year>.csv the
    # post-run seats. The two coincide in these years, because every market
    # student takes a seat, but they are distinct selections and both
    # consumers must read the same file for each.
    for year in TRANSFER_YEARS:
        entry = base["school_years"][year]
        enrolled = entry["optimization"]["students"]["enrolled"]
        assert entry["assignment"]["students"]["enrolled"] == enrolled
        assert base["files"][enrolled]["path"] == f"Data/Cleaned/enrolled_{year}.csv"
