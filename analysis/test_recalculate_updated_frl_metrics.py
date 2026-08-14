import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import recalculate_updated_frl_metrics as recalculate  # noqa: E402


def block_geometry(*block_ids: int) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"census_block_2020": [str(block_id) for block_id in block_ids]},
        geometry=[box(index, 0, index + 1, 1) for index in range(len(block_ids))],
        crs="EPSG:4326",
    )


def test_default_soft_matches_root_matches_source_report_policy():
    assert recalculate.DEFAULT_SOFT_MATCHES_ROOT.name == "zones+soft_reserves_05frl_25"


def test_enrich_student_frl_overrides_rate_and_uses_legacy_fallback():
    students = pd.DataFrame(
        {
            "census_block": [999, 999, 999, 999],
            "latitude": [0.5, 0.5, 0.5, None],
            "longitude": [0.5, 1.5, 2.5, None],
            "freelunch_prob": [0.1, 0.2, None, 0.4],
            "reducedlunch_prob": [0.05, 0.1, None, 0.2],
        }
    )
    lookup = pd.Series({"100": 0.8, "200": float("nan")})

    result = recalculate.enrich_student_frl(
        students, lookup, block_geometry(100, 200, 300)
    )

    assert result["freelunch_prob"].tolist() == pytest.approx([0.8, 0.3, 0, 0.6])
    assert result["reducedlunch_prob"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert result["census_block_2020"].iloc[:3].tolist() == ["100", "200", "300"]
    assert pd.isna(result["census_block_2020"].iloc[3])
    assert students["freelunch_prob"].tolist()[:2] == [0.1, 0.2]


def test_load_frl_lookup_preserves_blank_rates_for_fallback(tmp_path):
    path = tmp_path / "frl.csv"
    path.write_text(
        "BlockID,FRL Rate\n60750000000001,0.75\n60750000000002,\n",
        encoding="utf-8",
    )

    lookup = recalculate.load_frl_lookup(path)

    assert lookup.loc["60750000000001"] == 0.75
    assert pd.isna(lookup.loc["60750000000002"])


def test_fallback_block_report_lists_absent_blank_and_missing_blocks():
    students = pd.DataFrame(
        {
            "grade": ["KG", "KG", "KG", "KG", "01"],
            "latitude": [0.5, 0.5, 0.5, None, 0.5],
            "longitude": [0.5, 1.5, 2.5, None, 3.5],
            "freelunch_prob": [0.1, 0.2, 0.3, 0.4, 0.5],
            "reducedlunch_prob": [0.05, 0.1, 0.15, 0.2, 0.25],
        }
    )
    lookup = pd.Series({"100": 0.8, "200": float("nan")})

    result = recalculate.fallback_block_report(
        students, lookup, block_geometry(100, 200, 300, 400)
    )

    assert result["census_block_2020"].iloc[:2].tolist() == ["200", "300"]
    assert pd.isna(result["census_block_2020"].iloc[2])
    assert result["frl_fallback_reason"].tolist() == [
        "blank updated FRL rate",
        "absent from updated lookup",
        "missing student coordinates",
    ]
    assert result["student_count"].tolist() == [1, 1, 1]
    assert result["legacy_frl"].tolist() == pytest.approx([0.3, 0.45, 0.6])


def test_zone_population_metrics_use_strict_ascending_zone_order():
    zone = recalculate.ZoneDefinition(
        unit="block_group",
        area_to_zone={"100": 1, "200": 0},
        zone_count=2,
    )
    students = pd.DataFrame(
        {
            "grade": ["KG", "KG", "KG", "01", "KG"],
            "census_blockgroup": [100, "100.0", 200, 200, None],
            "freelunch_prob": [0.8, 0.4, 0.2, 1.0, 0.9],
            "r1_programs": ["['GE', 'SE']", "[]", "['GE']", "['GE']", "['GE']"],
            "r1_ranked_idschool": ["[10, 11]", "[]", "[20]", "[20]", "[30]"],
        }
    )

    result = recalculate.zone_population_metrics(students, "KG", zone)

    assert result["district_frl"] == pytest.approx(1.4 / 3)
    assert result["frl_by_zone"] == pytest.approx([0.2, 0.6])
    assert result["frl_devs"] == pytest.approx([0.2 - 1.4 / 3, 0.6 - 1.4 / 3])
    assert result["frl_max_dev"] == pytest.approx(abs(0.2 - 1.4 / 3))
    assert result["ge_students"] == pytest.approx([1.0, 0.5])
    assert result["applicants"] == [1, 1]


def test_assignment_lists_use_program_capacity_and_attendance_ge_only():
    zone = recalculate.ZoneDefinition(
        unit="block_group",
        area_to_zone={"100": 0, "200": 1},
        zone_count=2,
    )
    students = pd.DataFrame(
        {
            "studentno": [1, 2],
            "census_blockgroup": [100, 200],
        }
    )
    assignment = pd.DataFrame(
        {
            "studentno": [1, 2],
            "programno": [1, 0],
            "programcodes": ["10-GE-KG", None],
        }
    )
    programs = pd.DataFrame(
        {
            "program_id": ["10-GE-KG", "10-SE-KG", "20-GE-KG", "30-GE-KG"],
            "school_id": [10, 10, 20, 30],
            "program_type": ["GE", "SE", "GE", "GE"],
            "capacity": [2, 1, 1, 4],
        }
    )
    schools = pd.DataFrame(
        {
            "school_id": [10, 20, 30],
            "category": ["Attendance", "Citywide", "Attendance"],
            "BlockGroup": [100, 200, 200],
        }
    )

    empty_school, unassigned, empty_ge = recalculate.assignment_list_metrics(
        assignment,
        students,
        programs,
        schools,
        zone,
    )

    assert empty_school == [2.0, 1.0, 4.0]
    assert unassigned == [0.0, 1.0]
    assert empty_ge == [1.0, 4.0]
    assert recalculate.ge_seats_by_zone(programs, schools, zone) == [2.0, 4.0]
    assert recalculate.ge_seat_disparity_by_zone([1.0, 2.0], [2.0, 4.0], zone) == [
        0.5,
        0.5,
    ]


def test_mean_lists_averages_elementwise_and_rejects_shape_changes():
    assert recalculate.mean_lists([[1, 3], [3, 5]]) == [2.0, 4.0]
    with pytest.raises(ValueError, match="lengths differ"):
        recalculate.mean_lists([[1], [1, 2]])


def test_non_zone_configuration_emits_empty_list_placeholders(monkeypatch, tmp_path):
    students = tmp_path / "students.csv"
    programs = tmp_path / "programs.csv"
    schools = tmp_path / "schools.csv"
    assignment = tmp_path / "assignment.csv"
    lookup = tmp_path / "lookup.csv"
    config = tmp_path / "config.yaml"

    pd.DataFrame(
        {
            "studentno": [1],
            "latitude": [0.5],
            "longitude": [0.5],
            "freelunch_prob": [0.1],
            "reducedlunch_prob": [0.0],
        }
    ).to_csv(students, index=False)
    pd.DataFrame({"program_id": []}).to_csv(programs, index=False)
    pd.DataFrame({"school_id": []}).to_csv(schools, index=False)
    pd.DataFrame({"studentno": [1]}).to_csv(assignment, index=False)
    lookup.write_text("BlockID,FRL Rate\n100,0.8\n", encoding="utf-8")
    config.write_text(
        "\n".join(
            [
                "grade: KG",
                "year: 23",
                "zone-building-blocks: attendance_area",
                "paths:",
                f"  student-data: {students}",
                f"  program-data: {programs}",
                f"  school-data: {schools}",
            ]
        ),
        encoding="utf-8",
    )

    first_round_values = []

    class FakeEvaluator:
        def __init__(self, *args, **kwargs):
            first_round_values.append(kwargs["first_round"])

        def eval_assignment_full(self):
            return pd.Series({"base metric": 1.0})

    monkeypatch.setattr(recalculate, "MatchEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        recalculate,
        "load_2020_block_geometry",
        lambda _: block_geometry(100),
    )
    task = recalculate.ConfigurationTask(
        label="status_quo",
        config_path=str(config),
        assignment_paths=(str(assignment),) * recalculate.ITERATION_COUNT,
        updated_frl_path=str(lookup),
        block_geometry_path=str(tmp_path / "blocks.zip"),
        all_students_path=str(students),
        new_ctip_path=None,
        first_round=False,
    )

    _, result = recalculate.evaluate_configuration(task)

    assert result["base metric"] == 1.0
    assert [json.loads(result[name]) for name in recalculate.LIST_METRICS] == [
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    assert pd.isna(result[recalculate.FRL_MAX_DEV_METRIC])
    assert first_round_values == [False] * recalculate.ITERATION_COUNT


def test_output_path_adds_updated_frl_before_new_timestamp(tmp_path):
    source = tmp_path / "report_20260723T192723268919Z.csv"

    result = recalculate.output_path_for(source, "20260724T000000000000Z")

    assert result.name == "report_updated_frl_20260724T000000000000Z.csv"


def test_build_tasks_adds_preference_label_suffix(tmp_path):
    root = tmp_path / "matches"
    policy_root = root / "policy"
    policy_root.mkdir(parents=True)
    (policy_root / "policy_config.generated.yaml").write_text("grade: KG\n")
    for iteration in range(recalculate.ITERATION_COUNT):
        (policy_root / f"assignment_iteration{iteration}.csv").touch()

    tasks = recalculate.build_tasks(
        ["policy"],
        root,
        "zone",
        tmp_path / "frl.csv",
        tmp_path / "blocks.zip",
        tmp_path / "students.csv",
        None,
        "__real_preferences",
    )

    assert [task.label for task in tasks] == ["policy__real_preferences"]


@pytest.mark.parametrize(
    ("population", "expected"),
    [("first_round", True), ("all_rounds", False)],
)
def test_build_tasks_reads_evaluation_population(tmp_path, population, expected):
    root = tmp_path / "matches"
    policy_root = root / "policy"
    policy_root.mkdir(parents=True)
    (policy_root / "policy_config.generated.yaml").write_text(
        f"evaluation-population: {population}\n",
        encoding="utf-8",
    )
    for iteration in range(recalculate.ITERATION_COUNT):
        (policy_root / f"assignment_iteration{iteration}.csv").touch()

    tasks = recalculate.build_tasks(
        ["policy"],
        root,
        "zone",
        tmp_path / "frl.csv",
        tmp_path / "blocks.zip",
        tmp_path / "students.csv",
        None,
    )

    assert tasks[0].first_round is expected
