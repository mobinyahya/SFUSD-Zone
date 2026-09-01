import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from optimization.data import graph_builder, loaders
from optimization.data.loaders import IngestConfig


def _ingest(scenario, unit="Block"):
    return IngestConfig(unit=unit, data=scenario)


def _student_frame(grades=("KG", "KG")):
    return pd.DataFrame(
        {
            "studentno": [1, 2],
            "grade": list(grades),
            "census_block": [1001, 1002],
            "census_blockgroup": [100, 100],
            "census_tract": [10, 10],
            "idschoolattendance": [10, 10],
            "enrolled_idschool": [909, 100],
            "resolved_ethnicity": ["Hispanic", "White"],
            "FRL Score": [1.0, 0.0],
            "r1_ranked_idschool": ["[]", "[100]"],
            "r1_programs": ["[]", "['GE']"],
            "r2_ranked_idschool": ["[909]", "[]"],
            "r2_programs": ["['GE']", "[]"],
            "r4_ranked_idschool": ["[200]", "[201]"],
            "r4_programs": ["['SA']", "['SA']"],
        }
    )


def _student_scenario(
    tmp_path,
    scenario_factory,
    *,
    frame=None,
    student_population="enrolled",
    frl_counts=None,
    **filter_overrides,
):
    cleaned = tmp_path / "Data" / "Cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    prefix = "enrolled" if student_population == "enrolled" else "student"
    student_path = cleaned / f"{prefix}_2122.csv"
    (frame if frame is not None else _student_frame()).to_csv(student_path, index=False)
    filters = {
        "years": ["2122"],
        "grades": ["KG"],
        "student_population": student_population,
        "rounds": [1, 2, 4],
        "special_programs": "include",
        "program_population": "GE",
    }
    filters.update(filter_overrides)
    sources = None
    if frl_counts is not None:
        frl_path = tmp_path / "frl-counts.csv"
        frl_counts.to_csv(frl_path, index=False)
        filters.update({"geography_vintage": "2020", "frl_estimate": "updated_2526"})
        sources = {
            "optimization.students": {
                "path": str(student_path),
                "geography_vintage": "2020",
            },
            "optimization.frl_estimate": {
                "path": str(frl_path),
                "geography_vintage": "2020",
            },
        }
    return scenario_factory(
        sources=sources,
        filters={"optimization": filters},
        data_root=tmp_path,
    )


def test_projected_centroids_latlon_avoids_geographic_crs_warning():
    gdf = gpd.GeoDataFrame(
        {"geometry": [box(-122.5, 37.7, -122.4, 37.8)]},
        crs="EPSG:4326",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Geometry is in a geographic CRS.*")
        centroids = loaders._projected_centroids_latlon(gdf)

    assert centroids.crs == "EPSG:4326"
    assert 37.7 < centroids.iloc[0].y < 37.8
    assert -122.5 < centroids.iloc[0].x < -122.4


def test_student_cache_uses_v8_content_addressed_layout(tmp_path, scenario_factory):
    baseline = _ingest(_student_scenario(tmp_path, scenario_factory))
    all_population = _ingest(
        _student_scenario(tmp_path, scenario_factory, program_population="All")
    )

    baseline_namespace = loaders._student_cache_namespace(baseline)
    changed_namespace = loaders._student_cache_namespace(all_population)

    assert baseline_namespace.path.parent.name == "v8"
    assert baseline_namespace.path.parent.parent.name == "students"
    assert loaders._student_cache_path(baseline).endswith("/students.csv")
    assert baseline_namespace.key != changed_namespace.key


def test_true_frl_counts_feed_optimization_with_student_file_fallback(
    tmp_path, scenario_factory
):
    counts = pd.DataFrame(
        {
            "BlockID": [1001, 1002],
            "Not FRL": [1, 0],
            "FRLunch": [3, 0],
            "Students": [4, 0],
            "FRL Rate": [0.8, None],
        }
    )
    cfg = _ingest(_student_scenario(tmp_path, scenario_factory, frl_counts=counts))

    students = loaders.load_students(cfg)

    assert students["FRL"].tolist() == pytest.approx([0.75, 0.0])
    assert loaders.student_source_roles(cfg) == [
        "optimization.students",
        "optimization.frl_estimate",
    ]


def test_student_cache_identity_includes_selected_frl_count_contents(
    tmp_path, scenario_factory
):
    first_counts = pd.DataFrame(
        {
            "BlockID": [1001],
            "Not FRL": [1],
            "FRLunch": [3],
            "Students": [4],
        }
    )
    first = _ingest(
        _student_scenario(tmp_path, scenario_factory, frl_counts=first_counts)
    )
    first_key = loaders._student_cache_namespace(first).key

    second_counts = first_counts.assign(FRLunch=2, **{"Not FRL": 2})
    second = _ingest(
        _student_scenario(tmp_path, scenario_factory, frl_counts=second_counts)
    )

    assert loaders._student_cache_namespace(second).key != first_key


def test_selected_rounds_keep_r2_only_student_and_do_not_inflate_program_types(
    tmp_path, scenario_factory, monkeypatch
):
    cfg = _ingest(_student_scenario(tmp_path, scenario_factory))
    selected = {}
    original_filter = loaders._filter_to_population

    def capture_selected_programs(frame, program_population, year):
        selected["participating_programs"] = frame["participating_programs"].tolist()
        selected["program_types"] = frame["program_types"].map(list).tolist()
        return original_filter(frame, program_population, year)

    monkeypatch.setattr(loaders, "_filter_to_population", capture_selected_programs)

    students = loaders.load_students(cfg)

    assert students["studentno"].tolist() == [1, 2]
    assert students["ge_students"].tolist() == [1.0, 1.0]
    assert selected == {
        "participating_programs": [["GE"], ["GE"]],
        "program_types": [["GE"], ["GE"]],
    }
    assert loaders._student_cache_namespace(cfg).payload_path("students.csv").exists()

    special_population = _ingest(
        _student_scenario(tmp_path, scenario_factory, program_population="SA")
    )
    assert loaders.load_students(special_population).empty


def test_grade_list_is_combined_in_area_population(tmp_path, scenario_factory):
    scenario = _student_scenario(
        tmp_path,
        scenario_factory,
        frame=_student_frame(("KG", "01")),
        grades=["KG", "01"],
    )
    cfg = _ingest(scenario)

    students = loaders.load_students(cfg)
    aggregated = loaders._aggregate_students(students, cfg)

    assert students["studentno"].tolist() == [1, 2]
    assert aggregated["ge_students"].sum() == 2.0


def test_outside_district_students_are_ignored_by_default(tmp_path, scenario_factory):
    frame = _student_frame()
    frame.loc[0, ["census_block", "census_blockgroup", "census_tract"]] = pd.NA
    cfg = _ingest(_student_scenario(tmp_path, scenario_factory, frame=frame))

    students = loaders.load_students(cfg)

    assert students["studentno"].tolist() == [2]


def test_included_outside_district_students_prevent_graph_construction(
    tmp_path, scenario_factory
):
    frame = _student_frame()
    frame.loc[0, ["census_block", "census_blockgroup", "census_tract"]] = pd.NA
    cfg = _ingest(
        _student_scenario(
            tmp_path,
            scenario_factory,
            frame=frame,
            outside_district_students="include",
        )
    )

    students = loaders.load_students(cfg)

    assert students["studentno"].tolist() == [1, 2]
    assert pd.isna(students.loc[students["studentno"] == 1, "Block"]).all()
    with pytest.raises(
        ValueError,
        match="Cannot construct a Block graph with 1 included students",
    ):
        loaders.load_area_table(cfg)


def test_compatible_source_duplicates_are_counted_once(tmp_path, scenario_factory):
    frame = _student_frame()
    frame["enrolled_pathway"] = [pd.NA, "GE"]
    duplicate = frame.iloc[[0]].copy()
    duplicate["enrolled_pathway"] = "GE"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    scenario = _student_scenario(
        tmp_path,
        scenario_factory,
        frame=frame,
    )

    students = loaders.load_students(_ingest(scenario))

    assert students["studentno"].tolist() == [1, 2]


def test_conflicting_source_duplicate_identities_are_rejected(
    tmp_path, scenario_factory
):
    frame = _student_frame()
    conflict = frame.iloc[[0]].copy()
    conflict["census_block"] = 9999
    frame = pd.concat([frame, conflict], ignore_index=True)
    scenario = _student_scenario(
        tmp_path,
        scenario_factory,
        frame=frame,
    )

    with pytest.raises(ValueError, match="duplicate studentno identities"):
        loaders.load_students(_ingest(scenario))


@pytest.mark.parametrize("student_population", ["applicant", "enrolled"])
def test_registry_population_selection_flows_through_ingestion(
    tmp_path, scenario_factory, student_population
):
    scenario = _student_scenario(
        tmp_path,
        scenario_factory,
        student_population=student_population,
    )

    students = loaders.load_students(_ingest(scenario))

    assert scenario.source("optimization.students").catalog_id == (
        f"optimization.students.{student_population}.2122"
    )
    assert students["studentno"].tolist() == [1, 2]
    assert students["ge_students"].tolist() == [1.0, 1.0]


def test_special_program_mode_changes_population_and_cache_identity(
    tmp_path, scenario_factory
):
    included = _ingest(_student_scenario(tmp_path, scenario_factory))
    excluded = _ingest(
        _student_scenario(
            tmp_path,
            scenario_factory,
            special_programs="exclude_any_special",
        )
    )

    included_students = loaders.load_students(included)
    excluded_students = loaders.load_students(excluded)

    assert included_students["all_prog_students"].sum() == 2
    assert excluded_students["all_prog_students"].sum() == 0
    assert loaders._student_cache_namespace(included).key != (
        loaders._student_cache_namespace(excluded).key
    )


def test_student_sources_must_match_configured_year_count_and_order(
    tmp_path, scenario_factory
):
    first = tmp_path / "enrolled_2122.csv"
    second = tmp_path / "enrolled_2223.csv"
    unlabeled = tmp_path / "students.csv"
    _student_frame().to_csv(first, index=False)
    _student_frame().to_csv(second, index=False)
    _student_frame().to_csv(unlabeled, index=False)
    count_mismatch = scenario_factory(
        sources={"optimization.students": [{"path": str(first)}]},
        filters={"optimization": {"years": ["2122", "2223"]}},
    )
    reversed_sources = scenario_factory(
        sources={
            "optimization.students": [
                {"path": str(second)},
                {"path": str(first)},
            ]
        },
        filters={"optimization": {"years": ["2122", "2223"]}},
    )
    aligned_sources = scenario_factory(
        sources={
            "optimization.students": [
                {"path": str(first)},
                {"path": str(second)},
            ]
        },
        filters={"optimization": {"years": ["2122", "2223"]}},
    )
    unlabeled_source = scenario_factory(
        sources={"optimization.students": [{"path": str(unlabeled)}]},
        filters={"optimization": {"years": ["2122"]}},
    )

    with pytest.raises(ValueError, match="1 sources for 2 configured years"):
        loaders.load_students(_ingest(count_mismatch))
    with pytest.raises(ValueError, match="does not align"):
        loaders.load_students(_ingest(reversed_sources))
    with pytest.raises(ValueError, match=r"found school years \[\]"):
        loaders.load_students(_ingest(unlabeled_source))
    assert [
        (year, source.path.name)
        for year, source in loaders._student_sources_by_year(_ingest(aligned_sources))
    ] == [("2122", "enrolled_2122.csv"), ("2223", "enrolled_2223.csv")]


def _write_school_sources(tmp_path, schools, capacities, *, program_capacities=None):
    school_path = tmp_path / "schools.csv"
    program_path = tmp_path / "programs.csv"
    capacity_path = tmp_path / "capacities.csv"
    schools.to_csv(school_path, index=False)
    source_capacities = (
        list(program_capacities)
        if program_capacities is not None
        else capacities["Scenario_A_Capacity"].tolist()
    )
    pd.DataFrame(
        {
            "program_id": [
                f"{school}-{program}-KG"
                for school, program in zip(
                    capacities["SchNum"], capacities["PathwayCode"]
                )
            ],
            "school_id": capacities["SchNum"],
            "program_type": capacities["PathwayCode"],
            "capacity": source_capacities,
        }
    ).to_csv(program_path, index=False)
    capacities.to_csv(capacity_path, index=False)
    return {
        "optimization.schools": {"path": str(school_path)},
        "optimization.programs": {"path": str(program_path)},
        "optimization.capacity": {"path": str(capacity_path)},
    }


def test_all_program_school_loading_respects_citywide_scenario_filter(
    tmp_path, scenario_factory
):
    schools = pd.DataFrame(
        {
            "school_id": [100, 618],
            "category": ["Attendance", "Citywide"],
            "Block": [1000, 2000],
        }
    )
    capacities = pd.DataFrame(
        {
            "SchNum": [100, 618],
            "PathwayCode": ["GE", "GE"],
            "Scenario_A_Capacity": [10, 10],
        }
    )
    sources = _write_school_sources(tmp_path, schools, capacities)
    include_citywide = scenario_factory(
        sources=sources,
        filters={
            "optimization": {
                "program_population": "All",
                "capacity_scenario": "A",
                "include_citywide": True,
                "include_k8": False,
            }
        },
    )
    exclude_citywide = scenario_factory(
        sources=sources,
        filters={
            "optimization": {
                "program_population": "All",
                "capacity_scenario": "A",
                "include_citywide": False,
                "include_k8": False,
            }
        },
    )
    ge = scenario_factory(
        sources=sources,
        filters={
            "optimization": {
                "program_population": "GE",
                "capacity_scenario": "A",
                "include_k8": False,
            }
        },
    )

    assert set(loaders.load_schools(_ingest(include_citywide))["school_id"]) == {
        100,
        618,
    }
    assert set(loaders.load_schools(_ingest(exclude_citywide))["school_id"]) == {100}
    assert set(loaders.load_schools(_ingest(ge))["school_id"]) == {100}


def test_school_loading_uses_program_capacities_unless_scenario_is_selected(
    tmp_path, scenario_factory
):
    schools = pd.DataFrame(
        {"school_id": [100], "category": ["Attendance"], "Block": [1000]}
    )
    capacities = pd.DataFrame(
        {
            "SchNum": [100],
            "PathwayCode": ["GE"],
            "Scenario_A_Capacity": [19],
        }
    )
    sources = _write_school_sources(
        tmp_path, schools, capacities, program_capacities=[7]
    )
    programs = scenario_factory(sources=sources)
    scenario = scenario_factory(
        sources=sources,
        filters={"optimization": {"capacity_scenario": "A"}},
    )

    assert loaders.load_schools(_ingest(programs))["ge_capacity"].tolist() == [7]
    assert loaders.load_schools(_ingest(scenario))["ge_capacity"].tolist() == [19]


def test_legacy_alias_canonically_selects_school_999_everywhere(
    tmp_path, scenario_factory
):
    schools = pd.DataFrame(
        {
            "school_id": [909, 999],
            "category": ["Attendance", "Attendance"],
            "Block": [1001, 1002],
            "BlockGroup": [100, 100],
            "lat": [37.1, 37.2],
            "lon": [-122.1, -122.2],
        }
    )
    capacities = pd.DataFrame(
        {
            "SchNum": [909, 999],
            "PathwayCode": ["GE", "GE"],
            "Scenario_A_Capacity": [10, 20],
        }
    )
    centroids = tmp_path / "centroids.yaml"
    centroids.write_text("test-centroids: [909]\n", encoding="utf-8")
    sources = _write_school_sources(tmp_path, schools, capacities)
    sources["optimization.centroids"] = {"path": str(centroids)}
    scenario = scenario_factory(
        sources=sources,
        filters={"optimization": {"capacity_scenario": "A"}},
    )
    cfg = _ingest(scenario)

    loaded_schools = loaders.load_schools(cfg)
    locations = loaders.load_school_locations(cfg)
    coordinates = loaders.load_school_coordinates(scenario)
    graph_school_data = graph_builder._school_data(cfg)

    assert loaded_schools[["school_id", "Block", "ge_capacity"]].to_dict("records") == [
        {"school_id": 999, "Block": 1002, "ge_capacity": 20}
    ]
    assert locations.to_dict("records") == [{"school_id": 999, "Block": 1002}]
    assert coordinates.to_dict("records") == [
        {"school_id": 999, "lat": 37.2, "lon": -122.2}
    ]
    assert set(graph_school_data) == {999}
    assert loaders.load_centroid_schools("test-centroids", scenario) == [999]


def test_load_census_shapefile_enriches_geographic_ids(tmp_path, scenario_factory):
    source = gpd.GeoDataFrame(
        {
            "geoid10": [1001, 1002],
            "geometry": [Point(-122.4, 37.7), Point(-122.5, 37.8)],
        },
        crs="EPSG:4326",
    )
    shape_path = tmp_path / "areas.shp"
    source.to_file(shape_path)
    crosswalk_path = tmp_path / "crosswalk.csv"
    crosswalk = pd.DataFrame(
        {"Block": [1001, 1002], "BlockGroup": [100, 100], "Tract": [10, 10]}
    )
    crosswalk.to_csv(crosswalk_path, index=False)
    scenario = scenario_factory(
        sources={
            "optimization.census": {"path": str(shape_path)},
            "optimization.crosswalk": {"path": str(crosswalk_path)},
        }
    )
    census = loaders.load_census_shapefile("BlockGroup", scenario)

    assert list(census["BlockGroup"]) == [100]


def _distance_scenario(tmp_path, scenario_factory, unit):
    census = tmp_path / f"{unit}.shp"
    census.write_bytes(b"synthetic geometry")
    crosswalk = tmp_path / f"{unit}-crosswalk.csv"
    pd.DataFrame(
        {
            "Block": [100, 200, 300],
            "BlockGroup": [100, 200, 300],
        }
    ).to_csv(crosswalk, index=False)
    schools = tmp_path / f"{unit}-schools.csv"
    pd.DataFrame(
        {
            "school_id": [1000],
            "Block": [100],
            "BlockGroup": [100],
        }
    ).to_csv(schools, index=False)
    return scenario_factory(
        sources={
            "optimization.census": {"path": str(census)},
            "optimization.crosswalk": {"path": str(crosswalk)},
            "optimization.schools": {"path": str(schools)},
        }
    )


def test_blockgroup_distances_use_complete_v3_cache(
    tmp_path, monkeypatch, scenario_factory
):
    locations = pd.DataFrame(
        {"Lat": [37.75, 37.80], "Lon": [-122.40, -122.45]},
        index=pd.Index([100, 200], name="BlockGroup"),
    )
    cfg = _ingest(
        _distance_scenario(tmp_path, scenario_factory, "BlockGroup"),
        "BlockGroup",
    )
    monkeypatch.setattr(loaders, "load_area_latlon", lambda _: locations)

    distances = loaders.load_distance_dict(cfg, {100: 4, 200: 9})
    namespace = loaders._distance_cache_namespace(cfg, [100, 200])

    assert distances[4][4] == 0.0
    assert distances[4][9] == pytest.approx(distances[9][4])
    assert distances[4][9] > 0
    assert namespace.path.parent.name == "v3"
    assert namespace.payload_path("distances.csv").exists()


def test_block_distances_cache_only_raw_school_rows(
    tmp_path, monkeypatch, scenario_factory
):
    locations = pd.DataFrame(
        {"Lat": [0.0, 0.0, 0.0], "Lon": [0.0, 45.0, 90.0]},
        index=pd.Index([100, 200, 300], name="Block"),
    )
    cfg = _ingest(_distance_scenario(tmp_path, scenario_factory, "Block"))
    monkeypatch.setattr(loaders, "load_area_latlon", lambda _: locations)

    distances = loaders.load_distance_dict(cfg, {100: 4, 200: 9, 300: 12})
    namespace = loaders._distance_cache_namespace(cfg, [100, 200, 300])
    matrix = namespace.load_dataframe("distances.csv", index_col="Block")

    assert list(matrix.index) == [100]
    assert list(matrix.columns) == ["100", "200", "300"]
    assert set(distances[4]) == {4, 9, 12}
    assert set(distances[9]) == {4}
    assert set(distances[12]) == {4}


def test_distance_cache_validates_destinations_and_used_source_rows(
    tmp_path, scenario_factory
):
    cfg = _ingest(
        _distance_scenario(tmp_path, scenario_factory, "BlockGroup"),
        "BlockGroup",
    )
    namespace = loaders._distance_cache_namespace(cfg, [100, 200])
    namespace.save_dataframe(
        "distances.csv",
        pd.DataFrame([[0.0]], index=pd.Index([100], name="BlockGroup"), columns=[100]),
        index=True,
    )

    with pytest.raises(ValueError, match="missing BlockGroup IDs"):
        loaders.load_distance_dict(cfg, {100: 4, 200: 9})

    namespace.save_dataframe(
        "distances.csv",
        pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.Index([999], name="BlockGroup"),
            columns=[100, 200],
        ),
        index=True,
    )
    with pytest.raises(ValueError, match="no BlockGroup rows used"):
        loaders.load_distance_dict(cfg, {100: 4, 200: 9})


def test_neighbors_use_unit_specific_scenario_role(tmp_path, scenario_factory):
    block = tmp_path / "block-adjacency.csv"
    blockgroup = tmp_path / "blockgroup-adjacency.csv"
    block.write_text("100,100,200\n200,100,200\n", encoding="utf-8")
    blockgroup.write_text("10,10\n", encoding="utf-8")
    scenario = scenario_factory(
        sources={
            "optimization.adjacency": {
                "block": {"path": str(block)},
                "blockgroup": {"path": str(blockgroup)},
            }
        }
    )

    neighbors = loaders.load_neighbors(_ingest(scenario), {100: 4, 200: 9})

    assert neighbors == {4: [9], 9: [4]}
