import json

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from optimization.data.loaders import CENSUS_ROLE, CROSSWALK_ROLE
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import make_grid_problem
from optimization.visualization import (
    VisualizationArtifactStore,
    render_solution_map,
    visualize_solutions,
)


def _solution(assignment=None):
    problem = make_grid_problem(2, 2)
    if assignment is None:
        assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    return ZoneSolution(
        problem=problem,
        assignment=assignment,
        status="FEASIBLE",
        objective=2.0,
        wall_time=0.1,
        metadata={"solver": "test"},
    )


def _geometry_loader(unit):
    assert unit == "BlockGroup"
    records = []
    for idx in range(4):
        row, col = divmod(idx, 2)
        records.append(
            {
                "BlockGroup": 1000 + idx,
                "geometry": box(col, row, col + 0.9, row + 0.9),
            }
        )
    return gpd.GeoDataFrame(records, crs="EPSG:4326")


def _scenario(tmp_path, scenario_factory):
    census_path = tmp_path / "census.geojson"
    crosswalk_path = tmp_path / "crosswalk.csv"
    census_path.write_text("census-v1", encoding="utf-8")
    crosswalk_path.write_text("crosswalk-v1", encoding="utf-8")
    cache_root = tmp_path / "shared-cache"
    scenario = scenario_factory(
        sources={
            CENSUS_ROLE: {"path": str(census_path)},
            CROSSWALK_ROLE: {"path": str(crosswalk_path)},
        },
        cache_root=cache_root,
    )
    return scenario, cache_root, crosswalk_path


def test_geometry_artifact_is_manifest_validated_and_cached(
    tmp_path, scenario_factory
):
    calls = 0

    def loader(unit):
        nonlocal calls
        calls += 1
        return _geometry_loader(unit)

    solution = _solution()
    scenario, cache_root, _ = _scenario(tmp_path, scenario_factory)
    store = VisualizationArtifactStore(
        scenario,
        geometry_loader=loader,
    )

    geometry1, path1 = store.geometry_for(solution.level, solution.problem.G)
    geometry2, path2 = store.geometry_for(solution.level, solution.problem.G)

    assert calls == 1
    assert path1 == path2
    assert path1.exists()
    assert path1.name == "geometry.pkl"
    assert path1.is_relative_to(cache_root / "visualization_geometry" / "v4")
    manifest_path = path1.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 4
    assert set(manifest["sources"]["sources"]) == {CENSUS_ROLE, CROSSWALK_ROLE}
    assert len(geometry1) == 4
    assert len(geometry2) == 4


def test_geometry_cache_changes_when_crosswalk_bytes_change(
    tmp_path, scenario_factory
):
    calls = 0

    def loader(unit):
        nonlocal calls
        calls += 1
        return _geometry_loader(unit)

    solution = _solution()
    scenario, _, crosswalk_path = _scenario(tmp_path, scenario_factory)
    store = VisualizationArtifactStore(scenario, geometry_loader=loader)

    _, first_path = store.geometry_for(solution.level, solution.problem.G)
    crosswalk_path.write_text("crosswalk-version-two", encoding="utf-8")
    _, second_path = store.geometry_for(solution.level, solution.problem.G)

    assert calls == 2
    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()


def test_geometry_cache_identity_includes_scenario_filters(tmp_path, scenario_factory):
    calls = 0

    def loader(unit):
        nonlocal calls
        calls += 1
        return _geometry_loader(unit)

    solution = _solution()
    scenario, cache_root, _ = _scenario(tmp_path, scenario_factory)
    changed_filters = scenario_factory(
        sources={
            CENSUS_ROLE: {"path": str(scenario.source(CENSUS_ROLE).path)},
            CROSSWALK_ROLE: {"path": str(scenario.source(CROSSWALK_ROLE).path)},
        },
        filters={"optimization": {"include_mission_bay": False}},
        cache_root=cache_root,
    )

    _, first_path = VisualizationArtifactStore(
        scenario, geometry_loader=loader
    ).geometry_for(solution.level, solution.problem.G)
    _, second_path = VisualizationArtifactStore(
        changed_filters, geometry_loader=loader
    ).geometry_for(solution.level, solution.problem.G)

    assert calls == 2
    assert first_path != second_path


def test_corrupt_geometry_payload_is_rebuilt(tmp_path, scenario_factory):
    calls = 0

    def loader(unit):
        nonlocal calls
        calls += 1
        return _geometry_loader(unit)

    solution = _solution()
    scenario, _, _ = _scenario(tmp_path, scenario_factory)
    _, path = VisualizationArtifactStore(
        scenario, geometry_loader=loader
    ).geometry_for(solution.level, solution.problem.G)
    path.write_bytes(b"not a pickle")

    geometry, rebuilt_path = VisualizationArtifactStore(
        scenario, geometry_loader=loader
    ).geometry_for(solution.level, solution.problem.G)

    assert calls == 2
    assert rebuilt_path == path
    assert len(geometry) == 4


def test_visualize_all_stages_writes_distinct_png_artifacts(
    tmp_path, scenario_factory
):
    output_dir = tmp_path / "optimization_output"
    scenario, cache_root, _ = _scenario(tmp_path, scenario_factory)
    solutions = [
        _solution({0: 0, 1: 0, 2: 1, 3: 1}),
        _solution({0: 0, 1: 1, 2: 1, 3: 1}),
    ]

    results = visualize_solutions(
        solutions,
        output_dir=output_dir,
        stages="all",
        config=scenario,
        geometry_loader=_geometry_loader,
    )

    assert [result.stage for result in results] == [
        "stage_00_BlockGroup_0",
        "stage_01_BlockGroup_0",
    ]
    for result in results:
        assert result.geometry_artifact.exists()
        assert result.geometry_artifact.is_relative_to(
            cache_root / "visualization_geometry" / "v4"
        )
        assert len(result.figure_paths) == 1
        assert result.figure_paths[0].exists()
        assert result.figure_paths[0].suffix == ".png"
        assert result.figure_paths[0].parent == output_dir
    assert sorted(path.name for path in output_dir.glob("*.png")) == [
        "visualization_stage_00_BlockGroup_0.png",
        "visualization_stage_01_BlockGroup_0.png",
    ]
    assert not list(cache_root.rglob("*.png"))


def test_visualize_defaults_to_png_and_never_shows(
    tmp_path, monkeypatch, scenario_factory
):
    output_dir = tmp_path / "optimization_output"
    scenario, cache_root, _ = _scenario(tmp_path, scenario_factory)
    monkeypatch.setattr(
        plt,
        "show",
        lambda: (_ for _ in ()).throw(
            AssertionError("visualization should not call plt.show()")
        ),
    )
    results = visualize_solutions(
        [_solution()],
        output_dir=output_dir,
        config=scenario,
        geometry_loader=_geometry_loader,
    )

    assert results[0].geometry_artifact.exists()
    assert results[0].geometry_artifact.is_relative_to(
        cache_root / "visualization_geometry" / "v4"
    )
    assert len(results[0].figure_paths) == 1
    assert results[0].figure_paths[0].suffix == ".png"
    assert results[0].figure_paths[0].exists()
    assert results[0].figure_paths[0].parent == output_dir
    plt.close("all")


def test_render_solution_map_marks_every_graph_school(tmp_path, scenario_factory):
    solution = _solution()
    solution.problem.G.graph["school_data"] = {
        100: {"lat": 0.25, "lon": 0.75},
        200: {"lat": 1.25, "lon": 1.75},
    }
    scenario, _, _ = _scenario(tmp_path, scenario_factory)
    store = VisualizationArtifactStore(scenario, geometry_loader=_geometry_loader)
    geometry, _path = store.geometry_for(solution.level, solution.problem.G)

    fig = render_solution_map(solution, geometry, "test")

    school_markers = [text for text in fig.axes[0].texts if text.get_text() == "S"]
    assert len(school_markers) == 2
    assert {text.get_position() for text in school_markers} == {
        (0.75, 0.25),
        (1.75, 1.25),
    }
    plt.close(fig)
