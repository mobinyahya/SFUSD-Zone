import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

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


def test_geometry_artifact_is_cached(tmp_path):
    calls = 0

    def loader(unit):
        nonlocal calls
        calls += 1
        return _geometry_loader(unit)

    solution = _solution()
    store = VisualizationArtifactStore(
        artifact_dir=tmp_path,
        geometry_loader=loader,
    )

    geometry1, path1 = store.geometry_for(solution.level, solution.problem.G)
    geometry2, path2 = store.geometry_for(solution.level, solution.problem.G)

    assert calls == 1
    assert path1 == path2
    assert path1.exists()
    assert path1.with_suffix(".json").exists()
    assert len(geometry1) == 4
    assert len(geometry2) == 4


def test_visualize_all_stages_writes_distinct_png_artifacts(tmp_path):
    output_dir = tmp_path / "optimization_output"
    artifact_dir = tmp_path / "visualization_artifacts"
    solutions = [
        _solution({0: 0, 1: 0, 2: 1, 3: 1}),
        _solution({0: 0, 1: 1, 2: 1, 3: 1}),
    ]

    results = visualize_solutions(
        solutions,
        output_dir=output_dir,
        stages="all",
        geometry_loader=_geometry_loader,
        artifact_dir=artifact_dir,
    )

    assert [result.stage for result in results] == [
        "stage_00_BlockGroup_0",
        "stage_01_BlockGroup_0",
    ]
    for result in results:
        assert result.geometry_artifact.exists()
        assert result.geometry_artifact.parent == artifact_dir
        assert len(result.figure_paths) == 1
        assert result.figure_paths[0].exists()
        assert result.figure_paths[0].suffix == ".png"
        assert result.figure_paths[0].parent == output_dir
    assert sorted(path.name for path in output_dir.glob("*.png")) == [
        "visualization_stage_00_BlockGroup_0.png",
        "visualization_stage_01_BlockGroup_0.png",
    ]
    assert not list(artifact_dir.glob("*.png"))


def test_visualize_defaults_to_png_and_never_shows(tmp_path, monkeypatch):
    output_dir = tmp_path / "optimization_output"
    artifact_dir = tmp_path / "visualization_artifacts"
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
        geometry_loader=_geometry_loader,
        artifact_dir=artifact_dir,
    )

    assert results[0].geometry_artifact.exists()
    assert results[0].geometry_artifact.parent == artifact_dir
    assert len(results[0].figure_paths) == 1
    assert results[0].figure_paths[0].suffix == ".png"
    assert results[0].figure_paths[0].exists()
    assert results[0].figure_paths[0].parent == output_dir
    plt.close("all")


def test_render_solution_map_marks_every_graph_school(tmp_path):
    solution = _solution()
    solution.problem.G.graph["school_data"] = {
        100: {"lat": 0.25, "lon": 0.75},
        200: {"lat": 1.25, "lon": 1.75},
    }
    store = VisualizationArtifactStore(
        artifact_dir=tmp_path,
        geometry_loader=_geometry_loader,
    )
    geometry, _path = store.geometry_for(solution.level, solution.problem.G)

    fig = render_solution_map(solution, geometry, "test")

    school_markers = [text for text in fig.axes[0].texts if text.get_text() == "🏫"]
    assert len(school_markers) == 2
    assert {text.get_position() for text in school_markers} == {
        (0.75, 0.25),
        (1.75, 1.25),
    }
    plt.close(fig)
