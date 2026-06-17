import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

from Zone_Generation.pipeline.solution import ZoneSolution
from Zone_Generation.pipeline.tests.synthetic import make_grid_problem
from Zone_Generation.pipeline.visualization import (
    VisualizationArtifactStore,
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


def _geometry_loader(unit, is_local):
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

    def loader(unit, is_local):
        nonlocal calls
        calls += 1
        return _geometry_loader(unit, is_local)

    solution = _solution()
    store = VisualizationArtifactStore(
        is_local=False,
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
    solutions = [
        _solution({0: 0, 1: 0, 2: 1, 3: 1}),
        _solution({0: 0, 1: 1, 2: 1, 3: 1}),
    ]

    results = visualize_solutions(
        solutions,
        is_local=False,
        stages="all",
        geometry_loader=_geometry_loader,
        artifact_dir=tmp_path,
    )

    assert [result.stage for result in results] == [
        "stage_00_BlockGroup_0",
        "stage_01_BlockGroup_0",
    ]
    for result in results:
        assert result.geometry_artifact.exists()
        assert len(result.figure_paths) == 1
        assert result.figure_paths[0].exists()
        assert result.figure_paths[0].suffix == ".png"
    assert sorted(path.name for path in tmp_path.glob("*.png")) == [
        "visualization_stage_00_BlockGroup_0.png",
        "visualization_stage_01_BlockGroup_0.png",
    ]


def test_visualize_defaults_to_png_and_never_shows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        plt,
        "show",
        lambda: (_ for _ in ()).throw(
            AssertionError("visualization should not call plt.show()")
        ),
    )
    results = visualize_solutions(
        [_solution()],
        is_local=False,
        geometry_loader=_geometry_loader,
        artifact_dir=tmp_path,
    )

    assert results[0].geometry_artifact.exists()
    assert len(results[0].figure_paths) == 1
    assert results[0].figure_paths[0].suffix == ".png"
    assert results[0].figure_paths[0].exists()
    plt.close("all")
