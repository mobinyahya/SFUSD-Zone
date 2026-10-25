import math

import networkx as nx
import pytest

from optimization import run_archive
from optimization.data import conversion
from optimization.levels import LevelSpec
from optimization.problem import ZoneProblem
from optimization.progress import SolverProgressEntry
from optimization.solution import ZoneSolution
from optimization.tests.synthetic import make_grid_problem


def _base_solution(**metadata):
    problem = make_grid_problem(3, 3)
    problem.level = LevelSpec("Block", 0)
    assignment = {node: int(node % 3 >= 1) for node in problem.G.nodes}
    return ZoneSolution(
        problem=problem,
        assignment=assignment,
        status="FEASIBLE",
        objective=4.0,
        wall_time=0.5,
        metadata=dict(metadata),
    )


def _coarse_solution():
    """Three aggregated nodes over the grid's nine blocks, one left unassigned."""
    base = make_grid_problem(3, 3).G
    G = nx.path_graph(3)
    for node in G.nodes:
        G.nodes[node].update(
            {
                key: value
                for key, value in base.nodes[3 * node].items()
                if key != "area_id"
            }
        )
        G.nodes[node]["block_ids"] = [1000 + 3 * node + col for col in range(3)]
    G.graph.update(base.graph)
    problem = ZoneProblem(
        G=G,
        level=LevelSpec("Block", 1),
        centroids=[0, 2],
        centroid_school_ids=[100, 200],
        frl_dev=1.0,
        racial_dev=1.0,
        overage=5.0,
        shortage=0.0,
    )
    return ZoneSolution(
        problem=problem,
        assignment={0: 0, 2: 1},
        status="OPTIMAL",
        objective=float("inf"),
        wall_time=1.0,
        metadata={"partial_assignment": True, "gap": float("nan")},
    )


def test_stages_round_trip_through_one_archive(tmp_path):
    coarse = _coarse_solution()
    base = _base_solution(capacity_reference={str(i): i / 7 for i in range(500)})
    config = {"levels": ["Block_1", "Block_0"], "max_distance": float("inf")}

    run_archive.save_zonings(
        tmp_path,
        [coarse, base],
        ["stage_00_Block_1", "stage_01_Block_0"],
        config=config,
    )

    assert sorted(p.name for p in tmp_path.iterdir()) == [run_archive.ZONINGS_FILENAME]
    archive = run_archive.read_zonings(tmp_path)
    assert archive["config"] == {
        "levels": ["Block_1", "Block_0"],
        "max_distance": "Infinity",
    }
    # Both stages share the Block unit's one id list.
    assert list(archive["area_ids"]) == ["Block"]
    first, second = archive["stages"]
    assert run_archive.node_assignment(first) == coarse.assignment
    assert run_archive.area_assignment(archive, first) == coarse.area_assignment()
    assert run_archive.node_assignment(second) == base.assignment
    assert run_archive.area_assignment(archive, second) == base.area_assignment()
    assert first["objective"] == math.inf
    assert math.isnan(first["metadata"]["gap"])
    assert second["metadata"]["capacity_reference"]["7"] == pytest.approx(1.0)
    assert first["contiguous"] is True
    # No stage is final until evaluation picks one; the last stands in.
    assert archive["final_stage"] is None
    assert run_archive.get_stage(archive)["name"] == "stage_01_Block_0"

    assert run_archive.set_final_stage(tmp_path, "stage_00_Block_1")
    assert run_archive.get_stage(run_archive.read_zonings(tmp_path))["index"] == 0
    with pytest.raises(ValueError, match="not in"):
        run_archive.set_final_stage(tmp_path, "stage_09_Block_0")


def test_archive_writes_are_byte_stable(tmp_path):
    solution = _base_solution()
    run_archive.save_zonings(tmp_path / "a", [solution], ["s"])
    run_archive.save_zonings(tmp_path / "b", [solution], ["s"])
    a = (tmp_path / "a" / run_archive.ZONINGS_FILENAME).read_bytes()
    assert a == (tmp_path / "b" / run_archive.ZONINGS_FILENAME).read_bytes()


def test_solver_progress_is_one_file_beside_the_archive(tmp_path):
    solution = _base_solution(solver="cp_int")
    nodes = list(solution.problem.nodes)
    solution.solver_progress = [
        SolverProgressEntry(9.0, 0.1, tuple(0 for _ in nodes)),
        SolverProgressEntry(4.0, 0.4, tuple(solution.assignment[n] for n in nodes), 2),
    ]

    run_archive.save_zonings(tmp_path, [solution], ["stage_00_Block_0"])

    assert solution.metadata["solver_progress_path"] == "solver_progress.json.xz"
    assert solution.metadata["solver_progress_count"] == 2
    stage = run_archive.get_stage(run_archive.read_zonings(tmp_path))
    assert stage["metadata"]["solver_progress_id"] == "Block_0_cp_int"
    progress = run_archive.read_solver_progress(tmp_path)
    (record,) = progress["stages"]
    assert record["stage"] == "stage_00_Block_0"
    first, last = record["entries"]
    assert first["node_zones"] == [0] * len(nodes)
    assert last["iteration"] == 2 and "iteration" not in first
    ids = progress["area_ids"]["Block"]
    assert dict(zip(ids, last["area_zones"])) == solution.area_assignment()

    # A later save of a run without snapshots removes the stale file.
    solution.solver_progress = []
    run_archive.save_zonings(tmp_path, [solution], ["stage_00_Block_0"])
    assert not (tmp_path / run_archive.SOLVER_PROGRESS_FILENAME).exists()


def test_summary_metadata_keeps_scalars_and_small_containers():
    metadata = {
        "solver": "cp_int",
        "mid_welfare": 1.5,
        "centroid_school_ids": list(range(60)),
        "capacity_reference": {str(i): float(i) for i in range(1000)},
    }
    summary = run_archive.summary_metadata(metadata)
    assert summary == {
        "solver": "cp_int",
        "mid_welfare": 1.5,
        "centroid_school_ids": list(range(60)),
        "archived_metadata_keys": ["capacity_reference"],
    }
    assert run_archive.summary_metadata({"a": 1}) == {"a": 1}


def test_archive_is_a_hint_zoning_source(tmp_path):
    coarse, base = _coarse_solution(), _base_solution()
    run_archive.save_zonings(tmp_path, [coarse, base], ["coarse", "fine"])
    path = str(tmp_path / run_archive.ZONINGS_FILENAME)

    assert conversion.load_area_assignment(path) == base.area_assignment()
    assert conversion.area_file_level(path, "BlockGroup_0") == LevelSpec("Block", 0)
    run_archive.set_final_stage(tmp_path, "coarse")
    assert conversion.load_area_assignment(path) == coarse.area_assignment()
    assert conversion.area_file_level(path, "BlockGroup_0") == LevelSpec("Block", 1)


def test_reader_rejects_other_files(tmp_path):
    run_archive.save_zonings(tmp_path, [_base_solution()], ["s"])
    (tmp_path / "x.json.xz").write_bytes(
        (tmp_path / run_archive.ZONINGS_FILENAME).read_bytes()
    )
    with pytest.raises(ValueError, match="solver-progress"):
        run_archive.read_solver_progress(tmp_path / "x.json.xz")
