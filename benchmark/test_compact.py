"""Legacy one-folder-per-stage runs: reading them, and compacting them."""

import json
import os

import pytest

from benchmark.compact import compact_run, compact_runs
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_solutions,
    stage_record,
)
from optimization import run_archive
from optimization.levels import LevelSpec
from optimization.progress import SolverProgressEntry
from optimization.solution import ZoneSolution, graph_fingerprint, save_zoning_config
from optimization.tests.synthetic import FakeDataset, make_grid_problem

CONFIG = {"strategy": "recursive", "levels": ["Block_0"], "seed": 3}


def _solution(problem, assignment, **metadata):
    return ZoneSolution(
        problem=problem,
        assignment=assignment,
        status="FEASIBLE",
        objective=float(sum(assignment.values())),
        wall_time=0.25,
        metadata={"solver": "cp_int", **metadata},
    )


def _legacy_save(solution, folder, config):
    """What ``ZoneSolution.save`` wrote before the archive."""
    os.makedirs(folder, exist_ok=True)
    level = solution.level.name
    with open(os.path.join(folder, f"zone_dict_{level}.json"), "w") as f:
        json.dump({str(k): v for k, v in solution.assignment.items()}, f)
    with open(os.path.join(folder, f"zone_dict_area_{level}.json"), "w") as f:
        json.dump({str(k): v for k, v in solution.area_assignment().items()}, f)
    if solution.solver_progress:
        rel = os.path.join("solver_progress", f"{level}_cp_int")
        os.makedirs(os.path.join(folder, rel), exist_ok=True)
        nodes = list(solution.problem.nodes)
        with open(os.path.join(folder, rel, "progress.jsonl"), "w") as log:
            for idx, entry in enumerate(solution.solver_progress):
                assignment = dict(zip(nodes, entry.assignment))
                zone, area = (
                    f"zone_dict_{level}_{idx:04d}.json",
                    (f"zone_dict_area_{level}_{idx:04d}.json"),
                )
                with open(os.path.join(folder, rel, zone), "w") as f:
                    json.dump({str(k): v for k, v in assignment.items()}, f)
                with open(os.path.join(folder, rel, area), "w") as f:
                    json.dump(
                        {
                            str(k): v
                            for k, v in solution._area_assignment_for(
                                assignment
                            ).items()
                        },
                        f,
                    )
                row = {
                    "solution_index": idx,
                    "objective": entry.objective,
                    "elapsed_seconds": entry.elapsed_seconds,
                    "assignment_path": zone,
                    "area_assignment_path": area,
                }
                log.write(json.dumps(row) + "\n")
        solution.metadata.update(
            {
                "solver_progress_enabled": True,
                "solver_progress_id": f"{level}_cp_int",
                "solver_progress_count": len(solution.solver_progress),
                "solver_progress_format": "jsonl",
                "solver_progress_path": os.path.join(rel, "progress.jsonl"),
                "solver_progress_dir": rel,
            }
        )
    info = {
        "level": level,
        "graph_fingerprint": graph_fingerprint(solution.problem.G),
        "status": solution.status,
        "objective": solution.objective,
        "wall_time": solution.wall_time,
        "num_zones": solution.problem.Z,
        "centroids": list(solution.problem.centroids),
        "contiguous": solution.is_contiguous(),
        "metadata": solution.metadata,
        "config": config,
    }
    save_zoning_config(config, folder)
    with open(os.path.join(folder, f"solution_{level}.json"), "w") as f:
        json.dump(info, f, indent=2)


def _legacy_run(tmp_path):
    problem = make_grid_problem(3, 3)
    problem.level = LevelSpec("Block", 0)
    first = _solution(
        problem,
        {n: int(n % 3 >= 1) for n in problem.G.nodes},
        capacity_reference={str(i): i / 3 for i in range(400)},
    )
    second = _solution(
        problem,
        {n: int(n % 3 >= 2) for n in problem.G.nodes},
        capacity_reference={str(i): i / 3 for i in range(400)},
    )
    nodes = list(problem.nodes)
    second.solver_progress = [
        SolverProgressEntry(5.0, 0.1, tuple(first.assignment[n] for n in nodes)),
        SolverProgressEntry(3.0, 0.2, tuple(second.assignment[n] for n in nodes)),
    ]
    run_dir = tmp_path / "sweep" / "run"
    names = ["stage_00_Block_0", "stage_01_Block_0"]
    records = []
    for idx, (solution, name) in enumerate(zip((first, second), names)):
        stage_dir = run_dir / "stages" / name
        _legacy_save(solution, str(stage_dir), CONFIG)
        record = stage_record(solution, idx, name, stage_dir, str(run_dir))
        # Legacy manifests kept every stage's full metadata.
        record["metadata"] = dict(solution.metadata)
        records.append(record)
    _legacy_save(second, str(run_dir), CONFIG)  # the root alias
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": CONFIG,
                "stages": records,
                "final_stage": names[1],
                "status": "FEASIBLE",
            }
        )
    )
    (run_dir / RESULT_FILENAME).write_text(
        json.dumps(
            {
                "run": {
                    "stages": [{**r, "metadata": dict(r["metadata"])} for r in records]
                }
            }
        )
    )
    return run_dir, problem, (first, second)


def _patch_config(monkeypatch):
    monkeypatch.setattr(
        "benchmark.runner.optimization_config_from_dict",
        lambda raw: type("Config", (), {"strategy": "recursive"})(),
    )


def test_legacy_run_loads_and_compacts_without_loss(tmp_path, monkeypatch):
    _patch_config(monkeypatch)
    run_dir, problem, (first, second) = _legacy_run(tmp_path)
    legacy_count = len(run_archive.legacy_files(run_dir))

    before, _, _ = load_solutions(str(run_dir), dataset=FakeDataset(problem))
    assert [s.assignment for s in before] == [first.assignment, second.assignment]

    result = compact_run(run_dir)
    assert result.status == "compacted", result.message
    # Legacy files stay until asked; the archive already wins.
    assert len(run_archive.legacy_files(run_dir)) == legacy_count
    after, _, _ = load_solutions(str(run_dir), dataset=FakeDataset(problem))
    assert [s.assignment for s in after] == [s.assignment for s in before]
    assert (
        after[0].metadata["capacity_reference"] == first.metadata["capacity_reference"]
    )
    assert after[1].metadata["solver_progress_path"] == "solver_progress.json.xz"

    archive = run_archive.read_zonings(run_dir)
    assert archive["final_stage"] == "stage_01_Block_0"
    assert archive["config"] == CONFIG
    progress = run_archive.read_solver_progress(run_dir)
    assert [r["stage"] for r in progress["stages"]] == ["stage_01_Block_0"]
    assert progress["stages"][0]["entries"][1]["node_zones"] == [
        second.assignment[n] for n in problem.nodes
    ]

    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    result_json = json.loads((run_dir / RESULT_FILENAME).read_text())
    for stage in [*manifest["stages"], *result_json["run"]["stages"]]:
        assert "capacity_reference" not in stage["metadata"]
        assert stage["metadata"]["archived_metadata_keys"] == ["capacity_reference"]
    assert manifest["stages"][1]["metadata"]["solver_progress_count"] == 2

    deleted = compact_run(run_dir, delete_legacy=True)
    # The root area zoning stays: sweeps name it as their hint_zoning.
    assert deleted.status == "already" and deleted.deleted == legacy_count - 2
    assert (run_dir / "zone_dict_area_Block_0.json").exists()
    compact_run(run_dir, delete_legacy=True, drop_root_zoning=True)
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(
        [
            MANIFEST_FILENAME,
            RESULT_FILENAME,
            run_archive.SOLVER_PROGRESS_FILENAME,
            run_archive.ZONINGS_FILENAME,
            "zoning_config.yaml",
        ]
    )
    final, _, _ = load_solutions(str(run_dir), dataset=FakeDataset(problem))
    assert [s.assignment for s in final] == [s.assignment for s in before]


def test_compaction_keeps_a_run_whose_alias_disagrees(tmp_path):
    run_dir, _, (first, _) = _legacy_run(tmp_path)
    alias = run_dir / "zone_dict_area_Block_0.json"
    alias.write_text(
        json.dumps({str(k): v for k, v in first.area_assignment().items()})
    )

    result = compact_run(run_dir, delete_legacy=True)

    assert result.status == "failed"
    assert "root alias" in result.message
    assert not (run_dir / run_archive.ZONINGS_FILENAME).exists()
    assert alias.exists()
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    assert "capacity_reference" in manifest["stages"][0]["metadata"]


def test_compact_runs_reports_every_run(tmp_path):
    _legacy_run(tmp_path)
    summary = compact_runs(str(tmp_path / "sweep"), delete_legacy=True)
    assert summary.count("compacted") == 1
    assert summary.bytes_after < summary.bytes_before
    again = compact_runs(str(tmp_path / "sweep"))
    assert again.count("already") == 1


@pytest.mark.parametrize("layout", ["archive", "legacy"])
def test_completed_run_target_reads_either_layout(tmp_path, layout):
    from assignment.generated_zones import GENERATED_ZONE_FILENAME
    from benchmark.assignment import prepare_completed_run_target

    run_dir, _, (_, second) = _legacy_run(tmp_path)
    if layout == "archive":
        assert compact_run(run_dir, delete_legacy=True).status == "compacted"

    assert prepare_completed_run_target(run_dir, "Block_0") is True
    rows = (run_dir / GENERATED_ZONE_FILENAME).read_text().splitlines()
    by_zone = {}
    for area, zone in second.area_assignment().items():
        by_zone.setdefault(zone, []).append(area)
    assert len(rows) == len(by_zone)
