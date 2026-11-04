"""Rewrite legacy benchmark runs into the one-archive layout.

    uv run python -m benchmark.compact <benchmark_root> [--delete-legacy [--drop-root-zoning]]
                                                     [--workers N]

A legacy run saved one folder per stage (``zone_dict_<level>.json``,
``zone_dict_area_<level>.json``, ``solution_<level>.json``,
``zoning_config.yaml``, and ``solver_progress/<id>/`` snapshots), a root alias
of the final stage, and every stage's full metadata again in its manifest and
``result.json``. Compaction writes the run's ``zonings.json.xz`` (and
``solver_progress.json.xz``), reads them back and checks every assignment,
area assignment, progress snapshot and metadata value against the legacy
files, and only then slims the manifest and ``result.json`` to
:func:`~optimization.run_archive.summary_metadata`. Legacy files are deleted
only with ``--delete-legacy``; until then readers prefer the archive and the
old files are an unused copy.

Even then the root ``zone_dict_area_<level>.json`` stays unless
``--drop-root-zoning`` is also given: sweep configs name those files as their
``hint_zoning``, and a task's identity hashes that file's bytes, so moving a
config to the archive would also rerun its sweep.

Do not compact a sweep that is still writing: a run rewritten mid-evaluation
can lose its archive's final-stage mark.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.results import discover_run_dirs
from benchmark.runner import (
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    load_manifest,
    write_json,
)
from optimization import run_archive
from optimization.levels import LevelSpec
from optimization.solution import ZONING_CONFIG_FILENAME, save_zoning_config


@dataclass
class CompactionResult:
    run_dir: str
    status: str  # compacted, already, skipped, failed
    bytes_before: int = 0
    bytes_after: int = 0
    deleted: int = 0
    message: str = ""


@dataclass
class CompactionSummary:
    results: list[CompactionResult] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(result.status == status for result in self.results)

    @property
    def bytes_before(self) -> int:
        return sum(r.bytes_before for r in self.results if r.status == "compacted")

    @property
    def bytes_after(self) -> int:
        return sum(r.bytes_after for r in self.results if r.status == "compacted")


def compact_run(
    run_dir: str | os.PathLike,
    *,
    delete_legacy: bool = False,
    drop_root_zoning: bool = False,
) -> CompactionResult:
    """Compact one run; never raises, and leaves a failed run as it was."""
    run_dir = Path(run_dir)
    try:
        return _compact_run(run_dir, delete_legacy, drop_root_zoning)
    except Exception as exc:
        return CompactionResult(str(run_dir), "failed", message=str(exc) or repr(exc))


def _compact_run(
    run_dir: Path, delete_legacy: bool, drop_root_zoning: bool
) -> CompactionResult:
    archive_path = run_dir / run_archive.ZONINGS_FILENAME
    legacy = run_archive.legacy_files(run_dir)
    if archive_path.is_file():
        if delete_legacy and legacy:
            # A previous pass compacted and verified this run but kept its files.
            deleted = _delete(run_dir, legacy, drop_root_zoning)
            return CompactionResult(str(run_dir), "already", deleted=deleted)
        return CompactionResult(str(run_dir), "already")
    manifest = load_manifest(run_dir)
    records = list(manifest.get("stages") or [])
    if not records:
        return CompactionResult(str(run_dir), "skipped", message="no saved stages")

    index_files = [run_dir / MANIFEST_FILENAME, run_dir / RESULT_FILENAME]
    before = _size([*legacy, *index_files])

    archive = run_archive.legacy_zonings(run_dir, records)
    for saved, record in zip(archive["stages"], records):
        # The old reader fell back to the manifest when a stage's info file was
        # missing; the archive takes those values so it stands on its own.
        if saved["metadata"] is None:
            saved["metadata"] = dict(record.get("metadata") or {})
        if saved["status"] is None:
            for key in ("status", "objective", "wall_time", "num_zones"):
                saved[key] = record.get(key)
    if archive["config"] is None:
        archive["config"] = manifest.get("config")
    archive["final_stage"] = manifest.get("final_stage")

    progress = run_archive.legacy_progress_records(run_dir, archive)
    progress_stages = {record["stage"] for record in progress}
    for saved in archive["stages"]:
        if saved["name"] in progress_stages:
            _point_at_progress_archive(saved["metadata"])

    progress_path = run_dir / run_archive.SOLVER_PROGRESS_FILENAME
    try:
        run_archive.write_archive(archive_path, archive)
        if progress:
            run_archive.write_solver_progress(progress_path, progress)
        _verify(run_dir, archive_path, progress_path, progress, records)
    except Exception:
        archive_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)
        raise

    _slim_index_files(run_dir, manifest, progress_stages)
    if not (run_dir / ZONING_CONFIG_FILENAME).is_file() and archive["config"]:
        save_zoning_config(archive["config"], str(run_dir))
    deleted = 0
    if delete_legacy:
        deleted = _delete(run_dir, run_archive.legacy_files(run_dir), drop_root_zoning)
    after_files = [archive_path, *index_files, *(p for p in legacy if p.exists())]
    if progress:
        after_files.append(progress_path)
    return CompactionResult(
        str(run_dir),
        "compacted",
        bytes_before=before,
        bytes_after=_size(after_files),
        deleted=deleted,
    )


def _point_at_progress_archive(metadata: dict[str, Any]) -> None:
    metadata.update(
        {
            "solver_progress_format": "json.xz",
            "solver_progress_path": run_archive.SOLVER_PROGRESS_FILENAME,
        }
    )
    metadata.pop("solver_progress_dir", None)


def _verify(
    run_dir: Path,
    archive_path: Path,
    progress_path: Path,
    progress: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Check the written archive against the legacy files it replaces."""
    archive = run_archive.read_zonings(archive_path)
    legacy = run_archive.legacy_zonings(run_dir, records)
    if len(archive["stages"]) != len(legacy["stages"]):
        raise ValueError("archive stage count differs from the legacy files")
    for saved, old in zip(archive["stages"], legacy["stages"]):
        name = saved["name"]
        if run_archive.node_assignment(saved) != run_archive.node_assignment(old):
            raise ValueError(f"stage {name}: node assignment differs")
        if run_archive.area_assignment(archive, saved) != run_archive.area_assignment(
            legacy, old
        ):
            raise ValueError(f"stage {name}: area assignment differs")
        if old["metadata"] is not None:
            expected = dict(old["metadata"])
            if saved["metadata"].get("solver_progress_path") == (
                run_archive.SOLVER_PROGRESS_FILENAME
            ):
                _point_at_progress_archive(expected)
            if _canonical(saved["metadata"]) != _canonical(expected):
                raise ValueError(f"stage {name}: metadata differs")
        for key in ("graph_fingerprint", "level", "centroids", "contiguous"):
            if old[key] is not None and _canonical(saved[key]) != _canonical(old[key]):
                raise ValueError(f"stage {name}: {key} differs")
    _verify_root_alias(run_dir, archive)
    if progress:
        written = run_archive.read_solver_progress(progress_path)
        for record, stage in zip(progress, written["stages"]):
            if len(record["entries"]) != len(stage["entries"]):
                raise ValueError(f"progress {record['stage']}: entry count differs")
            ids = written["area_ids"].get(LevelSpec.parse(stage["level"]).unit, [])
            for old, new in zip(record["entries"], stage["entries"]):
                if old["node_zones"] != new["node_zones"]:
                    raise ValueError(f"progress {record['stage']}: zones differ")
                if old["areas"] is not None:
                    areas = {
                        a: z for a, z in zip(ids, new["area_zones"]) if z is not None
                    }
                    if areas != old["areas"]:
                        raise ValueError(f"progress {record['stage']}: areas differ")


def _verify_root_alias(run_dir: Path, archive: dict[str, Any]) -> None:
    """The root alias must be the final stage's zoning, or it is not redundant."""
    final = run_archive.get_stage(archive)
    alias = run_dir / f"zone_dict_area_{final['level']}.json"
    if not alias.is_file():
        return
    with open(alias, encoding="utf-8") as f:
        areas = {int(k): int(v) for k, v in json.load(f).items()}
    if areas != run_archive.area_assignment(archive, final):
        raise ValueError(
            f"root alias {alias.name} is not stage {final['name']}'s zoning"
        )


def _slim_index_files(
    run_dir: Path, manifest: dict[str, Any], progress_stages: set[str]
) -> None:
    for stage in manifest.get("stages", []):
        metadata = dict(stage.get("metadata") or {})
        if stage.get("name") in progress_stages:
            _point_at_progress_archive(metadata)
        stage["metadata"] = run_archive.summary_metadata(metadata)
    write_json(str(run_dir / MANIFEST_FILENAME), manifest)
    result_path = run_dir / RESULT_FILENAME
    if not result_path.is_file():
        return
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    for stage in (result.get("run") or {}).get("stages", []):
        if "metadata" in stage:
            metadata = dict(stage["metadata"] or {})
            if stage.get("name") in progress_stages:
                _point_at_progress_archive(metadata)
            stage["metadata"] = run_archive.summary_metadata(metadata)
    write_json(str(result_path), result)


def _delete(run_dir: Path, files: list[Path], drop_root_zoning: bool) -> int:
    keep = {run_dir / ZONING_CONFIG_FILENAME}
    if not drop_root_zoning:
        keep.update(run_dir.glob("zone_dict_area_*.json"))
    deleted = 0
    for path in files:
        if path in keep:
            continue
        path.unlink()
        deleted += 1
    # Empty folders the old layout left; stage folders holding assignment
    # targets are not empty and stay.
    for folder in sorted(
        {p.parent for p in files} | {p.parent.parent for p in files},
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        if folder != run_dir and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    stages = run_dir / "stages"
    if stages.is_dir() and not any(stages.iterdir()):
        stages.rmdir()
    return deleted


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _size(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


def compact_runs(
    root_folder: str,
    *,
    delete_legacy: bool = False,
    drop_root_zoning: bool = False,
    workers: int = 1,
) -> CompactionSummary:
    jobs = [
        (run_dir, delete_legacy, drop_root_zoning)
        for run_dir in discover_run_dirs(root_folder)
    ]
    summary = CompactionSummary()
    if workers <= 1:
        summary.results.extend(map(_compact_worker, jobs))
        return summary
    with ProcessPoolExecutor(max_workers=workers) as pool:
        summary.results.extend(pool.map(_compact_worker, jobs, chunksize=8))
    return summary


def _compact_worker(job: tuple[str, bool, bool]) -> CompactionResult:
    run_dir, delete_legacy, drop_root_zoning = job
    return compact_run(
        run_dir, delete_legacy=delete_legacy, drop_root_zoning=drop_root_zoning
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite legacy benchmark runs into one zonings archive each."
    )
    parser.add_argument("root", help="Benchmark output root (or one run folder).")
    parser.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Delete the per-stage files once the archive is verified.",
    )
    parser.add_argument(
        "--drop-root-zoning",
        action="store_true",
        help="With --delete-legacy, also delete the root zone_dict_area_<level>.json "
        "(check no sweep names it as hint_zoning first).",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    summary = compact_runs(
        args.root,
        delete_legacy=args.delete_legacy,
        drop_root_zoning=args.drop_root_zoning,
        workers=args.workers,
    )
    for result in summary.results:
        if result.status == "failed":
            print(f"FAILED {result.run_dir}: {result.message}")
    mb = 1024 * 1024
    print(
        f"compacted={summary.count('compacted')} already={summary.count('already')} "
        f"skipped={summary.count('skipped')} failed={summary.count('failed')}; "
        f"{summary.bytes_before / mb:.1f} MB -> {summary.bytes_after / mb:.1f} MB"
        + ("" if args.delete_legacy else " (legacy files kept)")
    )


if __name__ == "__main__":
    main()
