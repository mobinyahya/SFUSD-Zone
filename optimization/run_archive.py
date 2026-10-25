"""Every zoning a run saves, in one compressed file.

``zonings.json.xz`` holds each stage's node and area assignment, its status,
graph fingerprint and full metadata, and the config that produced the run,
once. Area ids are listed once per unit and each stage stores only its zones,
aligned to that list. LZMA's large window then folds the metadata that stages
repeat (a pipeline's ``capacity_reference``) into about one copy, which gzip's
32 KB window cannot. Incumbent snapshots go to ``solver_progress.json.xz``
beside it, so reading a zoning never decodes them.

The archive is the only place full stage metadata is kept. The manifest and
``result.json`` carry :func:`summary_metadata`, which drops large containers,
because those two are read for every run whenever a sweep resumes or
aggregates.

Runs saved before this file existed kept one folder per stage
(``stages/<name>/zone_dict_<level>.json`` and friends, plus a root alias of the
final stage). :func:`load_run_zonings` reads either layout, and
``benchmark.compact`` rewrites the old one.
"""

from __future__ import annotations

import glob
import json
import lzma
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from optimization.levels import LevelSpec

ZONINGS_FILENAME = "zonings.json.xz"
SOLVER_PROGRESS_FILENAME = "solver_progress.json.xz"
ZONINGS_FORMAT = "sfusd-zone/zonings"
SOLVER_PROGRESS_FORMAT = "sfusd-zone/solver-progress"
ARCHIVE_VERSION = 1
# Preset 6 (the xz default) holds an 8 MB window, enough to see every stage of
# a run at once. A three-stage pipeline archive writes in ~0.2 s and reads in
# ~10 ms; higher presets only cost time.
LZMA_PRESET = 6
# Metadata values whose JSON is longer than this stay in the archive only.
SUMMARY_METADATA_LIMIT = 2048


# ---------------------------------------------------------------------- #
# writing
# ---------------------------------------------------------------------- #
def save_zonings(
    folder: str | os.PathLike,
    solutions: Sequence[Any],
    stage_names: Sequence[str],
    *,
    config: Mapping | None = None,
    final_stage: str | None = None,
    progress: bool = True,
) -> str:
    """Write ``solutions`` as one ``zonings.json.xz`` in ``folder``.

    With ``progress``, the stages' incumbent snapshots are written to
    ``solver_progress.json.xz`` first and each stage's metadata records where,
    so the archive's copy of that metadata is complete. A run that captured no
    snapshots removes a stale progress file rather than leaving one behind.
    """
    from optimization.solution import graph_fingerprint

    if len(solutions) != len(stage_names):
        raise ValueError("Every saved solution needs exactly one stage name.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if progress:
        _save_solver_progress(folder, solutions, stage_names)

    stages = []
    for index, (solution, name) in enumerate(zip(solutions, stage_names)):
        nodes = sorted(int(node) for node in solution.assignment)
        stages.append(
            {
                "name": name,
                "index": index,
                "level": solution.level.name,
                "graph_fingerprint": graph_fingerprint(solution.problem.G),
                "status": solution.status,
                "objective": solution.objective,
                "wall_time": solution.wall_time,
                "num_zones": solution.problem.Z,
                "centroids": list(solution.problem.centroids),
                "contiguous": solution.is_contiguous() if solution.feasible else None,
                "metadata": solution.metadata,
                "node_ids": nodes,
                "node_zones": [int(solution.assignment[node]) for node in nodes],
                "areas": solution.area_assignment(),
            }
        )
    path = folder / ZONINGS_FILENAME
    write_zonings(path, stages, config=config, final_stage=final_stage)
    return str(path)


def write_zonings(
    path: str | os.PathLike,
    stages: Sequence[Mapping[str, Any]],
    *,
    config: Mapping | None = None,
    final_stage: str | None = None,
) -> None:
    """Encode stage records whose ``"areas"`` is ``{area_id: zone}`` or ``None``.

    Shared by :func:`save_zonings` and the legacy-layout compaction, so both
    write the same bytes for the same zonings.
    """
    from optimization.solution import _json_ready

    names = [stage["name"] for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"Stage names must be unique, got {names}.")
    if final_stage is not None and final_stage not in names:
        raise ValueError(f"Final stage {final_stage!r} is not one of {names}.")
    area_ids, encoded = _encode_areas(stages)
    _write_xz_json(
        path,
        {
            "format": ZONINGS_FORMAT,
            "version": ARCHIVE_VERSION,
            "final_stage": final_stage,
            "config": _json_ready(config) if config is not None else None,
            "area_ids": area_ids,
            "stages": encoded,
        },
    )


def set_final_stage(folder: str | os.PathLike, final_stage: str) -> bool:
    """Mark the run's selected stage in its archive; ``False`` if it has none.

    The archive is patched rather than rewritten from the solutions, so a run
    re-evaluated on another machine keeps the fingerprints and node ids it was
    solved on.
    """
    path = Path(folder) / ZONINGS_FILENAME
    if not path.is_file():
        return False
    archive = read_zonings(path)
    if archive.get("final_stage") == final_stage:
        return True
    if final_stage not in {stage["name"] for stage in archive["stages"]}:
        raise ValueError(f"Final stage {final_stage!r} is not in {path}.")
    archive["final_stage"] = final_stage
    write_archive(path, archive)
    return True


def write_archive(path: str | os.PathLike, archive: Mapping[str, Any]) -> None:
    """Write an already-encoded archive, such as :func:`legacy_zonings` returns."""
    if archive.get("format") != ZONINGS_FORMAT:
        raise ValueError("Only a zonings archive can be written here.")
    _write_xz_json(path, archive)


def _save_solver_progress(
    folder: Path, solutions: Sequence[Any], stage_names: Sequence[str]
) -> None:
    path = folder / SOLVER_PROGRESS_FILENAME
    records = []
    for solution, name in zip(solutions, stage_names):
        metadata = solution.metadata
        if not solution.solver_progress:
            if metadata.get("solver_progress_enabled"):
                metadata["solver_progress_count"] = 0
            continue
        nodes = [int(node) for node in solution.problem.nodes]
        entries = []
        for idx, entry in enumerate(solution.solver_progress):
            if len(entry.assignment) != len(nodes):
                raise ValueError(
                    "Solver progress assignment length does not match problem nodes."
                )
            assignment = {
                node: int(zone) for node, zone in zip(nodes, entry.assignment)
            }
            row = {
                "solution_index": idx,
                "objective": entry.objective,
                "elapsed_seconds": entry.elapsed_seconds,
                "node_zones": [assignment[node] for node in nodes],
                "areas": solution._area_assignment_for(assignment),
            }
            if entry.iteration is not None:
                row["iteration"] = entry.iteration
            entries.append(row)
        progress_id = str(
            metadata.get("solver_progress_id")
            or f"{solution.level.name}_{metadata.get('solver', 'solver')}"
        )
        records.append(
            {
                "stage": name,
                "level": solution.level.name,
                "progress_id": progress_id,
                "node_ids": nodes,
                "entries": entries,
            }
        )
        metadata.update(
            {
                "solver_progress_enabled": True,
                "solver_progress_id": progress_id,
                "solver_progress_count": len(entries),
                "solver_progress_format": "json.xz",
                "solver_progress_path": SOLVER_PROGRESS_FILENAME,
            }
        )
        metadata.pop("solver_progress_dir", None)
    if records:
        write_solver_progress(path, records)
    else:
        path.unlink(missing_ok=True)


def write_solver_progress(
    path: str | os.PathLike, records: Sequence[Mapping[str, Any]]
) -> None:
    """Encode per-stage progress records whose entries carry ``"areas"``."""
    flat = [
        {**entry, "level": record["level"]}
        for record in records
        for entry in record["entries"]
    ]
    area_ids, encoded = _encode_areas(flat)
    encoded_iter = iter(encoded)
    stages = []
    for record in records:
        entries = []
        for _ in record["entries"]:
            entry = next(encoded_iter)
            entry.pop("level")
            entries.append(entry)
        stages.append({**record, "entries": entries})
    _write_xz_json(
        path,
        {
            "format": SOLVER_PROGRESS_FORMAT,
            "version": ARCHIVE_VERSION,
            "area_ids": area_ids,
            "stages": stages,
        },
    )


def _encode_areas(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    """Replace each record's ``"areas"`` dict by zones aligned to its unit's ids.

    Areas a partial assignment leaves out are ``null``; a record saved without
    an area assignment keeps ``"area_zones": null``.
    """
    universe: dict[str, set[int]] = {}
    for record in records:
        if record.get("areas") is not None:
            unit = LevelSpec.parse(record["level"]).unit
            universe.setdefault(unit, set()).update(
                int(area) for area in record["areas"]
            )
    area_ids = {unit: sorted(ids) for unit, ids in sorted(universe.items())}
    encoded = []
    for record in records:
        out = {key: value for key, value in record.items() if key != "areas"}
        areas = record.get("areas")
        if areas is None:
            out["area_zones"] = None
        else:
            areas = {int(area): int(zone) for area, zone in areas.items()}
            ids = area_ids.get(LevelSpec.parse(record["level"]).unit, [])
            out["area_zones"] = [areas.get(area) for area in ids]
        encoded.append(out)
    return area_ids, encoded


def _write_xz_json(path: str | os.PathLike, payload: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with ``payload`` as LZMA-compressed JSON.

    Non-finite floats are written as JSON's ``Infinity``/``NaN`` extensions, as
    the per-stage ``solution_<level>.json`` files were, so metadata reads back
    unchanged.
    """
    data = lzma.compress(
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        preset=LZMA_PRESET,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------- #
# reading
# ---------------------------------------------------------------------- #
def read_zonings(path: str | os.PathLike) -> dict[str, Any]:
    """Decode a ``zonings.json.xz`` (or the folder holding one)."""
    return _read_xz_json(_archive_path(path, ZONINGS_FILENAME), ZONINGS_FORMAT)


def read_solver_progress(path: str | os.PathLike) -> dict[str, Any]:
    """Decode a ``solver_progress.json.xz`` (or the folder holding one)."""
    return _read_xz_json(
        _archive_path(path, SOLVER_PROGRESS_FILENAME), SOLVER_PROGRESS_FORMAT
    )


def is_zonings_archive(path: str | os.PathLike) -> bool:
    return Path(path).name == ZONINGS_FILENAME


def get_stage(archive: Mapping[str, Any], name: str | None = None) -> dict[str, Any]:
    """The named stage, or else the run's final stage (its last if unmarked)."""
    stages = archive["stages"]
    if not stages:
        raise ValueError("The archive holds no stages.")
    name = name if name is not None else archive.get("final_stage")
    if name is None:
        return stages[-1]
    for stage in stages:
        if stage["name"] == name:
            return stage
    raise KeyError(f"No saved stage named {name!r}.")


def node_assignment(stage: Mapping[str, Any]) -> dict[int, int]:
    return {
        int(node): int(zone)
        for node, zone in zip(stage["node_ids"], stage["node_zones"])
    }


def area_assignment(
    archive: Mapping[str, Any], stage: Mapping[str, Any]
) -> dict[int, int] | None:
    """``{area_id: zone}`` for ``stage``, or ``None`` if it saved no areas."""
    zones = stage.get("area_zones")
    if zones is None:
        return None
    ids = archive["area_ids"][LevelSpec.parse(stage["level"]).unit]
    return {int(area): int(zone) for area, zone in zip(ids, zones) if zone is not None}


def load_run_zonings(
    run_dir: str | os.PathLike,
    stage_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Every zoning ``run_dir`` saved, from its archive or its legacy files.

    A legacy run's stages are found from ``stage_records`` (the manifest's,
    with ``name``, ``level`` and ``path``), else from its ``stages/`` folders,
    else from the files at its root; the result has the archive's shape.
    """
    run_dir = Path(run_dir).expanduser()
    if (run_dir / ZONINGS_FILENAME).is_file():
        return read_zonings(run_dir)
    return legacy_zonings(run_dir, stage_records)


def summary_metadata(
    metadata: Mapping[str, Any], limit: int = SUMMARY_METADATA_LIMIT
) -> dict[str, Any]:
    """``metadata`` without containers whose JSON is longer than ``limit``.

    Scalars always stay. The dropped keys are listed in
    ``archived_metadata_keys``; their values are in the run's archive.
    """
    kept: dict[str, Any] = {}
    archived = []
    for key, value in metadata.items():
        if isinstance(value, (Mapping, list, tuple, set)) and (
            len(json.dumps(value, default=str)) > limit
        ):
            archived.append(str(key))
        else:
            kept[key] = value
    if archived:
        kept["archived_metadata_keys"] = sorted(archived)
    return kept


def _archive_path(path: str | os.PathLike, filename: str) -> Path:
    path = Path(path).expanduser()
    return path / filename if path.is_dir() else path


def _read_xz_json(path: Path, expected_format: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        payload = json.loads(lzma.decompress(f.read()))
    if payload.get("format") != expected_format:
        raise ValueError(f"{path} is not a {expected_format} file.")
    if payload.get("version") != ARCHIVE_VERSION:
        raise ValueError(
            f"{path} is version {payload.get('version')}; this code reads "
            f"version {ARCHIVE_VERSION}."
        )
    return payload


# ---------------------------------------------------------------------- #
# legacy layout
# ---------------------------------------------------------------------- #
def legacy_zonings(
    run_dir: str | os.PathLike,
    stage_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read the one-folder-per-stage layout into the archive's shape.

    Missing ``solution_<level>.json`` fields are ``None``, as the old reader
    left them for the caller to fill from the manifest. A stage without its
    node assignment gets no node ids and no fingerprint, so a graph reader
    rebuilds its nodes from areas, or refuses if it has none either; a stage
    with no file at all raises ``FileNotFoundError``.
    """
    run_dir = Path(run_dir).expanduser()
    if stage_records is None:
        stage_records = _discover_legacy_stages(run_dir)
    stages = []
    config = None
    for position, record in enumerate(stage_records):
        level = record["level"]
        stage_dir = run_dir / record.get("path", ".")
        info = _read_json_if_present(stage_dir / f"solution_{level}.json")
        area_path = stage_dir / f"zone_dict_area_{level}.json"
        node_path = stage_dir / f"zone_dict_{level}.json"
        if node_path.is_file() or not (info or area_path.is_file()):
            assignment = {int(k): int(v) for k, v in _read_json(node_path).items()}
            fingerprint = info.get("graph_fingerprint")
        else:
            assignment, fingerprint = {}, None
        if config is None and info.get("config") is not None:
            config = info["config"]
        nodes = sorted(assignment)
        stages.append(
            {
                "name": record.get("name", level),
                "index": record.get("index", position),
                "level": level,
                "graph_fingerprint": fingerprint,
                "status": info.get("status"),
                "objective": info.get("objective"),
                "wall_time": info.get("wall_time"),
                "num_zones": info.get("num_zones"),
                "centroids": info.get("centroids"),
                "contiguous": info.get("contiguous"),
                "metadata": info.get("metadata"),
                "node_ids": nodes,
                "node_zones": [assignment[node] for node in nodes],
                "areas": _read_json(area_path) if area_path.is_file() else None,
            }
        )
    area_ids, encoded = _encode_areas(stages)
    return {
        "format": ZONINGS_FORMAT,
        "version": ARCHIVE_VERSION,
        "final_stage": None,
        "config": config,
        "area_ids": area_ids,
        "stages": encoded,
    }


def legacy_files(run_dir: str | os.PathLike) -> list[Path]:
    """Every file the legacy layout wrote under ``run_dir``.

    Stage folders themselves stay: assignment jobs write their targets there.
    """
    run_dir = Path(run_dir)
    patterns = [
        "zone_dict_*.json",
        "solution_*.json",
        "zoning_config.yaml",
    ]
    files = [p for pattern in patterns for p in run_dir.glob(pattern)]
    for stage_dir in run_dir.glob("stages/*"):
        files.extend(p for pattern in patterns for p in stage_dir.glob(pattern))
        files.extend(stage_dir.glob("solver_progress/*/*"))
    files.extend(run_dir.glob("solver_progress/*/*"))
    return sorted(p for p in files if p.is_file())


def legacy_progress_records(
    run_dir: str | os.PathLike, archive: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Read the legacy ``solver_progress/<id>/`` snapshots of ``archive``'s stages.

    Each stage's ``solver_progress_dir`` is relative to the folder its files
    were saved in: its stage folder, or the run root for the final alias.
    """
    run_dir = Path(run_dir)
    records = []
    for stage in archive["stages"]:
        metadata = stage.get("metadata") or {}
        rel = metadata.get("solver_progress_dir")
        if not rel:
            continue
        candidates = [run_dir / "stages" / stage["name"] / rel, run_dir / rel]
        progress_dir = next((p for p in candidates if p.is_dir()), None)
        if progress_dir is None:
            continue
        log = progress_dir / "progress.jsonl"
        entries = []
        node_ids: list[int] | None = None
        with open(log, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                with open(progress_dir / row["assignment_path"], encoding="utf-8") as g:
                    assignment = {int(k): int(v) for k, v in json.load(g).items()}
                area_path = progress_dir / row["area_assignment_path"]
                areas = (
                    {int(k): int(v) for k, v in _read_json(area_path).items()}
                    if area_path.is_file()
                    else None
                )
                if node_ids is None:
                    node_ids = list(assignment)
                entry = {
                    "solution_index": row["solution_index"],
                    "objective": row["objective"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "node_zones": [assignment[node] for node in node_ids],
                    "areas": areas,
                }
                if "iteration" in row:
                    entry["iteration"] = row["iteration"]
                entries.append(entry)
        records.append(
            {
                "stage": stage["name"],
                "level": stage["level"],
                "progress_id": str(metadata.get("solver_progress_id") or rel),
                "node_ids": node_ids or [],
                "entries": entries,
            }
        )
    return records


def _discover_legacy_stages(run_dir: Path) -> list[dict[str, Any]]:
    stage_dirs = sorted(p for p in run_dir.glob("stages/*") if p.is_dir())
    folders = [p for p in stage_dirs if glob.glob(str(p / "solution_*.json"))] or [
        run_dir
    ]
    records = []
    for folder in folders:
        for solution_path in sorted(folder.glob("solution_*.json")):
            level = solution_path.stem.removeprefix("solution_")
            records.append(
                {
                    "name": folder.name if folder != run_dir else level,
                    "level": level,
                    "path": os.path.relpath(folder, run_dir),
                }
            )
    return records


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_json_if_present(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}
