"""Strict, CWD-independent data scenario configuration."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import yaml

PACKAGE_CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG_PATH = PACKAGE_CONFIG_ROOT / "base.yaml"

_RUN_KEYS = {"scenario", "overrides"}
_OVERRIDE_KEYS = {"roots", "sources", "filters"}
_BASE_KEYS = {
    "schema_version",
    "roots",
    "files",
    "geographies",
    "student_frl_estimates",
    "school_years",
}
_SCENARIO_KEYS = {"id", "sources", "filters"}
_SOURCE_KEYS = {
    "path",
    "root",
    "companions",
    "classification",
    "geography_vintage",
}
_SPECIAL_ROOTS = {"package", "repository"}
_FILTER_KEYS = {
    "optimization": {
        "years",
        "grades",
        "student_population",
        "rounds",
        "special_programs",
        "program_population",
        "capacity_scenario",
        "include_k8",
        "include_citywide",
        "include_mission_bay",
        "geography_vintage",
        "frl_estimate",
        "outside_district_students",
    },
    "assignment": {
        "year",
        "grades",
        "student_population",
        "rounds",
        "special_programs",
        "capacity_profile",
        "capacity_scenario",
        "include_mission_bay",
        "geography_vintage",
        "frl_estimate",
        "outside_district_students",
    },
}
_CANONICAL_YEAR = re.compile(r"^\d{4}$")
_CANONICAL_GRADES = {"PK", "TK", "KG", *(f"{grade:02d}" for grade in range(1, 13))}
_STUDENT_POPULATIONS = {"applicant", "enrolled"}
_SPECIAL_PROGRAM_MODES = {"include", "exclude_only_special", "exclude_any_special"}
_FILTER_DEFAULTS = {
    "capacity_scenario": "programs",
    "geography_vintage": "2010",
    "frl_estimate": None,
    "outside_district_students": "ignore",
}
_ANNUAL_ASSIGNMENT_ROLES = {
    "assignment.students",
    "assignment.programs",
    "assignment.programs.catalog",
    "assignment.schools",
    "assignment.school_coordinates",
}
_YEAR_KEYS = {"optimization", "assignment"}
_OPTIMIZATION_YEAR_KEYS = {"students"}
_ASSIGNMENT_YEAR_KEYS = {"students", "grades"}
_ASSIGNMENT_GRADE_KEYS = {"profiles"}
_ASSIGNMENT_VARIANT_KEYS = {"programs", "programs_catalog", "schools"}
_GEOGRAPHY_KEYS = {
    "blocks",
    "blockgroups",
    "tracts",
    "crosswalk",
    "adjacency",
    "manual_edges",
}
_MISSING = object()
_CHECKSUM_CACHE: dict[tuple[str, int, int, int, int, int], str] = {}
_CHECKSUM_LOCK = threading.Lock()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _absolute_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _unknown_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed, key=str)
    if unknown:
        raise ValueError(f"Unknown {label} keys: {unknown}.")


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Could not read {label} YAML {path}: {exc}.") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid {label} YAML {path}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label.capitalize()} YAML {path} must contain a map.")
    return payload


def _require_map(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a map.")
    return value


def _deep_merge(base: Any, override: Any) -> Any:
    """Deep-merge maps while replacing lists and scalar values."""
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        merged = copy.deepcopy(dict(base))
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def _anchor_direct_paths(value: Any, base: Path, inherited: Any = None) -> Any:
    """Make rootless direct source paths relative to their declaring YAML."""
    if isinstance(value, list):
        return [_anchor_direct_paths(item, base) for item in value]
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    if set(value) & _SOURCE_KEYS:
        anchored = copy.deepcopy(dict(value))
        inherits_root = (
            isinstance(inherited, Mapping)
            and bool(set(inherited) & _SOURCE_KEYS)
            and "root" in inherited
        )
        if "root" not in anchored and not inherits_root:
            if "path" in anchored:
                anchored["path"] = str(_absolute_path(anchored["path"], base))
            if "companions" in anchored:
                anchored["companions"] = [
                    str(_absolute_path(path, base)) for path in anchored["companions"]
                ]
        return anchored
    inherited_map = inherited if isinstance(inherited, Mapping) else {}
    return {
        key: _anchor_direct_paths(item, base, inherited_map.get(key))
        for key, item in value.items()
    }


def _serializable_copy(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str | int | float | bool):
                raise TypeError(f"{label} key {key!r} is not serializable.")
            result[key] = _serializable_copy(item, label)
        return result
    if isinstance(value, list | tuple):
        return [_serializable_copy(item, label) for item in value]
    if isinstance(value, Path | os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, str | int | float | bool):
        return copy.deepcopy(value)
    raise TypeError(f"{label} value {value!r} is not serializable.")


def anchor_data_config(
    data_config: Mapping[str, Any], base_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Return a strict serializable data config anchored to its declaring file."""
    if not isinstance(data_config, Mapping):
        raise ValueError("Data configuration must be a {scenario, overrides} map.")
    run = _serializable_copy(data_config, "Data configuration")
    _unknown_keys(run, _RUN_KEYS, "run configuration")
    missing = sorted(_RUN_KEYS - set(run))
    if missing:
        raise ValueError(f"Run configuration is missing keys: {missing}.")

    declaring_dir = _absolute_path(base_dir, Path.cwd())
    scenario = run["scenario"]
    if not isinstance(scenario, str | os.PathLike):
        raise ValueError("Run configuration scenario must be a name or YAML path.")
    scenario_text = os.fspath(scenario)
    bundled = PACKAGE_CONFIG_ROOT / "scenarios" / f"{scenario_text}.yaml"
    if Path(scenario_text).is_absolute():
        run["scenario"] = str(_absolute_path(scenario_text, declaring_dir))
    elif not bundled.is_file():
        run["scenario"] = str(_absolute_path(scenario_text, declaring_dir))

    overrides = _require_map(run["overrides"], "Run overrides")
    _unknown_keys(overrides, _OVERRIDE_KEYS, "override")
    roots = _require_map(overrides.get("roots", {}), "Overrides roots")
    for name, value in roots.items():
        if not isinstance(name, str) or not isinstance(value, str | os.PathLike):
            raise ValueError("Root overrides must map names to path strings.")
        roots[name] = str(_absolute_path(value, declaring_dir))

    sources = _require_map(overrides.get("sources", {}), "Overrides sources")
    for role, source in sources.items():
        if not isinstance(role, str) or "." not in role:
            raise ValueError(
                f"Override source roles must be flat dotted names, got {role!r}."
            )
        _validate_source_ref(source, f"overrides.sources.{role}", partial=True)
    if sources:
        overrides["sources"] = _anchor_direct_paths(sources, declaring_dir)
    _validate_filters(overrides.get("filters", {}), "Override filters", partial=True)
    return run


def _checksum(path: Path) -> str | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None

    key = (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    with _CHECKSUM_LOCK:
        cached = _CHECKSUM_CACHE.get(key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    with _CHECKSUM_LOCK:
        _CHECKSUM_CACHE[key] = checksum
    return checksum


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    """One fully resolved source file and its required companion files."""

    path: Path
    companions: tuple[Path, ...] = ()
    classification: str = "unspecified"
    catalog_id: str | None = None
    geography_vintage: str | None = None

    def manifest(self) -> dict[str, Any]:
        """Return paths, presence state, and current content checksums."""

        def file_entry(path: Path) -> dict[str, Any]:
            checksum = _checksum(path)
            return {
                "path": str(path),
                "status": "present" if checksum is not None else "missing",
                "sha256": checksum,
            }

        entry = file_entry(self.path)
        entry.update(
            {
                "catalog_id": self.catalog_id,
                "classification": self.classification,
                "geography_vintage": self.geography_vintage,
                "companions": [file_entry(path) for path in self.companions],
            }
        )
        return entry


@dataclass(frozen=True, slots=True)
class DataScenario:
    """Resolved, immutable view of one data scenario."""

    id: str
    schema_version: int
    roots: Mapping[str, Path]
    filters: Mapping[str, Mapping[str, Any]]
    _source_values: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        run_config: str | os.PathLike[str] | Mapping[str, Any],
        *,
        base_path: str | os.PathLike[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> Self:
        return load_scenario(run_config, base_path=base_path, environ=environ)

    @property
    def cache_root(self) -> Path:
        try:
            return self.roots["cache"]
        except KeyError as exc:
            raise ValueError(
                "Base configuration must define the 'cache' root."
            ) from exc

    def resolved(self, role: str) -> Any:
        """Return a role's resolved scalar, tuple, or named map."""
        try:
            return self._source_values[role]
        except KeyError as exc:
            raise KeyError(f"Unknown source role {role!r}.") from exc

    def sources(self, role: str) -> tuple[ResolvedSource, ...]:
        """Flatten all resolved files for a role in configured order."""
        flattened: list[ResolvedSource] = []

        def visit(value: Any) -> None:
            if isinstance(value, ResolvedSource):
                flattened.append(value)
            elif isinstance(value, Mapping):
                for child in value.values():
                    visit(child)
            elif isinstance(value, tuple):
                for child in value:
                    visit(child)
            else:  # pragma: no cover - construction guarantees this invariant
                raise TypeError(f"Unsupported resolved source value {value!r}.")

        visit(self.resolved(role))
        return tuple(flattened)

    def source(self, role: str) -> ResolvedSource:
        """Return a role that resolves to exactly one source file."""
        sources = self.sources(role)
        if len(sources) != 1:
            raise ValueError(
                f"Source role {role!r} resolves to {len(sources)} files, not one."
            )
        return sources[0]

    def source_map(self, role: str) -> Mapping[str, Any]:
        """Return a role whose top-level value is a named source map."""
        value = self.resolved(role)
        if not isinstance(value, Mapping):
            raise ValueError(f"Source role {role!r} is not a named map.")
        return value

    def filter(self, group: str, key: str, default: Any = _MISSING) -> Any:
        """Read one validated filter, optionally returning a default."""
        if group not in _FILTER_KEYS:
            raise ValueError(f"Unknown filter group {group!r}.")
        if key not in _FILTER_KEYS[group]:
            raise ValueError(f"Unknown {group} filter key {key!r}.")
        try:
            return self.filters[group][key]
        except KeyError:
            if default is not _MISSING:
                return default
            raise KeyError(f"Filter {group}.{key} is not configured.") from None

    def source_manifest(
        self, roles: str | Iterable[str] | None = None
    ) -> dict[str, Any]:
        """Build a deterministic content and selector manifest for source roles."""
        selected = _selected_roles(self._source_values, roles)

        def build(value: Any) -> Any:
            if isinstance(value, ResolvedSource):
                return value.manifest()
            if isinstance(value, Mapping):
                return {key: build(child) for key, child in value.items()}
            if isinstance(value, tuple):
                return [build(child) for child in value]
            raise TypeError(f"Unsupported resolved source value {value!r}.")

        return {
            "schema_version": self.schema_version,
            "scenario": self.id,
            "filters": _plain(self.filters),
            "sources": {role: build(self._source_values[role]) for role in selected},
        }

    @property
    def semantic_fingerprint(self) -> str:
        """Fingerprint scenario semantics, excluding source contents/cache root."""

        def describe(value: Any) -> Any:
            if isinstance(value, ResolvedSource):
                return {
                    "path": str(value.path),
                    "companions": [str(path) for path in value.companions],
                    "classification": value.classification,
                    "catalog_id": value.catalog_id,
                    "geography_vintage": value.geography_vintage,
                }
            if isinstance(value, Mapping):
                return {key: describe(child) for key, child in value.items()}
            if isinstance(value, tuple):
                return [describe(child) for child in value]
            raise TypeError(f"Unsupported resolved source value {value!r}.")

        return _fingerprint(
            {
                "id": self.id,
                "schema_version": self.schema_version,
                "sources": {
                    role: describe(value) for role, value in self._source_values.items()
                },
                "filters": _plain(self.filters),
            }
        )

    @property
    def source_fingerprint(self) -> str:
        """Fingerprint all resolved source paths and current file contents."""
        return _fingerprint(self.source_manifest())


def _selected_roles(
    sources: Mapping[str, Any], roles: str | Iterable[str] | None
) -> list[str]:
    if roles is None:
        return sorted(sources)
    requested = [roles] if isinstance(roles, str) else list(roles)
    unknown = sorted(set(requested) - set(sources))
    if unknown:
        raise KeyError(f"Unknown source roles: {unknown}.")
    return sorted(dict.fromkeys(requested))


def _validate_source_ref(value: Any, label: str, *, partial: bool = False) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{label} cannot be an empty source reference.")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_source_ref(child, f"{label}[{index}]", partial=partial)
        return
    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must be a catalog ID, direct source object, list, or named map."
        )

    source_object = bool(set(value) & _SOURCE_KEYS)
    if source_object:
        _unknown_keys(value, _SOURCE_KEYS, f"source-ref {label}")
        if "path" not in value and not partial:
            raise ValueError(f"Direct source {label} must define 'path'.")
        if "path" in value and not isinstance(value["path"], str | os.PathLike):
            raise ValueError(f"Direct source {label}.path must be a path string.")
        if "root" in value and not isinstance(value["root"], str):
            raise ValueError(f"Direct source {label}.root must be a root name.")
        if "classification" in value and not isinstance(value["classification"], str):
            raise ValueError(f"Direct source {label}.classification must be a string.")
        if "geography_vintage" in value:
            _geography_vintage(
                value["geography_vintage"], f"Direct source {label}.geography_vintage"
            )
        companions = value.get("companions", [])
        if not isinstance(companions, list) or not all(
            isinstance(path, str | os.PathLike) for path in companions
        ):
            raise ValueError(f"Direct source {label}.companions must be a path list.")
        return

    for name, child in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Named source keys in {label} must be non-empty strings.")
        _validate_source_ref(child, f"{label}.{name}", partial=partial)


def _canonical_grade(value: Any, label: str) -> str:
    if not isinstance(value, str) or value not in _CANONICAL_GRADES:
        expected = "PK, TK, KG, or a zero-padded grade 01 through 12"
        raise ValueError(f"{label} must be a canonical grade ({expected}).")
    return value


def _canonical_year(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _CANONICAL_YEAR.fullmatch(value):
        raise ValueError(
            f"{label} must be a four-character school-year string such as '2324'."
        )
    return value


def _geography_vintage(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}", value):
        raise ValueError(f"{label} must be a four-digit Census vintage such as '2020'.")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _validate_filters(
    filters: Any, label: str, *, partial: bool = False
) -> dict[str, Any]:
    filters = _require_map(filters, label)
    _unknown_keys(filters, set(_FILTER_KEYS), "filter group")
    for group, values in filters.items():
        values = _require_map(values, f"{label}.{group}")
        _unknown_keys(values, _FILTER_KEYS[group], f"{group} filter")
        if not partial:
            for key, value in _FILTER_DEFAULTS.items():
                values.setdefault(key, value)
            missing = sorted(_FILTER_KEYS[group] - set(values))
            if missing:
                raise ValueError(f"{label}.{group} is missing filters: {missing}.")

        if "years" in values:
            years = values["years"]
            if not isinstance(years, list) or not years:
                raise ValueError(f"{label}.{group}.years must be a non-empty list.")
            values["years"] = [
                _canonical_year(year, f"{label}.{group}.years[{index}]")
                for index, year in enumerate(years)
            ]
            if len(set(values["years"])) != len(values["years"]):
                raise ValueError(f"{label}.{group}.years cannot contain duplicates.")
        if "year" in values:
            values["year"] = _canonical_year(values["year"], f"{label}.{group}.year")
        if "grades" in values:
            grades = values["grades"]
            if not isinstance(grades, list) or not grades:
                raise ValueError(f"{label}.{group}.grades must be a non-empty list.")
            values["grades"] = [
                _canonical_grade(grade, f"{label}.{group}.grades[{index}]")
                for index, grade in enumerate(grades)
            ]
            if len(set(values["grades"])) != len(values["grades"]):
                raise ValueError(f"{label}.{group}.grades cannot contain duplicates.")
        if "student_population" in values and (
            values["student_population"] not in _STUDENT_POPULATIONS
        ):
            raise ValueError(
                f"{label}.{group}.student_population must be applicant or enrolled."
            )
        if "special_programs" in values and (
            values["special_programs"] not in _SPECIAL_PROGRAM_MODES
        ):
            raise ValueError(
                f"{label}.{group}.special_programs must be include, "
                "exclude_only_special, or exclude_any_special."
            )
        if "rounds" in values:
            rounds = values["rounds"]
            if rounds != "all" and (
                not isinstance(rounds, list)
                or not rounds
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in rounds
                )
            ):
                raise ValueError(
                    f"{label}.{group}.rounds must be all or a non-empty list of "
                    "positive integers."
                )
            if rounds != "all":
                if len(set(rounds)) != len(rounds):
                    raise ValueError(
                        f"{label}.{group}.rounds cannot contain duplicates."
                    )
                values["rounds"] = sorted(rounds)
        for key in ("program_population", "capacity_scenario", "capacity_profile"):
            if key in values:
                values[key] = _nonempty_string(values[key], f"{label}.{group}.{key}")
        if "geography_vintage" in values:
            values["geography_vintage"] = _geography_vintage(
                values["geography_vintage"], f"{label}.{group}.geography_vintage"
            )
        if "frl_estimate" in values and values["frl_estimate"] is not None:
            values["frl_estimate"] = _nonempty_string(
                values["frl_estimate"], f"{label}.{group}.frl_estimate"
            )
        if "outside_district_students" in values and values[
            "outside_district_students"
        ] not in {"ignore", "include"}:
            raise ValueError(
                f"{label}.{group}.outside_district_students must be ignore or include."
            )
        for key in ("include_k8", "include_citywide", "include_mission_bay"):
            if key in values and not isinstance(values[key], bool):
                raise ValueError(f"{label}.{group}.{key} must be a boolean.")
    return filters


def _validate_catalog_reference(
    value: Any, label: str, files: Mapping[str, Any]
) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must reference one catalog file ID.")
    if value not in files:
        raise ValueError(f"{label} references unknown catalog file ID {value!r}.")


def _validate_population_sources(
    value: Any, label: str, files: Mapping[str, Any]
) -> None:
    sources = _require_map(value, label)
    _unknown_keys(sources, _STUDENT_POPULATIONS, label)
    missing = sorted(_STUDENT_POPULATIONS - set(sources))
    if missing:
        raise ValueError(f"{label} is missing populations: {missing}.")
    for population, source in sources.items():
        _validate_catalog_reference(source, f"{label}.{population}", files)


def _validate_school_years(value: Any, files: Mapping[str, Any]) -> None:
    school_years = _require_map(value, "Base school_years")
    if not school_years:
        raise ValueError("Base school_years must not be empty.")
    for year, entry_value in school_years.items():
        _canonical_year(year, "Base school_years key")
        entry = _require_map(entry_value, f"Base school_years.{year}")
        _unknown_keys(entry, _YEAR_KEYS, f"school_years.{year}")
        if "optimization" not in entry:
            raise ValueError(
                f"Base school_years.{year} must define optimization students."
            )
        optimization = _require_map(
            entry["optimization"], f"Base school_years.{year}.optimization"
        )
        _unknown_keys(
            optimization,
            _OPTIMIZATION_YEAR_KEYS,
            f"school_years.{year}.optimization",
        )
        if set(optimization) != _OPTIMIZATION_YEAR_KEYS:
            raise ValueError(
                f"Base school_years.{year}.optimization must define students."
            )
        _validate_population_sources(
            optimization["students"],
            f"Base school_years.{year}.optimization.students",
            files,
        )

        if "assignment" not in entry:
            continue
        assignment = _require_map(
            entry["assignment"], f"Base school_years.{year}.assignment"
        )
        _unknown_keys(
            assignment,
            _ASSIGNMENT_YEAR_KEYS,
            f"school_years.{year}.assignment",
        )
        missing = sorted(_ASSIGNMENT_YEAR_KEYS - set(assignment))
        if missing:
            raise ValueError(
                f"Base school_years.{year}.assignment is missing keys: {missing}."
            )
        _validate_population_sources(
            assignment["students"],
            f"Base school_years.{year}.assignment.students",
            files,
        )
        grades = _require_map(
            assignment["grades"], f"Base school_years.{year}.assignment.grades"
        )
        if not grades:
            raise ValueError(
                f"Base school_years.{year}.assignment.grades must not be empty."
            )
        for grade, grade_value in grades.items():
            _canonical_grade(grade, f"Base school_years.{year}.assignment grade key")
            grade_entry = _require_map(
                grade_value,
                f"Base school_years.{year}.assignment.grades.{grade}",
            )
            _unknown_keys(
                grade_entry,
                _ASSIGNMENT_GRADE_KEYS,
                f"school_years.{year}.assignment.grades.{grade}",
            )
            if set(grade_entry) != _ASSIGNMENT_GRADE_KEYS:
                raise ValueError(
                    f"Base school_years.{year}.assignment.grades.{grade} "
                    "must define profiles."
                )
            profiles = _require_map(
                grade_entry["profiles"],
                f"Base school_years.{year}.assignment.grades.{grade}.profiles",
            )
            if not profiles:
                raise ValueError(
                    f"Base school_years.{year}.assignment.grades.{grade}.profiles "
                    "must not be empty."
                )
            for profile, profile_value in profiles.items():
                _nonempty_string(
                    profile,
                    f"Base school_years.{year}.assignment profile key",
                )
                variants = _require_map(
                    profile_value,
                    f"Base school_years.{year}.assignment.grades.{grade}.profiles."
                    f"{profile}",
                )
                _unknown_keys(
                    variants,
                    {"standard", "mission_bay"},
                    f"school_years.{year}.{grade}.{profile} variant",
                )
                if not variants:
                    raise ValueError(
                        f"Base school_years.{year} grade {grade} profile {profile!r} "
                        "must define a standard or mission_bay variant."
                    )
                for variant, variant_value in variants.items():
                    bundle = _require_map(
                        variant_value,
                        f"Base school_years.{year}.{grade}.{profile}.{variant}",
                    )
                    _unknown_keys(
                        bundle,
                        _ASSIGNMENT_VARIANT_KEYS,
                        f"school_years.{year}.{grade}.{profile}.{variant}",
                    )
                    missing_bundle = {"programs", "schools"} - set(bundle)
                    if missing_bundle:
                        raise ValueError(
                            f"Base school_years.{year} grade {grade} profile "
                            f"{profile!r} variant {variant!r} is missing: "
                            f"{sorted(missing_bundle)}."
                        )
                    for role, source in bundle.items():
                        _validate_catalog_reference(
                            source,
                            f"Base school_years.{year}.{grade}.{profile}."
                            f"{variant}.{role}",
                            files,
                        )


def _validate_geographies(value: Any, files: Mapping[str, Any]) -> None:
    geographies = _require_map(value, "Base geographies")
    if not geographies:
        raise ValueError("Base geographies must not be empty.")
    required = {"blocks", "crosswalk", "adjacency", "manual_edges"}
    for vintage, bundle_value in geographies.items():
        _geography_vintage(vintage, "Base geographies key")
        bundle = _require_map(bundle_value, f"Base geographies.{vintage}")
        _unknown_keys(bundle, _GEOGRAPHY_KEYS, f"geographies.{vintage}")
        missing = sorted(required - set(bundle))
        if missing:
            raise ValueError(f"Base geographies.{vintage} is missing keys: {missing}.")
        for key in ("blocks", "blockgroups", "tracts", "crosswalk"):
            if key in bundle:
                _validate_catalog_reference(
                    bundle[key], f"Base geographies.{vintage}.{key}", files
                )
        adjacency = _require_map(
            bundle["adjacency"], f"Base geographies.{vintage}.adjacency"
        )
        if set(adjacency) != {"block", "blockgroup", "tract"}:
            raise ValueError(
                f"Base geographies.{vintage}.adjacency must define block, "
                "blockgroup, and tract."
            )
        for unit, source in adjacency.items():
            _validate_catalog_reference(
                source, f"Base geographies.{vintage}.adjacency.{unit}", files
            )
        _validate_source_ref(
            bundle["manual_edges"], f"Base geographies.{vintage}.manual_edges"
        )


def _validate_student_frl_estimates(value: Any, files: Mapping[str, Any]) -> None:
    estimates = _require_map(value, "Base student_frl_estimates")
    if not estimates:
        raise ValueError("Base student_frl_estimates must not be empty.")
    for name, source in estimates.items():
        _nonempty_string(name, "Base student_frl_estimates key")
        _validate_catalog_reference(source, f"Base student_frl_estimates.{name}", files)


def _validate_base(payload: dict[str, Any], path: Path) -> None:
    _unknown_keys(payload, _BASE_KEYS, "base")
    missing = sorted(_BASE_KEYS - set(payload))
    if missing:
        raise ValueError(f"Base configuration {path} is missing keys: {missing}.")
    version = payload["schema_version"]
    if version != 2:
        raise ValueError("Base schema_version must be 2 for the school-year registry.")
    roots = _require_map(payload["roots"], "Base roots")
    if "cache" not in roots or "data" not in roots:
        raise ValueError("Base roots must define both 'data' and 'cache'.")
    invalid_roots = sorted(set(roots) & _SPECIAL_ROOTS)
    if invalid_roots:
        raise ValueError(f"Base cannot replace special roots: {invalid_roots}.")
    for name, value in roots.items():
        if not isinstance(name, str) or (
            value is not None and not isinstance(value, str | os.PathLike)
        ):
            raise ValueError(
                "Base roots must map names to path strings or null required roots."
            )
    if roots["data"] is None or roots["cache"] is None:
        raise ValueError("Base data and cache roots cannot be null.")
    files = _require_map(payload["files"], "Base files")
    for file_id, source in files.items():
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("Base file IDs must be non-empty strings.")
        if not isinstance(source, dict) or not (set(source) & _SOURCE_KEYS):
            raise ValueError(
                f"Catalog file {file_id!r} must be a direct source object."
            )
        _validate_source_ref(source, f"files.{file_id}")
    _validate_geographies(payload["geographies"], files)
    _validate_student_frl_estimates(payload["student_frl_estimates"], files)
    _validate_school_years(payload["school_years"], files)


def _validate_scenario(payload: dict[str, Any], path: Path) -> None:
    _unknown_keys(payload, _SCENARIO_KEYS, "scenario")
    missing = sorted(_SCENARIO_KEYS - set(payload))
    if missing:
        raise ValueError(f"Scenario configuration {path} is missing keys: {missing}.")
    if not isinstance(payload["id"], str) or not payload["id"].strip():
        raise ValueError("Scenario id must be a non-empty string.")
    sources = _require_map(payload["sources"], "Scenario sources")
    for role, source in sources.items():
        if not isinstance(role, str) or "." not in role:
            raise ValueError(
                f"Scenario source roles must be flat dotted names, got {role!r}."
            )
        _validate_source_ref(source, f"sources.{role}")
    _validate_filters(payload["filters"], "Scenario filters")


def _root_paths(
    base_roots: Mapping[str, Any],
    base_dir: Path,
    overrides: Mapping[str, Any],
    override_dir: Path,
    environ: Mapping[str, str],
) -> dict[str, Path]:
    roots = {
        name: _absolute_path(value, base_dir)
        for name, value in base_roots.items()
        if value is not None
    }
    if environ.get("SFUSD_DATA_ROOT"):
        roots["data"] = _absolute_path(environ["SFUSD_DATA_ROOT"], REPOSITORY_ROOT)
    if environ.get("SFUSD_CACHE_ROOT"):
        roots["cache"] = _absolute_path(environ["SFUSD_CACHE_ROOT"], REPOSITORY_ROOT)

    root_overrides = _require_map(overrides.get("roots", {}), "Overrides roots")
    unknown = sorted(set(root_overrides) - set(base_roots))
    if unknown:
        raise ValueError(f"Unknown root override names: {unknown}.")
    for name, value in root_overrides.items():
        if not isinstance(value, str | os.PathLike):
            raise ValueError(f"Root override {name!r} must be a path string.")
        roots[name] = _absolute_path(value, override_dir)
    return roots


def _source_base(
    root: str | None, roots: Mapping[str, Path], default_dir: Path
) -> Path:
    if root is None:
        return default_dir
    if root == "package":
        return PACKAGE_CONFIG_ROOT
    if root == "repository":
        return REPOSITORY_ROOT
    try:
        return roots[root]
    except KeyError as exc:
        known = sorted(set(roots) | _SPECIAL_ROOTS)
        raise ValueError(
            f"Unknown source root {root!r}; expected one of {known}."
        ) from exc


def _resolve_direct_source(
    source: Mapping[str, Any],
    roots: Mapping[str, Path],
    default_dir: Path,
    *,
    catalog_id: str | None = None,
) -> ResolvedSource:
    base = _source_base(source.get("root"), roots, default_dir)
    path = _absolute_path(source["path"], base)
    companion_paths = [
        _absolute_path(value, base) for value in source.get("companions", [])
    ]
    if path.suffix.lower() == ".shp":
        for suffix in (".dbf", ".shx", ".prj"):
            companion = path.with_suffix(suffix)
            if companion not in companion_paths:
                companion_paths.append(companion)
    return ResolvedSource(
        path=path,
        companions=tuple(companion_paths),
        classification=source.get("classification", "unspecified"),
        catalog_id=catalog_id,
        geography_vintage=source.get("geography_vintage"),
    )


def _resolve_source_value(
    value: Any,
    catalog: Mapping[str, Any],
    roots: Mapping[str, Path],
    scenario_dir: Path,
    base_dir: Path,
) -> Any:
    if isinstance(value, str):
        if Path(value).is_absolute():
            return ResolvedSource(path=_absolute_path(value, scenario_dir))
        try:
            catalog_source = catalog[value]
        except KeyError as exc:
            raise ValueError(
                f"Unknown catalog file ID {value!r}; use a direct source object "
                "for non-catalog paths."
            ) from exc
        return _resolve_direct_source(catalog_source, roots, base_dir, catalog_id=value)
    if isinstance(value, list):
        return tuple(
            _resolve_source_value(child, catalog, roots, scenario_dir, base_dir)
            for child in value
        )
    if isinstance(value, dict) and set(value) & _SOURCE_KEYS:
        return _resolve_direct_source(value, roots, scenario_dir)
    if isinstance(value, dict):
        return MappingProxyType(
            {
                name: _resolve_source_value(
                    child, catalog, roots, scenario_dir, base_dir
                )
                for name, child in value.items()
            }
        )
    raise TypeError(f"Unsupported source reference {value!r}.")


def _run_payload(
    run_config: str | os.PathLike[str] | Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    if isinstance(run_config, Mapping):
        return copy.deepcopy(dict(run_config)), REPOSITORY_ROOT
    run_path = _absolute_path(run_config, REPOSITORY_ROOT)
    return _load_yaml(run_path, "run configuration"), run_path.parent


def _scenario_path(value: Any, run_dir: Path) -> Path:
    if not isinstance(value, str | os.PathLike):
        raise ValueError("Run configuration scenario must be a name or YAML path.")
    text = os.fspath(value)
    bundled = PACKAGE_CONFIG_ROOT / "scenarios" / f"{text}.yaml"
    if not Path(text).is_absolute() and bundled.is_file():
        return bundled.resolve()
    path = _absolute_path(text, run_dir)
    if not path.is_file():
        raise ValueError(f"Unknown bundled scenario or scenario YAML path: {text!r}.")
    return path


def _optimization_sources(
    filters: Mapping[str, Any], school_years: Mapping[str, Any]
) -> dict[str, Any]:
    population = filters["student_population"]
    selected: list[str] = []
    for year in filters["years"]:
        entry = school_years.get(year)
        source = (
            entry.get("optimization", {}).get("students", {}).get(population)
            if isinstance(entry, Mapping)
            else None
        )
        if source is None:
            available_years = [
                registered_year
                for registered_year, registered in school_years.items()
                if population in registered.get("optimization", {}).get("students", {})
            ]
            raise ValueError(
                "Optimization students are unavailable for school year "
                f"{year!r} and population {population!r}; available years are "
                f"{available_years}."
            )
        selected.append(source)
    return {"optimization.students": selected}


def _assignment_sources(
    filters: Mapping[str, Any], school_years: Mapping[str, Any]
) -> dict[str, Any]:
    year = filters["year"]
    grades = filters["grades"]
    if len(grades) != 1:
        raise ValueError(
            "Assignment source resolution requires exactly one grade; selected "
            f"grades are {list(grades)}."
        )
    grade = grades[0]
    entry = school_years.get(year)
    assignment = entry.get("assignment") if isinstance(entry, Mapping) else None
    if not isinstance(assignment, Mapping):
        available = [
            registered_year
            for registered_year, registered in school_years.items()
            if "assignment" in registered
        ]
        raise ValueError(
            "Assignment sources are unavailable for school year "
            f"{year!r}: no program/school bundle is registered; available years "
            f"are {available}."
        )

    grades_registry = assignment["grades"]
    if grade not in grades_registry:
        raise ValueError(
            f"Assignment grade {grade!r} is unavailable for school year {year!r}; "
            f"available grades are {list(grades_registry)}."
        )
    profile = filters["capacity_profile"]
    profiles = grades_registry[grade]["profiles"]
    if profile not in profiles:
        raise ValueError(
            f"Assignment capacity profile {profile!r} is unavailable for school "
            f"year {year!r}, grade {grade!r}; available profiles are "
            f"{list(profiles)}."
        )
    variant = "mission_bay" if filters["include_mission_bay"] else "standard"
    variants = profiles[profile]
    if variant not in variants:
        policy = "included" if filters["include_mission_bay"] else "excluded"
        available = [
            "Mission Bay included" if key == "mission_bay" else "Mission Bay excluded"
            for key in variants
        ]
        raise ValueError(
            f"Assignment sources are unavailable for school year {year!r}, grade "
            f"{grade!r}, capacity profile {profile!r}, with Mission Bay {policy}; "
            f"available variants are {available}."
        )

    population = filters["student_population"]
    students = assignment["students"].get(population)
    if students is None:  # pragma: no cover - base validation currently requires both
        raise ValueError(
            f"Assignment {population!r} students are unavailable for school year "
            f"{year!r}."
        )
    bundle = variants[variant]
    generated = {
        "assignment.students": students,
        "assignment.programs": bundle["programs"],
        "assignment.schools": bundle["schools"],
        "assignment.school_coordinates": bundle["schools"],
    }
    if "programs_catalog" in bundle:
        generated["assignment.programs.catalog"] = bundle["programs_catalog"]
    return generated


def _geography_sources(
    filters: Mapping[str, Mapping[str, Any]], geographies: Mapping[str, Any]
) -> dict[str, Any]:
    generated: dict[str, Any] = {}
    for group, group_filters in filters.items():
        vintage = group_filters["geography_vintage"]
        bundle = geographies.get(vintage)
        if not isinstance(bundle, Mapping):
            raise ValueError(
                f"Census geography vintage {vintage!r} is unavailable; available "
                f"vintages are {list(geographies)}."
            )
        if group == "optimization":
            generated.update(
                {
                    "optimization.census": bundle["blocks"],
                    "optimization.crosswalk": bundle["crosswalk"],
                    "optimization.adjacency": bundle["adjacency"],
                    "optimization.manual_edges": bundle["manual_edges"],
                }
            )
            for key in ("blockgroups", "tracts"):
                if key in bundle:
                    generated[f"optimization.geography.{key}"] = bundle[key]
        else:
            for key in ("blocks", "blockgroups", "tracts", "crosswalk"):
                if key in bundle:
                    generated[f"assignment.geography.{key}"] = bundle[key]
    return generated


def _student_frl_sources(
    filters: Mapping[str, Mapping[str, Any]], estimates: Mapping[str, Any]
) -> dict[str, Any]:
    generated: dict[str, Any] = {}
    for group, group_filters in filters.items():
        estimate = group_filters["frl_estimate"]
        if estimate is None:
            continue
        source = estimates.get(estimate)
        if source is None:
            raise ValueError(
                f"Student FRL estimate {estimate!r} is unavailable; available "
                f"estimates are {list(estimates)}."
            )
        generated[f"{group}.frl_estimate"] = source
    return generated


def _registry_sources(
    filters: Mapping[str, Mapping[str, Any]],
    school_years: Mapping[str, Any],
    geographies: Mapping[str, Any],
    student_frl_estimates: Mapping[str, Any],
) -> dict[str, Any]:
    generated = _geography_sources(filters, geographies)
    generated.update(_student_frl_sources(filters, student_frl_estimates))
    if "optimization" in filters:
        generated.update(_optimization_sources(filters["optimization"], school_years))
    if "assignment" in filters:
        generated.update(_assignment_sources(filters["assignment"], school_years))
    return generated


def _validate_student_frl_source_vintages(
    filters: Mapping[str, Mapping[str, Any]], source_values: Mapping[str, Any]
) -> None:
    for group, group_filters in filters.items():
        estimate = group_filters["frl_estimate"]
        if estimate is None:
            continue
        role = f"{group}.frl_estimate"
        source = source_values.get(role)
        if not isinstance(source, ResolvedSource):
            raise ValueError(
                f"Student FRL source role {role!r} must resolve to exactly one file."
            )
        target_vintage = group_filters["geography_vintage"]
        if source.geography_vintage != target_vintage:
            raise ValueError(
                f"Student FRL estimate {estimate!r} uses Census geography "
                f"{source.geography_vintage!r}, but {group}.geography_vintage is "
                f"{target_vintage!r}."
            )


def load_scenario(
    run_config: str | os.PathLike[str] | Mapping[str, Any],
    *,
    base_path: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> DataScenario:
    """Load bundled base + one scenario + strict run overrides."""
    run, run_dir = _run_payload(run_config)
    _unknown_keys(run, _RUN_KEYS, "run configuration")
    missing = sorted(_RUN_KEYS - set(run))
    if missing:
        raise ValueError(f"Run configuration is missing keys: {missing}.")
    overrides = _require_map(run["overrides"], "Run overrides")
    _unknown_keys(overrides, _OVERRIDE_KEYS, "override")

    base_file = _absolute_path(base_path or BASE_CONFIG_PATH, REPOSITORY_ROOT)
    base = _load_yaml(base_file, "base configuration")
    _validate_base(base, base_file)

    scenario_file = _scenario_path(run["scenario"], run_dir)
    scenario = _load_yaml(scenario_file, "scenario configuration")
    _validate_scenario(scenario, scenario_file)

    override_sources = _require_map(overrides.get("sources", {}), "Overrides sources")
    for role, source in override_sources.items():
        if not isinstance(role, str) or "." not in role:
            raise ValueError(
                f"Override source roles must be flat dotted names, got {role!r}."
            )
        _validate_source_ref(source, f"overrides.sources.{role}", partial=True)
    override_filters = _validate_filters(
        overrides.get("filters", {}), "Override filters", partial=True
    )

    roots = _root_paths(
        base["roots"],
        base_file.parent,
        overrides,
        run_dir,
        os.environ if environ is None else environ,
    )
    declared_sources = _anchor_direct_paths(scenario["sources"], scenario_file.parent)
    declared_overrides = _anchor_direct_paths(
        override_sources, run_dir, scenario["sources"]
    )
    filters = _deep_merge(scenario["filters"], override_filters)
    _validate_filters(filters, "Merged filters")

    if "optimization" in filters:
        declared_sources.pop("optimization.students", None)
    if "assignment" in filters:
        for role in _ANNUAL_ASSIGNMENT_ROLES:
            declared_sources.pop(role, None)
    generated_sources = _registry_sources(
        filters,
        base["school_years"],
        base["geographies"],
        base["student_frl_estimates"],
    )
    sources = _deep_merge(declared_sources, generated_sources)
    sources = _deep_merge(sources, declared_overrides)
    for role, source in sources.items():
        _validate_source_ref(source, f"merged sources.{role}")

    source_values = {
        role: _resolve_source_value(
            source,
            base["files"],
            roots,
            run_dir if run_dir != REPOSITORY_ROOT else scenario_file.parent,
            base_file.parent,
        )
        for role, source in sources.items()
    }
    _validate_student_frl_source_vintages(filters, source_values)
    return DataScenario(
        id=scenario["id"],
        schema_version=base["schema_version"],
        roots=MappingProxyType(roots),
        filters=_freeze(filters),
        _source_values=MappingProxyType(source_values),
    )


__all__ = [
    "BASE_CONFIG_PATH",
    "PACKAGE_CONFIG_ROOT",
    "REPOSITORY_ROOT",
    "DataScenario",
    "ResolvedSource",
    "anchor_data_config",
    "load_scenario",
]
