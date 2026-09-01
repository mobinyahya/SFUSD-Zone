"""Content-addressed cache storage with caller-owned artifact schemas.

Create namespaces with ``store.namespace(..., schema_version=1)``. The explicit
version controls cache identity and the ``v<schema_version>`` layout; it is
independent from the data-catalog version embedded in source manifests.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import pickle
import re
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

from loaders.config import DataScenario

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _normalized(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path | os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str | int | float | bool):
                raise TypeError(f"Cache parameter key {key!r} is not JSON-compatible.")
            normalized[str(key)] = _normalized(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, list | tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [_normalized(item) for item in value]
        return sorted(items, key=_canonical_json)
    raise TypeError(f"Cache parameter value {value!r} is not JSON-compatible.")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_identity(value: Any) -> str:
    if value is None or value is pd.NA:
        raise ValueError("Identity values cannot be null.")
    if isinstance(value, bool):
        raise ValueError("Identity values cannot be booleans.")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Identity values must be finite.")
        if number.is_integer():
            return str(int(number))
        return format(number, ".17g")
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
        raise ValueError("Identity values cannot be empty.")
    raise TypeError(f"Identity value {value!r} is not a supported scalar.")


def identity_fingerprint(identities: Iterable[Any]) -> str:
    """Hash ordered normalized scalar identities without exposing their values."""
    digest = hashlib.sha256()
    for identity in identities:
        encoded = _normalized_identity(identity).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_name(name: str, label: str) -> str:
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"{label} must contain only letters, numbers, '.', '_', and '-'."
        )
    return name


class CacheStore:
    """Create cache namespaces rooted at a resolved data scenario."""

    def __init__(self, scenario: DataScenario):
        self.scenario = scenario
        self.root = scenario.cache_root

    def namespace(
        self,
        artifact: str,
        parameters: Mapping[str, Any] | None = None,
        *,
        schema_version: int,
        roles: str | Iterable[str] | None = None,
        classification: str = "derived",
    ) -> CacheNamespace:
        """Resolve a namespace for one explicit artifact schema version."""
        artifact = _validate_name(artifact, "Artifact")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version <= 0
        ):
            raise ValueError("Cache schema_version must be a positive integer.")
        if not isinstance(classification, str) or not classification:
            raise ValueError("Cache classification must be a non-empty string.")
        normalized_parameters = _normalized(parameters or {})
        source_manifest = self.scenario.source_manifest(roles)
        identity = {
            "artifact": artifact,
            "schema_version": schema_version,
            "parameters": normalized_parameters,
            "sources": source_manifest,
            "classification": classification,
        }
        key = _sha256_bytes(_canonical_json(identity).encode("utf-8"))
        return CacheNamespace(
            root=self.root,
            artifact=artifact,
            schema_version=schema_version,
            key=key,
            parameters=normalized_parameters,
            source_manifest=source_manifest,
            classification=classification,
        )


class CacheNamespace:
    """Validated multi-payload storage for one content-addressed cache key."""

    def __init__(
        self,
        *,
        root: Path,
        artifact: str,
        schema_version: int,
        key: str,
        parameters: Any,
        source_manifest: Mapping[str, Any],
        classification: str,
    ):
        self.root = root
        self.artifact = artifact
        self.schema_version = schema_version
        self.key = key
        self.parameters = parameters
        self.source_manifest = source_manifest
        self.classification = classification
        self._restricted = "restricted" in classification.casefold()
        self.version_dir = root / artifact / f"v{schema_version}"
        self.path = self.version_dir / key
        self.manifest_path = self.path / "manifest.json"
        self.lock_path = self.version_dir / f".{key}.lock"

    def payload_path(self, name: str) -> Path:
        return self.path / _validate_name(name, "Payload name")

    def reference(self, payload: str) -> dict[str, Any]:
        """Return a path-free, serializable reference to one payload."""
        payload = _validate_name(payload, "Payload name")
        return {
            "artifact": self.artifact,
            "schema_version": self.schema_version,
            "key": self.key,
            "classification": self.classification,
            "parameters": self.parameters,
            "roles": list(self.source_manifest["sources"]),
            "payload": payload,
        }

    def load_pickle(self, name: str) -> Any | None:
        """Load a validated pickle payload, or return ``None`` on any miss."""
        name = _validate_name(name, "Payload name")
        with self._lock(shared=True):
            payload = self._validated_payload(name, "pickle")
            if payload is None:
                return None
            try:
                with payload.open("rb") as stream:
                    return pickle.load(stream)
            except (
                OSError,
                EOFError,
                pickle.PickleError,
                AttributeError,
                ImportError,
                TypeError,
                ValueError,
            ):
                return None

    def save_pickle(self, name: str, value: Any) -> Path:
        """Atomically save a pickle payload and update the namespace manifest."""
        return self._save_payload(name, pickle.dumps(value), "pickle")

    def load_dataframe(self, name: str, **read_csv_kwargs: Any) -> pd.DataFrame | None:
        """Load a validated CSV payload as a DataFrame."""
        name = _validate_name(name, "Payload name")
        with self._lock(shared=True):
            payload = self._validated_payload(name, "csv")
            if payload is None:
                return None
            try:
                return pd.read_csv(payload, **read_csv_kwargs)
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
                return None

    def save_dataframe(
        self,
        name: str,
        frame: pd.DataFrame,
        *,
        index: bool = False,
        **to_csv_kwargs: Any,
    ) -> Path:
        """Atomically save a DataFrame as UTF-8 CSV."""
        payload = frame.to_csv(index=index, **to_csv_kwargs).encode("utf-8")
        return self._save_payload(name, payload, "csv")

    load_csv = load_dataframe
    save_csv = save_dataframe

    def manifest(self) -> dict[str, Any] | None:
        """Return the trusted namespace manifest, if it is valid."""
        with self._lock(shared=True):
            return self._read_valid_manifest()

    def _expected_manifest_values(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "schema_version": self.schema_version,
            "key": self.key,
            "parameters": self.parameters,
            "sources": self.source_manifest,
            "classification": self.classification,
        }

    def _read_valid_manifest(self) -> dict[str, Any] | None:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None
        expected = self._expected_manifest_values()
        if any(manifest.get(key) != value for key, value in expected.items()):
            return None
        if not isinstance(manifest.get("created_at"), str) or not isinstance(
            manifest.get("payloads"), dict
        ):
            return None
        return manifest

    def _validated_payload(self, name: str, payload_format: str) -> Path | None:
        manifest = self._read_valid_manifest()
        if manifest is None:
            return None
        record = manifest["payloads"].get(name)
        if not isinstance(record, dict):
            return None
        if record.get("file") != name or record.get("format") != payload_format:
            return None
        checksum = record.get("sha256")
        if not isinstance(checksum, str):
            return None
        path = self.payload_path(name)
        try:
            if _sha256_file(path) != checksum:
                return None
        except OSError:
            return None
        return path

    def _save_payload(self, name: str, payload: bytes, payload_format: str) -> Path:
        name = _validate_name(name, "Payload name")
        destination = self.payload_path(name)
        with self._lock(shared=False):
            self._ensure_directory(self.path)
            self._atomic_write(destination, payload)
            manifest = self._read_valid_manifest()
            if manifest is None:
                manifest = {
                    **self._expected_manifest_values(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "payloads": {},
                }
            manifest["payloads"][name] = {
                "file": name,
                "format": payload_format,
                "sha256": _sha256_bytes(payload),
                "size": len(payload),
            }
            encoded = (_canonical_json(manifest) + "\n").encode("utf-8")
            self._atomic_write(self.manifest_path, encoded)
        return destination

    @contextmanager
    def _lock(self, *, shared: bool):
        self._ensure_directory(self.version_dir)
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o660)
        with os.fdopen(descriptor, "r+b") as stream:
            if self._restricted:
                os.fchmod(stream.fileno(), 0o660)
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _ensure_directory(self, path: Path) -> None:
        if not self._restricted:
            path.mkdir(parents=True, exist_ok=True)
            return

        self.root.mkdir(parents=True, exist_ok=True)
        current = self.root
        for part in path.relative_to(self.root).parts:
            current /= part
            current.mkdir(mode=0o770, exist_ok=True)
            current.chmod(0o770)

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                if self._restricted:
                    os.fchmod(stream.fileno(), 0o660)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


__all__ = ["CacheNamespace", "CacheStore", "identity_fingerprint"]
