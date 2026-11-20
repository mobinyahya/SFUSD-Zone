"""Content-addressed caches of the loaders' normalised tables.

Every sweep phase starts a fresh process that loads the same tables, so the
normalised output of the public loaders (``load_student_records``,
``load_school_records``, ``load_program_records``, ``load_census_geometry``,
``load_geography_crosswalk``) is kept at two levels:

* **On disk**, under ``<cache root>/<artifact>/v<TABLE_CACHE_SCHEMA_VERSION>/
  <sha256>/`` through :class:`loaders.cache.CacheStore`, as one protocol-5
  pickle. The key is the exact content of every source role the build may read,
  the filter values it may read, the function's own arguments, the loaders'
  source code, and the versions of the libraries that shape a frame. One
  process builds an entry under the key's build lock; any other process that
  wants it meanwhile waits on that lock and then loads it.
* **In process**, as the same pickled bytes (LRU, at most
  :data:`MEMO_LIMIT_BYTES`). Each call unpickles a fresh copy, so a caller may
  mutate the frame it gets -- its cells included -- without touching what the
  next caller sees.

A build records every role and filter it reads (``DataScenario`` reports them
through ``SCENARIO_ACCESS_RECORDERS``). An entry whose build read an input its
key does not name is neither saved nor memoised, so a key can only be too
wide, never stale.

``SFUSD_TABLE_CACHE`` controls both levels: unset, ``1`` or ``on`` uses both;
``memory`` keeps only the in-process memo; ``0``, ``off``, ``false`` or ``no``
computes every call afresh. Any cache failure (unwritable root, corrupt entry,
lock error) falls back to computing the table, and an error raised by the
build itself propagates unchanged.

Warnings: the loaders raise :class:`LoaderWarning` through :func:`loader_warn`.
A build stores the messages it emitted with its entry, and a hit replays them,
so a cached table warns exactly as a computed one would. Each distinct message
is then shown once per process: :func:`loader_warn` keeps its own registry of
messages shown, because the interpreter's ``once`` registry is cleared by every
``warnings.catch_warnings`` block (pandas and geopandas open many). A filter
that resolves a ``LoaderWarning`` to ``always`` or ``error`` -- such as
``pytest.warns`` -- still sees every emission; ``ignore`` still hides it.
:func:`show_loader_warnings_once` puts a ``once`` filter for the category ahead
of a worker's own ``always`` filter.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, TypeVar

from loaders.config import SCENARIO_ACCESS_RECORDERS, DataScenario

#: Bump when a cached table's contents change for reasons the key does not
#: see. Source edits to the loaders already re-key every entry (see
#: ``_code_fingerprint``), so this is for anything else.
TABLE_CACHE_SCHEMA_VERSION = 1
TABLE_CACHE_ENV = "SFUSD_TABLE_CACHE"
#: Upper bound on the pickled bytes the in-process memo holds. The tables of
#: one scenario take a few MB; the bound only matters for a process that
#: walks many scenarios.
MEMO_LIMIT_BYTES = 256 * 1024 * 1024
_PAYLOAD = "table.pkl"
_DISABLED = {"0", "off", "false", "no"}
_MEMORY_ONLY = {"memory", "mem"}
#: Source files whose behaviour a cached table depends on.
_CODE_FILES = ("cache.py", "config.py", "geography.py", "table_cache.py", "tables.py")
_LIBRARIES = ("numpy", "pandas", "geopandas", "shapely", "pyogrio", "pyproj")

T = TypeVar("T")


class LoaderWarning(UserWarning):
    """A data-quality notice from the shared loaders, shown once per process."""


def show_loader_warnings_once() -> None:
    """Show each distinct :class:`LoaderWarning` message once in this process.

    That is already the default; call this at worker start to put a ``once``
    filter for the category *ahead* of any filter the worker set itself, such
    as ``warnings.simplefilter("always")``.
    """
    warnings.filterwarnings("once", category=LoaderWarning)


warnings.filterwarnings("once", category=LoaderWarning, append=True)
#: Loader-warning messages already shown in this process.
_SHOWN: set[str] = set()
_SHOWN_LOCK = threading.Lock()


def _action(message: str) -> str:
    """The action the current filters give a ``LoaderWarning`` with ``message``.

    Filters tied to a module or line are skipped: a loader warning is
    attributed to whichever caller loaded the table first.
    """
    for action, pattern, category, module, lineno in warnings.filters:
        if module is not None or lineno:
            continue
        if not issubclass(LoaderWarning, category):
            continue
        if pattern is None or pattern.match(message):
            return action
    return warnings.defaultaction


@dataclass
class _Recorder:
    """Inputs one build read, and the loader warnings it emitted."""

    roles: set[str] = field(default_factory=set)
    filters: set[tuple[str, str]] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    def note_role(self, role: str) -> None:
        self.roles.add(role)

    def note_filter(self, group: str, key: str) -> None:
        self.filters.add((group, key))

    def note_warning(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


def loader_warn(message: str, *, stacklevel: int = 2) -> None:
    """Emit a :class:`LoaderWarning` and record it for every active build."""
    for recorder in SCENARIO_ACCESS_RECORDERS.get():
        recorder.note_warning(message)
    if _action(message) not in {"always", "error"}:
        with _SHOWN_LOCK:
            if message in _SHOWN:
                return
            _SHOWN.add(message)
        if not _first_in_shared_dir(message):
            return
    warnings.warn(message, LoaderWarning, stacklevel=stacklevel + 1)


#: Directory of one marker file per warning shown; when set, a warning is shown
#: once across every process that shares the directory (a Slurm allocation
#: points it at node-local /dev/shm), not once per process.
WARN_ONCE_DIR_ENV = "SFUSD_WARN_ONCE_DIR"


def _first_in_shared_dir(message: str) -> bool:
    """Whether this process is the first sharing the warn-once dir to show it."""
    directory = os.environ.get(WARN_ONCE_DIR_ENV)
    if not directory:
        return True
    marker = Path(directory) / hashlib.sha256(message.encode()).hexdigest()[:32]
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    except FileExistsError:
        return False
    except OSError:
        return True
    return True


def _mode() -> str:
    value = os.environ.get(TABLE_CACHE_ENV, "").strip().lower()
    if value in _DISABLED:
        return "off"
    if value in _MEMORY_ONLY:
        return "memory"
    return "disk"


_CODE_FINGERPRINT: str | None = None
_LIBRARY_VERSIONS: dict[str, str] | None = None


def _code_fingerprint() -> str:
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is None:
        digest = hashlib.sha256()
        root = Path(__file__).resolve().parent
        for name in _CODE_FILES:
            payload = (root / name).read_bytes()
            digest.update(name.encode())
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        _CODE_FINGERPRINT = digest.hexdigest()
    return _CODE_FINGERPRINT


def _library_versions() -> dict[str, str]:
    global _LIBRARY_VERSIONS
    if _LIBRARY_VERSIONS is None:
        versions = {"python": "%d.%d" % sys.version_info[:2]}
        for name in _LIBRARIES:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "absent"
        _LIBRARY_VERSIONS = versions
    return _LIBRARY_VERSIONS


class _Memo:
    """Thread-safe LRU of pickled entries, bounded by total bytes."""

    def __init__(self, limit: int):
        self.limit = limit
        self.size = 0
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            payload = self._entries.get(key)
            if payload is not None:
                self._entries.move_to_end(key)
            return payload

    def put(self, key: str, payload: bytes) -> None:
        if len(payload) > self.limit:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self.size -= len(previous)
            self._entries[key] = payload
            self.size += len(payload)
            while self.size > self.limit:
                _, evicted = self._entries.popitem(last=False)
                self.size -= len(evicted)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.size = 0


_MEMO = _Memo(MEMO_LIMIT_BYTES)


def clear_table_memo() -> None:
    """Forget every in-process table (the disk cache is untouched)."""
    _MEMO.clear()


def table_memo_bytes() -> int:
    """Pickled bytes the in-process memo currently holds."""
    return _MEMO.size


@dataclass(frozen=True)
class TableKey:
    """What one cached table may depend on.

    ``roles`` are the source roles the build may resolve (present or not);
    ``filters`` the ``(group, key)`` filters it may read, where a key of
    ``"*"`` admits the whole group, whose values then join the key.
    """

    artifact: str
    roles: frozenset[str]
    filters: frozenset[tuple[str, str]]
    parameters: Mapping[str, Any]

    def admits(self, recorder: _Recorder) -> bool:
        if not recorder.roles <= self.roles:
            return False
        groups = {group for group, key in self.filters if key == "*"}
        return all(
            group in groups or (group, key) in self.filters
            for group, key in recorder.filters
        )


def _filter_values(
    scenario: DataScenario, filters: Iterable[tuple[str, str]]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group, key in sorted(filters):
        if key == "*":
            values[group] = _plain(scenario.filters.get(group, {}))
            # The registries a group's filters select are input data too.
            adjustments = scenario.capacity_adjustments(group)
            values[f"{group}:capacity_adjustments"] = _plain(adjustments)
            values[f"{group}:planned_schools"] = [
                _plain(item) for item in scenario.selected_planned_schools(group)
            ]
        else:
            values[f"{group}.{key}"] = _plain(scenario.filter(group, key, None))
    return values


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _classification(scenario: DataScenario, roles: Iterable[str]) -> str:
    present = set(scenario.roles)
    for role in roles:
        if role in present and any(
            "restricted" in source.classification.casefold()
            for source in scenario.sources(role)
        ):
            return "restricted"
    return "derived"


def _namespace(scenario: DataScenario, key: TableKey):
    from loaders.cache import CacheStore

    # Reading the scenario to build the key is not a read by the build.
    token = SCENARIO_ACCESS_RECORDERS.set(())
    try:
        parameters = {
            **key.parameters,
            "filters": _filter_values(scenario, key.filters),
            "roles": sorted(key.roles),
            "code": _code_fingerprint(),
            "libraries": _library_versions(),
        }
        manifest = {"sources": scenario.role_sources_manifest(key.roles)}
        classification = _classification(scenario, key.roles)
    finally:
        SCENARIO_ACCESS_RECORDERS.reset(token)
    return CacheStore(scenario).namespace(
        key.artifact,
        parameters,
        schema_version=TABLE_CACHE_SCHEMA_VERSION,
        classification=classification,
        source_manifest=manifest,
    )


def _note_outer(roles: Iterable[str], filters: Iterable[tuple[str, str]]) -> None:
    for recorder in SCENARIO_ACCESS_RECORDERS.get():
        recorder.roles.update(roles)
        recorder.filters.update(filters)


def _unpack(payload: bytes | None) -> dict[str, Any] | None:
    """A fresh copy of a pickled entry, or ``None`` if it cannot be read."""
    if payload is None:
        return None
    try:
        entry = pickle.loads(payload)
        if not isinstance(entry, dict) or not {
            "value",
            "warnings",
            "roles",
            "filters",
        } <= set(entry):
            return None
        return entry
    except Exception:  # noqa: BLE001 - a corrupt entry is a miss
        return None


def _replay(entry: Mapping[str, Any]) -> Any:
    """Return an entry's value as its build would: inputs noted, warnings shown."""
    _note_outer(entry["roles"], (tuple(item) for item in entry["filters"]))
    for message in entry["warnings"]:
        loader_warn(message, stacklevel=4)
    return entry["value"]


def _json_compatible(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list | tuple):
        return all(_json_compatible(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(name, str) and _json_compatible(item)
            for name, item in value.items()
        )
    return False


def cached_table(
    scenario: DataScenario,
    key: TableKey,
    build: Callable[[], T],
) -> T:
    """Return ``build()``, from the memo or disk cache when an entry exists."""
    mode = _mode()
    if mode == "off" or not _json_compatible(dict(key.parameters)):
        return build()
    try:
        namespace = _namespace(scenario, key)
    except Exception:  # noqa: BLE001 - an unkeyable call is simply uncached
        return build()

    entry = _unpack(_MEMO.get(namespace.key))
    if entry is None and mode == "disk":
        payload = _load(namespace)
        entry = _unpack(payload)
        if entry is not None:
            _MEMO.put(namespace.key, payload)
    if entry is not None:
        return _replay(entry)

    with ExitStack() as stack:
        locked = False
        if mode == "disk":
            try:
                stack.enter_context(namespace.build_lock())
                locked = True
            except Exception:  # noqa: BLE001
                locked = False
            if locked:
                # Another process may have built it while this one waited.
                payload = _load(namespace)
                entry = _unpack(payload)
                if entry is not None:
                    _MEMO.put(namespace.key, payload)
                    return _replay(entry)

        recorder = _Recorder()
        outer = SCENARIO_ACCESS_RECORDERS.get()
        token = SCENARIO_ACCESS_RECORDERS.set((*outer, recorder))
        try:
            value = build()
        finally:
            SCENARIO_ACCESS_RECORDERS.reset(token)
            _note_outer(recorder.roles, recorder.filters)

        if not key.admits(recorder):
            unkeyed = sorted(recorder.roles - key.roles)
            warnings.warn(
                f"Loader cache key {key.artifact!r} does not name every input "
                f"its build read (roles {unkeyed}, filters "
                f"{sorted(recorder.filters)}); the table is not cached.",
                RuntimeWarning,
                stacklevel=3,
            )
            return value
        try:
            payload = pickle.dumps(
                {
                    "value": value,
                    "warnings": recorder.warnings,
                    "roles": sorted(recorder.roles),
                    "filters": sorted(recorder.filters),
                },
                protocol=5,
            )
        except Exception:  # noqa: BLE001 - an unpicklable table is uncached
            return value
        _MEMO.put(namespace.key, payload)
        if locked:
            try:
                namespace.save_pickle_bytes(_PAYLOAD, payload)
            except Exception:  # noqa: BLE001 - an unwritable cache is a miss
                pass
        return value


def _load(namespace) -> bytes | None:
    try:
        return namespace.load_pickle_bytes(_PAYLOAD)
    except Exception:  # noqa: BLE001 - an unreadable cache is a miss
        return None


__all__ = [
    "MEMO_LIMIT_BYTES",
    "TABLE_CACHE_ENV",
    "TABLE_CACHE_SCHEMA_VERSION",
    "LoaderWarning",
    "TableKey",
    "cached_table",
    "clear_table_memo",
    "loader_warn",
    "show_loader_warnings_once",
    "table_memo_bytes",
]
