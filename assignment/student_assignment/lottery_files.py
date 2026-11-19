"""One gzipped long-format CSV of every lottery of one policy variant.

A policy variant's save folder used to hold one ``<variant>_iteration<N>.csv``
per lottery (25 small files per run, ~16 GB of them across a sweep). It now
holds two files:

``<variant>_lotteries.csv.gz``
    Every lottery's rows, grouped by lottery in ascending order, behind a
    leading integer ``lottery`` column. Columns that are constant or derivable
    are left out (see :data:`DERIVED_COLUMNS`).
``<variant>_lotteries.json``
    The sidecar: the schema version, the full column order, the values of the
    columns left out, the ``programno -> programcodes`` map, the row count of
    each lottery, and the data file's SHA-256. It is the commit point: it is
    replaced after the data file, and a reader trusts the data only when its
    checksum matches.

Each lottery's cells are written with the same ``DataFrame.to_csv`` call the
per-lottery file used, and :func:`read_lotteries` parses each lottery's rows
with its own ``pd.read_csv`` before putting the dropped columns back, so the
frames it returns are the ones ``pd.read_csv`` returned for the per-lottery
files: same values, dtypes and column order. The one deliberate difference is
``assignment_schema_version``, which is the current version.

Writers replace both files with ``os.replace`` under a directory lock, so a
reader never sees a partial file, and an interrupted run leaves the previous
pair (or none) in place.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .choice_ranks import ASSIGNMENT_SCHEMA_VERSION

#: Version of the long-format file pair. It equals the row schema version it
#: was introduced with (4); the row semantics are those of version 3.
LOTTERY_FILE_SCHEMA_VERSION = ASSIGNMENT_SCHEMA_VERSION
LOTTERY_FILE_FORMAT = "sfusd-assignment-lotteries"
DATA_SUFFIX = "_lotteries.csv.gz"
SIDECAR_SUFFIX = "_lotteries.json"
LOTTERY_COLUMN = "lottery"
#: Columns the long file may leave out, when every lottery agrees with the
#: sidecar: two constants, ``utility_rank`` when it is empty everywhere (the
#: utility model is off), and ``programcodes`` (derivable from programno).
DERIVED_COLUMNS = (
    "assignment_schema_version",
    "rank_basis",
    "utility_rank",
    "programcodes",
)
LEGACY_ITERATION_PATTERN = re.compile(
    r"^(?P<stem>.+)_iteration(?P<iteration>\d+)\.csv$"
)
_LOCK_TIMEOUT_SECONDS = 900.0
_STALE_LOCK_SECONDS = 1800.0
_GZIP_LEVEL = 6


def lottery_file_paths(legacy_iteration_path: str | Path) -> tuple[Path, Path]:
    """Return the ``(data, sidecar)`` paths for a per-lottery file's variant."""
    path = Path(legacy_iteration_path)
    match = LEGACY_ITERATION_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Not a per-lottery assignment path: {path}")
    stem = match.group("stem")
    return path.with_name(stem + DATA_SUFFIX), path.with_name(stem + SIDECAR_SUFFIX)


def sidecar_path_for(data_path: str | Path) -> Path:
    data_path = Path(data_path)
    if not data_path.name.endswith(DATA_SUFFIX):
        raise ValueError(f"Not a lottery data file: {data_path}")
    return data_path.with_name(data_path.name[: -len(DATA_SUFFIX)] + SIDECAR_SUFFIX)


def render_lottery(frame: pd.DataFrame) -> tuple[list[str], str]:
    """Render one lottery's frame exactly as its per-lottery CSV would be.

    Returns the column list and the CSV text (header included).
    """
    return list(frame.columns), frame.to_csv(index=False)


class LotteryFileError(ValueError):
    """A lottery file pair is missing, inconsistent, or corrupt."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _DirectoryLock:
    """An ``mkdir`` lock next to the data file; works on NFS."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        delay = 0.02
        while True:
            try:
                self.path.mkdir()
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > _STALE_LOCK_SECONDS:
                    try:
                        self.path.rmdir()
                    except OSError:
                        pass
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Timed out waiting for lock {self.path}")
                time.sleep(delay)
                delay = min(delay * 2, 0.5)

    def __exit__(self, *exc):
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass
        return False


def _lock_for(data_path: Path) -> _DirectoryLock:
    return _DirectoryLock(data_path.with_name(f".{data_path.name}.lock"))


class LotteryFile:
    """A parsed lottery file: its sidecar and each lottery's raw CSV lines."""

    def __init__(self, data_path: Path, sidecar: dict, lines: dict[int, list[str]]):
        self.data_path = data_path
        self.sidecar = sidecar
        self._lines = lines
        self._frames: dict[int, pd.DataFrame] = {}

    @property
    def lotteries(self) -> list[int]:
        return sorted(self._lines)

    @property
    def stored_columns(self) -> list[str]:
        return list(self.sidecar["stored_columns"])

    def lottery_text(self, lottery: int) -> str:
        """One lottery's rows as the CSV text of its stored columns."""
        header = ",".join(_csv_header_cells(self.stored_columns))
        return header + "\n" + "".join(self._lines[lottery])

    def frame(self, lottery: int) -> pd.DataFrame:
        """One lottery, rebuilt to the per-lottery file's exact frame."""
        if lottery not in self._lines:
            raise KeyError(lottery)
        cached = self._frames.get(lottery)
        if cached is None:
            cached = self._rebuild(lottery)
            self._frames[lottery] = cached
        return cached.copy()

    def _rebuild(self, lottery: int) -> pd.DataFrame:
        # Rebuild the per-lottery CSV text cell for cell, then parse it the
        # way the per-lottery file was parsed: values, dtypes and column
        # order are then those of ``pd.read_csv`` by construction.
        frame = pd.read_csv(io.StringIO(_full_text(self, lottery)))
        expected_rows = self.sidecar["rows"][str(lottery)]
        if len(frame) != expected_rows or list(frame.columns) != list(
            self.sidecar["columns"]
        ):
            raise LotteryFileError(
                f"Lottery {lottery} of {self.data_path} does not match its sidecar."
            )
        return frame


def _csv_header_cells(columns: Iterable[str]) -> list[str]:
    # The header as DataFrame.to_csv writes it (quoting where csv needs it).
    buffer = io.StringIO()
    pd.DataFrame(columns=list(columns)).to_csv(buffer, index=False)
    return [buffer.getvalue().rstrip("\r\n")]


def _derivable_columns(
    columns: list[str], frames_text: Mapping[int, str]
) -> tuple[dict, dict, list[str]]:
    """Pick the columns every lottery lets us drop, and their sidecar values."""
    parsed = {
        lottery: pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        for lottery, text in frames_text.items()
    }
    derived: dict[str, dict] = {}
    programcodes: dict[str, str] = {}
    for column in DERIVED_COLUMNS:
        if column not in columns:
            continue
        cells = [frame[column] for frame in parsed.values()]
        if column == "assignment_schema_version":
            values = {value for series in cells for value in series.unique()}
            if len(values) == 1:
                (value,) = values
                if re.fullmatch(r"-?\d+", value):
                    derived[column] = {"kind": "int", "value": int(value)}
        elif column == "rank_basis":
            values = {value for series in cells for value in series.unique()}
            if len(values) == 1:
                (value,) = values
                # A plain string re-reads as itself; anything read_csv would
                # coerce (numbers, NA tokens, quoting) stays in the data.
                if re.fullmatch(r"[A-Za-z_]+", value) and value not in {
                    "nan",
                    "NaN",
                    "NA",
                    "NULL",
                    "null",
                    "None",
                    "True",
                    "False",
                    "TRUE",
                    "FALSE",
                    "true",
                    "false",
                }:
                    derived[column] = {"kind": "str", "value": value}
        elif column == "utility_rank":
            if all(series.eq("").all() for series in cells):
                derived[column] = {"kind": "empty"}
        elif column == "programcodes" and "programno" in columns:
            # programno is written as the frame holds it ("85" or "85.0"),
            # so the map is keyed by that text; it must be a function, and
            # only a zero programno may carry a blank code.
            mapping: dict[str, str] = {}
            consistent = True
            for frame in parsed.values():
                pairs = set(zip(frame["programno"], frame["programcodes"]))
                for number, code in pairs:
                    try:
                        value = float(number)
                    except ValueError:
                        consistent = False
                        break
                    if (code == "") != (value == 0) or (
                        code and not re.fullmatch(r"[A-Za-z0-9_-]+", code)
                    ):
                        consistent = False
                        break
                    if mapping.setdefault(number, code) != code:
                        consistent = False
                        break
                if not consistent:
                    break
            if consistent:
                derived[column] = {"kind": "programcodes"}
                programcodes = dict(
                    sorted(mapping.items(), key=lambda kv: (float(kv[0]), kv[0]))
                )
    stored = [column for column in columns if column not in derived]
    return derived, programcodes, stored


def _lottery_lines(frame_text: str, stored: list[str], columns: list[str]) -> list[str]:
    """One lottery's data lines restricted to ``stored`` columns."""
    if stored == columns:
        body = frame_text.split("\n", 1)[1] if "\n" in frame_text else ""
    else:
        frame = pd.read_csv(io.StringIO(frame_text), dtype=str, keep_default_na=False)
        body_text = frame[stored].to_csv(index=False)
        body = body_text.split("\n", 1)[1]
    lines = body.splitlines(keepends=True)
    return [line if line.endswith("\n") else line + "\n" for line in lines]


def write_lotteries(
    data_path: str | Path,
    rendered: Mapping[int, tuple[list[str], str]],
    *,
    replace_lotteries: Iterable[int] = (),
) -> None:
    """Upsert lotteries into a variant's file pair, atomically and under a lock.

    ``rendered`` maps each lottery to :func:`render_lottery` output. Existing
    lotteries not in ``rendered`` (nor in ``replace_lotteries``, which are
    dropped) are kept, so concurrent single-lottery jobs of one variant
    compose. Every lottery must share one column list.
    """
    data_path = Path(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    dropped = set(int(value) for value in replace_lotteries)
    with _lock_for(data_path):
        existing = _read_unlocked(data_path, missing_ok=True, invalid_ok=True)
        texts: dict[int, str] = {}
        columns = None
        if existing is not None:
            base_columns = list(existing.sidecar["columns"])
            for lottery in existing.lotteries:
                if lottery in rendered or lottery in dropped:
                    continue
                texts[lottery] = _full_text(existing, lottery)
                columns = base_columns
        for lottery, (frame_columns, text) in rendered.items():
            if columns is None:
                columns = list(frame_columns)
            elif list(frame_columns) != columns:
                raise LotteryFileError(
                    f"Lottery {lottery} columns {frame_columns} differ from the "
                    f"variant's {columns} in {data_path}."
                )
            texts[int(lottery)] = text
        if not texts:
            _unlink_pair(data_path)
            return
        _write_unlocked(data_path, columns, texts)


def remove_lotteries(data_path: str | Path, lotteries: Iterable[int]) -> None:
    """Drop lotteries from a variant's file pair; remove it once empty."""
    data_path = Path(data_path)
    lotteries = {int(value) for value in lotteries}
    sidecar = sidecar_path_for(data_path)
    if not data_path.exists() and not sidecar.exists():
        return
    with _lock_for(data_path):
        existing = _read_unlocked(data_path, missing_ok=True, invalid_ok=True)
        if existing is None:
            _unlink_pair(data_path)
            return
        kept = [lottery for lottery in existing.lotteries if lottery not in lotteries]
        if len(kept) == len(existing.lotteries):
            return
        if not kept:
            _unlink_pair(data_path)
            return
        texts = {lottery: _full_text(existing, lottery) for lottery in kept}
        _write_unlocked(data_path, list(existing.sidecar["columns"]), texts)


def _full_text(existing: LotteryFile, lottery: int) -> str:
    """A stored lottery's per-lottery CSV text, rebuilt cell for cell.

    Works on the cells as text, never through parsed values, so re-encoding a
    file (adding or removing a lottery) cannot change a digit.
    """
    sidecar = existing.sidecar
    stored = pd.read_csv(
        io.StringIO(existing.lottery_text(lottery)),
        dtype=str,
        keep_default_na=False,
    )
    codes = sidecar.get("programcodes", {})
    columns = {}
    for column in sidecar["columns"]:
        if column in stored.columns:
            columns[column] = stored[column]
            continue
        spec = sidecar["derived_columns"][column]
        if spec["kind"] == "programcodes":
            missing = set(stored["programno"]) - set(codes)
            if missing:
                raise LotteryFileError(
                    f"{existing.data_path} has no programcodes for programno "
                    f"{sorted(missing)}."
                )
            columns[column] = stored["programno"].map(codes)
        elif spec["kind"] == "empty":
            columns[column] = pd.Series([""] * len(stored), dtype=object)
        else:
            columns[column] = pd.Series(
                [str(spec["value"])] * len(stored), dtype=object
            )
    return pd.DataFrame(columns).to_csv(index=False)


def _unlink_pair(data_path: Path) -> None:
    sidecar_path_for(data_path).unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)


def _write_unlocked(data_path: Path, columns: list[str], texts: Mapping[int, str]):
    lotteries = sorted(texts)
    derived, programcodes, stored = _derivable_columns(
        columns, {lottery: texts[lottery] for lottery in lotteries}
    )
    header_cells = _csv_header_cells([LOTTERY_COLUMN, *stored])
    pieces = [header_cells[0], "\n"]
    rows = {}
    for lottery in lotteries:
        lines = _lottery_lines(texts[lottery], stored, columns)
        rows[str(lottery)] = len(lines)
        prefix = f"{lottery},"
        pieces.extend(prefix + line for line in lines)
    raw = "".join(pieces).encode("utf-8")
    payload = gzip.compress(raw, compresslevel=_GZIP_LEVEL, mtime=0)
    sidecar = {
        "format": LOTTERY_FILE_FORMAT,
        "schema_version": LOTTERY_FILE_SCHEMA_VERSION,
        "data_file": data_path.name,
        "data_sha256": _sha256_bytes(payload),
        "columns": list(columns),
        "stored_columns": stored,
        "derived_columns": {
            column: spec for column, spec in derived.items() if column != "programcodes"
        },
        "programcodes": programcodes if "programcodes" in derived else {},
        "lotteries": lotteries,
        "rows": rows,
    }
    if "programcodes" in derived:
        sidecar["derived_columns"]["programcodes"] = {"kind": "programcodes"}
    sidecar_path = sidecar_path_for(data_path)
    _atomic_write_bytes(data_path, payload)
    _atomic_write_bytes(
        sidecar_path,
        (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_unlocked(
    data_path: Path, *, missing_ok: bool = False, invalid_ok: bool = False
) -> LotteryFile | None:
    sidecar_path = sidecar_path_for(data_path)
    try:
        sidecar_bytes = sidecar_path.read_bytes()
        payload = data_path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    try:
        return _parse(data_path, sidecar_bytes, payload)
    except LotteryFileError:
        if invalid_ok:
            return None
        raise


def _parse(data_path: Path, sidecar_bytes: bytes, payload: bytes) -> LotteryFile:
    try:
        sidecar = json.loads(sidecar_bytes)
    except json.JSONDecodeError as exc:
        raise LotteryFileError(f"Invalid lottery sidecar for {data_path}.") from exc
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("format") != LOTTERY_FILE_FORMAT
        or sidecar.get("schema_version") != LOTTERY_FILE_SCHEMA_VERSION
    ):
        raise LotteryFileError(
            f"Lottery sidecar for {data_path} has an unsupported format or schema "
            f"version (expected {LOTTERY_FILE_SCHEMA_VERSION})."
        )
    if sidecar.get("data_sha256") != _sha256_bytes(payload):
        raise LotteryFileError(f"Lottery file {data_path} failed its checksum.")
    text = gzip.decompress(payload).decode("utf-8")
    header, _, body = text.partition("\n")
    stored = list(sidecar["stored_columns"])
    if header != _csv_header_cells([LOTTERY_COLUMN, *stored])[0]:
        raise LotteryFileError(f"Lottery file {data_path} has an unexpected header.")
    lines: dict[int, list[str]] = {lottery: [] for lottery in sidecar["lotteries"]}
    order = []
    for line in body.splitlines(keepends=True):
        prefix, comma, rest = line.partition(",")
        if not comma:
            raise LotteryFileError(f"Lottery file {data_path} has a malformed row.")
        lottery = int(prefix)
        bucket = lines.get(lottery)
        if bucket is None:
            raise LotteryFileError(
                f"Lottery file {data_path} has rows for unlisted lottery {lottery}."
            )
        if not order or order[-1] != lottery:
            if lottery in order:
                raise LotteryFileError(
                    f"Lottery file {data_path} interleaves lottery {lottery}."
                )
            order.append(lottery)
        bucket.append(rest)
    for lottery, bucket in lines.items():
        if len(bucket) != sidecar["rows"][str(lottery)]:
            raise LotteryFileError(
                f"Lottery file {data_path} has {len(bucket)} rows for lottery "
                f"{lottery}, expected {sidecar['rows'][str(lottery)]}."
            )
    return LotteryFile(data_path, sidecar, lines)


def read_lotteries(data_path: str | Path) -> LotteryFile:
    """Read and verify a variant's file pair.

    A checksum mismatch can only be a writer between its two ``os.replace``
    calls, so the read is retried once under the writers' lock.
    """
    data_path = Path(data_path)
    try:
        return _read_unlocked(data_path)
    except LotteryFileError:
        with _lock_for(data_path):
            return _read_unlocked(data_path)


def saved_lotteries(data_path: str | Path) -> set[int]:
    """The lotteries a variant's sidecar lists; empty when there is none.

    Cheap: reads only the sidecar, not the data file.
    """
    sidecar_path = sidecar_path_for(Path(data_path))
    try:
        sidecar = json.loads(sidecar_path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("format") != LOTTERY_FILE_FORMAT
        or sidecar.get("schema_version") != LOTTERY_FILE_SCHEMA_VERSION
        or not Path(data_path).is_file()
    ):
        return set()
    return {int(value) for value in sidecar.get("lotteries", [])}


class LotteryFileCache:
    """Parsed lottery files, re-read whenever the sidecar changes on disk."""

    def __init__(self):
        self._files: dict[Path, tuple[tuple, LotteryFile]] = {}

    @staticmethod
    def _signature(data_path: Path):
        sidecar = sidecar_path_for(data_path)
        try:
            a, b = sidecar.stat(), data_path.stat()
        except FileNotFoundError:
            return None
        return (a.st_mtime_ns, a.st_size, a.st_ino, b.st_mtime_ns, b.st_size, b.st_ino)

    def get(self, data_path: str | Path) -> LotteryFile | None:
        data_path = Path(data_path)
        signature = self._signature(data_path)
        if signature is None:
            self._files.pop(data_path, None)
            return None
        cached = self._files.get(data_path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        parsed = read_lotteries(data_path)
        self._files[data_path] = (self._signature(data_path), parsed)
        return parsed

    def invalidate(self, data_path: str | Path | None = None) -> None:
        if data_path is None:
            self._files.clear()
        else:
            self._files.pop(Path(data_path), None)


def read_saved_lottery(legacy_iteration_path: str | Path) -> pd.DataFrame:
    """Read one lottery by its per-lottery path, from whichever layout holds it.

    The long-format file wins; an old run's ``*_iteration<N>.csv`` is the
    fallback. Returns the frame ``pd.read_csv`` gave for the per-lottery file.
    """
    legacy = Path(legacy_iteration_path)
    match = LEGACY_ITERATION_PATTERN.match(legacy.name)
    if match is not None:
        data_path, _ = lottery_file_paths(legacy)
        iteration = int(match.group("iteration"))
        if iteration in saved_lotteries(data_path):
            return read_lotteries(data_path).frame(iteration)
    return pd.read_csv(legacy)


def saved_lottery_exists(legacy_iteration_path: str | Path) -> bool:
    legacy = Path(legacy_iteration_path)
    match = LEGACY_ITERATION_PATTERN.match(legacy.name)
    if match is not None:
        data_path, _ = lottery_file_paths(legacy)
        if int(match.group("iteration")) in saved_lotteries(data_path):
            return True
    return legacy.is_file()
