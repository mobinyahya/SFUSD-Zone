"""Deterministic short code for a zoning solution.

Hashes a `{area_id: zone_id}` mapping into a fixed-length lowercase base36
string so that each particular zoning has a compact, citable identifier.
"""
import hashlib
import json
import os
import string

_BASE36_ALPHABET = string.digits + string.ascii_lowercase


def _int_to_base36(n: int) -> str:
    if n == 0:
        return "0"
    chars = []
    while n:
        n, rem = divmod(n, 36)
        chars.append(_BASE36_ALPHABET[rem])
    return "".join(reversed(chars))


def compute_solution_code(zone_dict: dict, length: int = 7) -> str:
    """Return a deterministic lowercase base36 code for `zone_dict`.

    Same `{area_id: zone_id}` mapping always yields the same code; any
    change to which area lands in which zone changes the code.
    """
    canonical = sorted(
        ((int(area_id), int(zone_id)) for area_id, zone_id in zone_dict.items()),
        key=lambda x: x[0],
    )
    payload = json.dumps(canonical, separators=(",", ":")).encode()
    digest_int = int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")
    encoded = _int_to_base36(digest_int).rjust(length, "0")
    return encoded[-length:]


def solution_code_from_folder(folder_path: str) -> str:
    """Read the final-level zone_dict from a benchmark output folder and code it.

    Picks the zone_dict matching the last entry in `result.json`'s `levels`
    list; falls back to `zone_dict_BlockGroup_0.json`.
    """
    folder = os.path.expanduser(folder_path)
    result_path = os.path.join(folder, "result.json")
    level = "BlockGroup_0"
    if os.path.exists(result_path):
        with open(result_path) as f:
            levels = json.load(f).get("levels") or []
        if levels:
            level = levels[-1]
    zone_dict_path = os.path.join(folder, f"zone_dict_{level}.json")
    with open(zone_dict_path) as f:
        zone_dict = json.load(f)
    return compute_solution_code(zone_dict)


# ============================================================================
# Code -> solution mapping
# ============================================================================

INDEX_FILENAME = "solution_codes.json"


def build_solution_code_index(root_folder: str, write: bool = True) -> dict[str, list[str]]:
    """Walk `root_folder` and build `{code: [relative_solution_folder, ...]}`.

    Reads `metrics.solution_code` from each `result.json`; falls back to
    computing it from the zone_dict if missing. Writes the mapping to
    `<root_folder>/solution_codes.json` when `write=True`.
    """
    root = os.path.expanduser(root_folder)
    index: dict[str, list[str]] = {}

    for dirpath, _, files in os.walk(root):
        if "result.json" not in files:
            continue
        try:
            with open(os.path.join(dirpath, "result.json")) as f:
                data = json.load(f)
            code = (data.get("metrics") or {}).get("solution_code")
            if not code:
                try:
                    code = solution_code_from_folder(dirpath)
                except FileNotFoundError:
                    continue
        except (OSError, json.JSONDecodeError):
            continue
        rel = os.path.relpath(dirpath, root)
        index.setdefault(code, []).append(rel)

    if write:
        with open(os.path.join(root, INDEX_FILENAME), "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)

    return index


def _load_index(root_folder: str) -> dict[str, list[str]]:
    root = os.path.expanduser(root_folder)
    index_path = os.path.join(root, INDEX_FILENAME)
    if os.path.exists(index_path):
        with open(index_path) as f:
            raw = json.load(f)
        return {k: ([v] if isinstance(v, str) else list(v)) for k, v in raw.items()}
    return build_solution_code_index(root, write=False)


def resolve_solution_code(
    code: str,
    root_folder: str,
    *,
    all_matches: bool = False,
) -> str | list[str]:
    """Return the absolute folder path(s) for `code` under `root_folder`.

    Loads `<root>/solution_codes.json` if present, otherwise walks the tree.
    Returns the first match by default; pass `all_matches=True` to get every
    folder whose solution hashes to this code.
    """
    root = os.path.expanduser(root_folder)
    matches = _load_index(root).get(code, [])
    if not matches:
        raise KeyError(f"solution code {code!r} not found under {root}")
    abs_matches = [os.path.join(root, rel) for rel in matches]
    return abs_matches if all_matches else abs_matches[0]


def load_solution_by_code(code: str, root_folder: str):
    """Convenience: resolve `code` to a folder and return its `BenchmarkResult`."""
    from Zone_Generation.Running_Analysis.benchmark.results import BenchmarkResult

    folder = resolve_solution_code(code, root_folder)
    return BenchmarkResult.load(folder)
