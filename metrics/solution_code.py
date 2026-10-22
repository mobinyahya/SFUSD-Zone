"""Deterministic short code for a zoning solution.

Hashes a `{area_id: zone_id}` mapping into a fixed-length lowercase base36
string so that each particular zoning has a compact, citable identifier.
"""

import hashlib
import json
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
    digest_int = int.from_bytes(
        hashlib.blake2b(payload, digest_size=16).digest(), "big"
    )
    encoded = _int_to_base36(digest_int).rjust(length, "0")
    return encoded[-length:]
