"""The zoning-independent part of one matching policy's market, built once.

A matching policy's market is the students, their submitted lists, the
program types they qualify for, their priorities and reserve classes, and the
order in which they would be designated. All of it is fixed by the data
scenario and the policy; only the zoning decides which programs each student
may enter and where they hold zone priority. :class:`PolicyMarket` holds the
fixed part, and its ``apply`` methods take a zoning -- the simulator's
``Zones.zone_priority_matrix`` -- and return exactly what
``PreferenceGenerator.initialize_real_preferences(designate=True)`` and
``PriorityGenerator`` produce for it.

Both consumers share it: the assignment simulator (``PreferenceGenerator``,
``PriorityGenerator``) applies a zoning to it, and the MID lottery market
(``optimization.data.mid``) keeps every candidate tagged by kind and lets the
lottery DA apply the zoning (``optimization.lottery_da``).

:func:`load_policy_market` caches it on disk, content-addressed by the
assignment sources and filters, the policy settings it reads
(:data:`POLICY_MARKET_SETTINGS`) and :data:`POLICY_MARKET_SCHEMA_VERSION`, and
memoises it in-process. Concurrent processes build it once: one builds under
an exclusive lock while the rest wait, then load it.

What it reproduces (the simulator falls back to its own code for anything
else, see :func:`unmodelled_policy_settings`): submitted lists from each
student's selected round, ``add_aa_schools``, ``designate`` with the
``in_zone`` designation ordering, the policy priorities of any grade, zone
eligibility exceptions (``restrict-zone``, ``sibling-access``, ``tk-access``,
``CTIP_access``) and the round and non-designation boosts.

With ``utility-model.enable: true`` the lists are drawn instead, one set per
lottery: :meth:`PolicyMarket.utility_draws` replays the simulator's random
stream for one iteration (the Gumbel utility draws, the outside-option draw,
then the MTB lottery), and :meth:`PolicyMarket.utility_preferences` truncates
and completes the lists as
``PreferenceGenerator.get_utility_model_preferences_after_truncation`` does.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

#: Bump whenever :class:`PolicyMarket`'s fields or how they are built change.
POLICY_MARKET_SCHEMA_VERSION = 1
POLICY_MARKET_ARTIFACT = "policy_market"
POLICY_MARKET_PAYLOAD = "policy_market.pkl"

#: The policy settings the market reads, with the default the simulator's code
#: assumes when one is absent. They, and nothing else of the policy config,
#: key the cache, so a setting the builder reads must be listed here.
POLICY_MARKET_SETTINGS: dict[str, Any] = {
    "ctip-options": None,
    "priority-weights": None,
    "restrict-zone": False,
    "sibling-access": False,
    "tk-access": False,
    "non_designation_boost": 100,
    "aa_boost": 0,
    "distance-priority": {"step-size": 10},
    "distance-boost": None,
    "guard-rails": -1,
    "reserve-settings": {},
    "citywide-reserve-ratios": [0.57, 0.43],
    "citywide-separate-reserves": True,
}

#: ``utility-model.list-length`` that ranks exactly the programs a student's
#: drawn utility puts above their drawn outside option. The estimate must be
#: utility net of the outside option (e.g. ``ChoiceModel_2526_outside``).
OUTSIDE_OPTION_LIST_LENGTH = "outside_option"

#: The ``utility-model`` keys the shared path models. ``save-path`` only writes
#: the first iteration's utilities to a file.
UTILITY_MODEL_KEYS = frozenset(
    {"enable", "list-length", "gumbel-scale", "designate-lp-for-all", "save-path"}
)

#: Geography columns kept for the MID market (``choice.estimate.student_area_column``).
STUDENT_AREA_COLUMNS = (
    "census_block",
    "census_blockgroup",
    "census_tract",
    "attendance_area",
)

_MEMO: OrderedDict[str, PolicyMarket] = OrderedDict()
_MEMO_SIZE = 4

# -- the whitelist ---------------------------------------------------------------

#: The two consumers of a policy market. ``simulator``: the assignment
#: simulator takes its lists and priorities from the market and runs its own
#: lottery, reserves and DA. ``mid``: the MID lottery market, matched by
#: ``optimization.lottery_da``, which models the lottery, reserves and DA too.
CONSUMERS = ("simulator", "mid")

_ABSENT = object()
_PRIORITY_CATEGORIES = {"ctip", "prek", "sibling", "zone", "promote", "language-programs"}
_LANGUAGE_PROGRAM_CATEGORIES = {"sibling", "ctip", "lp-sibling", "lp"}
_CTIP_OPTIONS = (0, 1, "new_ctip", "new_ctip_blockgroup")
_RESERVE_OPTION_KEYS = {"guard-rails": -1, "reserve-settings": {}}
_RESTRICT_OPTION_KEYS = {"restrict-zone": False, "citywide-or-lp": []}


def _falsy(value, config) -> bool:
    return value is _ABSENT or not value


def _empty(value, config) -> bool:
    return value is _ABSENT or value is None or (hasattr(value, "__len__") and not len(value))


def _boolean(value, config) -> bool:
    return value is _ABSENT or isinstance(value, bool)


def _number(value, config) -> bool:
    return value is _ABSENT or (
        isinstance(value, int | float) and not isinstance(value, bool)
    )


def _single_option(keys: dict):
    """Absent, or one option equal to the top-level settings it overrides
    (the simulator copies it over them before every run)."""

    def check(value, config) -> bool:
        if _empty(value, config):
            return True
        if not isinstance(value, list | tuple) or len(value) != 1:
            return False
        option = value[0]
        return isinstance(option, Mapping) and all(
            option.get(key, default) == config.get(key, default)
            for key, default in keys.items()
        )

    return check


def _priority_weights(distance: bool):
    allowed = _PRIORITY_CATEGORIES | ({"distance"} if distance else set())

    def check(value, config) -> bool:
        if not isinstance(value, Mapping) or not set(value) <= allowed:
            return False
        language = value.get("language-programs", {})
        return isinstance(language, Mapping) and set(language) <= (
            _LANGUAGE_PROGRAM_CATEGORIES
        )

    return check


def _ctip_options(single: bool):
    def check(value, config) -> bool:
        return (
            isinstance(value, list | tuple)
            and len(value) >= 1
            and (len(value) == 1 or not single)
            and all(option in _CTIP_OPTIONS and not isinstance(option, bool) for option in value)
        )

    return check


def _guard_rails(value, config) -> bool:
    return value is _ABSENT or (value in (-1, 0, 1) and not isinstance(value, bool))


def _reserve_settings(value, config) -> bool:
    if int(config.get("guard-rails", -1)) == -1:
        return True
    if not isinstance(value, Mapping):
        return False
    if not {"column", "lower_disadvantaged"} <= set(value):
        return False
    if "num_categories" not in value and "thresholds" not in value:
        return False
    # Zone-share reserves size each zone's pools by its class shares, which
    # change with the zoning: only fixed fractions are modelled.
    fraction = value.get("reserve_fraction", -1)
    return isinstance(fraction, list | tuple) and len(fraction) > 0


def utility_model_enabled(config: Mapping) -> bool:
    """Whether ``config`` draws its lists from the utility model.

    ``enable`` unset (``None``, as ``base_config`` leaves it) or false means
    the submitted lists.
    """
    value = config.get("utility-model")
    return isinstance(value, Mapping) and value.get("enable") is True


def utility_list_length(value) -> int | None:
    """``list-length`` as a count, or ``None`` for the outside option.

    Raises for any other value: only a whole number of programs (a digit
    string, as the simulator's schema types it) and the outside option are
    modelled.
    """
    if value == OUTSIDE_OPTION_LIST_LENGTH:
        return None
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise ValueError(
        f"utility-model list-length {value!r} is not modelled; use a whole "
        f"number such as '7' or {OUTSIDE_OPTION_LIST_LENGTH!r}."
    )


def _utility_model(value, config) -> bool:
    if value is _ABSENT or value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    enable = value.get("enable")
    if enable is None or enable is False:
        return True
    if enable is not True or not set(value) <= UTILITY_MODEL_KEYS:
        return False
    utility_list_length(value.get("list-length"))
    scale = value.get("gumbel-scale", 1.0)
    if isinstance(scale, bool) or not isinstance(scale, int | float):
        return False
    if not np.isfinite(scale) or scale < 0:
        return False
    if not isinstance(value.get("designate-lp-for-all", False), bool):
        return False
    # Without designation the simulator leaves its appended programs
    # unfiltered; only the designating path is modelled.
    return config.get("designate") is True


def _one_of(*values):
    def check(value, config) -> bool:
        return value is _ABSENT or any(
            value == allowed and type(value) is type(allowed) for allowed in values
        )

    return check


def _always(value, config) -> bool:
    return True


#: Every setting that changes the market or the match, with the values the
#: shared path models: ``name -> (supported values, check)``. A config may set
#: anything else (outputs, iterations, seeds, paths): nothing else reaches the
#: market or the match. See :func:`unmodelled_policy_settings`.
_SHARED_WHITELIST: dict[str, tuple[str, Callable]] = {
    "assignment-algorithm": ("absent or 'DA'", _one_of("DA", None)),
    "utility-model": (
        "absent, enable unset or false (the submitted lists), or enable true "
        f"with list-length a whole number or {OUTSIDE_OPTION_LIST_LENGTH!r}, "
        "gumbel-scale a finite non-negative number, designate-lp-for-all true "
        f"or false, no keys beyond {sorted(UTILITY_MODEL_KEYS)}, and designate: "
        "true",
        _utility_model,
    ),
    "ties-options": (
        "['MTB']",
        lambda value, config: isinstance(value, list | tuple) and list(value) == ["MTB"],
    ),
    "read-lotteries": ("absent or false (fresh draws)", _falsy),
    "designate": ("true or false", _boolean),
    "designation-ordering-type": ("absent or 'in_zone'", _one_of("in_zone")),
    "add_aa_schools": ("true or false", _boolean),
    "drop_below_aa": ("absent or false", _falsy),
    "remove_non_aa_or_citywide": ("absent or false", _falsy),
    "aa_ge_only": ("absent or false", _falsy),
    "truncate-at-AA-GE": ("absent or false", _falsy),
    "restrict-zone": ("true, false or 'CTIP_access'", _one_of(True, False, "CTIP_access")),
    "restrict-zone-options": (
        "absent, or one option equal to restrict-zone/citywide-or-lp",
        _single_option(_RESTRICT_OPTION_KEYS),
    ),
    "citywide-or-lp": ("absent or empty (no supplemental zones)", _empty),
    "sibling-access": ("true or false", _boolean),
    "tk-access": ("true or false", _boolean),
    "non_designation_boost": ("a number", _number),
    "guard-rails": ("-1, 0 or 1", _guard_rails),
    "guard-rails-reserve-options": (
        "absent, or one option equal to guard-rails/reserve-settings",
        _single_option(_RESERVE_OPTION_KEYS),
    ),
    "reserve-settings": (
        "with reserves: column, num_categories or thresholds, lower_disadvantaged "
        "and a reserve_fraction (not zone-share reserves)",
        _reserve_settings,
    ),
    "citywide-reserve-ratios": ("any", _always),
    "citywide-separate-reserves": ("true or false", _boolean),
    # Names the assignment file only (MarketGenerator._get_assignment_save_name).
    "soft_reserve_boost": ("any (labels the output only)", _always),
}

#: What each consumer adds. The simulator reproduces these exactly (they are
#: zoning-independent priority terms, or act after its own DA); the lottery
#: DA does not model them, and it holds one integer tier per alternative.
_CONSUMER_WHITELIST: dict[str, dict[str, tuple[str, Callable]]] = {
    "simulator": {
        "priority-weights": (
            f"categories {sorted(_PRIORITY_CATEGORIES | {'distance'})}; "
            f"language-programs {sorted(_LANGUAGE_PROGRAM_CATEGORIES)}",
            _priority_weights(distance=True),
        ),
        "ctip-options": (f"entries of {list(_CTIP_OPTIONS)}", _ctip_options(single=False)),
        "zone-building-blocks": (
            "absent, 'attendance_area', 'block', 'block_group' or 'tract'",
            _one_of("attendance_area", "block", "block_group", "tract"),
        ),
        "overscribe_aa": ("any (after the DA, on the same lists)", _always),
        "aa_boost": ("a number", _number),
        "distance-priority": ("any", _always),
        "distance-boost": ("any", _always),
    },
    "mid": {
        "priority-weights": (
            f"categories {sorted(_PRIORITY_CATEGORIES)}; "
            f"language-programs {sorted(_LANGUAGE_PROGRAM_CATEGORIES)}",
            _priority_weights(distance=False),
        ),
        "ctip-options": (f"one entry of {list(_CTIP_OPTIONS)}", _ctip_options(single=True)),
        "policies": (
            "one zone policy",
            lambda value, config: isinstance(value, list | tuple) and len(value) == 1,
        ),
        "overscribe_aa": ("absent or false", _falsy),
        "aa_boost": ("absent or 0", _falsy),
        "distance-priority": ("absent", lambda value, config: value is _ABSENT),
        "distance-boost": ("absent or empty", _falsy),
    },
}


def policy_market_whitelist(consumer: str) -> dict[str, str]:
    """``setting -> supported values`` for one consumer (see :data:`CONSUMERS`)."""
    if consumer not in CONSUMERS:
        raise ValueError(f"Unknown policy market consumer {consumer!r}.")
    merged = {**_SHARED_WHITELIST, **_CONSUMER_WHITELIST[consumer]}
    return {name: supported for name, (supported, _) in merged.items()}


def unmodelled_policy_settings(config: Mapping, consumer: str = "simulator") -> list[str]:
    """Each setting of ``config`` the shared path does not model, as
    ``"name=value (supported: ...)"``; empty when the shared path applies.

    A whitelist: the policy takes the shared path only if every setting that
    changes the market or the match has a supported value. The simulator then
    keeps its own ``PreferenceGenerator``/``PriorityGenerator`` code; MID
    raises (``optimization.data.mid``).
    """
    if consumer not in CONSUMERS:
        raise ValueError(f"Unknown policy market consumer {consumer!r}.")
    merged = {**_SHARED_WHITELIST, **_CONSUMER_WHITELIST[consumer]}
    reasons = []
    for name, (supported, check) in merged.items():
        value = config.get(name, _ABSENT)
        try:
            ok = check(value, config)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            shown = "absent" if value is _ABSENT else repr(value)
            reasons.append(f"{name}={shown} (supported: {supported})")
    return reasons


def policy_market_settings(config: Mapping) -> dict[str, Any]:
    """The policy settings a :class:`PolicyMarket` depends on, defaults filled."""
    settings = {}
    for name, default in POLICY_MARKET_SETTINGS.items():
        value = config.get(name, default)
        if value is None and default is None and name in (
            "ctip-options",
            "priority-weights",
        ):
            raise ValueError(f"A policy market needs {name!r} in the policy config.")
        settings[name] = copy.deepcopy(value)
    # The utility model adds the base utility matrix and, with
    # designate-lp-for-all, changes the designation order. Only an enabled one
    # keys the market, so every submitted-list market keeps its cache key.
    if utility_model_enabled(config):
        settings["utility-model"] = {
            "designate-lp-for-all": bool(
                config["utility-model"].get("designate-lp-for-all", False)
            )
        }
    # Reserve settings matter only with reserves; dropping them otherwise lets
    # a policy with and without unused reserve settings share one market.
    if int(settings["guard-rails"]) == -1:
        settings["reserve-settings"] = {}
        settings["citywide-reserve-ratios"] = POLICY_MARKET_SETTINGS[
            "citywide-reserve-ratios"
        ]
        settings["citywide-separate-reserves"] = True
    return json.loads(json.dumps(settings, sort_keys=True))


def ctip_key(ctip) -> str:
    """The key of one ``ctip-options`` entry in :attr:`PolicyMarket.tiers`."""
    return json.dumps(ctip, sort_keys=True)


def iteration_seed(base_seed: int, iteration: int) -> int:
    """``MarketGenerator.iteration_seed``: the NumPy seed of one iteration.

    Repeated here so the optimizer need not import the simulator's generator;
    ``test_policy_market`` checks the two agree.
    """
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Assignment iteration must be a non-negative integer.")
    payload = f"sfusd-assignment-v1:{int(base_seed)}:{iteration}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True, eq=False)
class UtilityDraw:
    """One iteration's random draws, in the simulator's order.

    ``utilities`` is ``mus`` plus the Gumbel draws
    (``UtilityModel.original_utilities``), ``outside`` the outside-option
    draws (``None`` without the outside option), and ``lottery`` the MTB
    lottery ``PriorityGenerator._mtb`` draws next from the same stream.
    """

    utilities: np.ndarray
    outside: np.ndarray | None
    lottery: np.ndarray


# -- shared priority arithmetic ------------------------------------------------


def not_designation_mask(
    preferences: np.ndarray, pref_length, num_programs: int
) -> np.ndarray:
    """1 at each student's listed (not designation) programs: the first
    ``pref_length`` entries of their list."""
    students = preferences.shape[0]
    not_designation = np.zeros([students, num_programs])
    pref_lengths = np.asarray(pref_length, dtype=int)
    pref_lengths = np.clip(pref_lengths, 0, preferences.shape[1])
    positions = np.arange(preferences.shape[1])
    listed = positions < pref_lengths[:, np.newaxis]
    program_idxs = preferences[listed].astype(int) - 1
    student_idxs = np.repeat(np.arange(students), pref_lengths)
    valid = np.logical_and(program_idxs >= 0, program_idxs < num_programs)
    not_designation[student_idxs[valid], program_idxs[valid]] = 1
    return not_designation


def priorities_without_lottery(
    round_priorities: np.ndarray,
    policy_priorities: np.ndarray,
    not_designation: np.ndarray,
    non_designation_boost: float,
    restrict_zone,
    zone_eligibility: np.ndarray | Callable[[], np.ndarray],
) -> np.ndarray:
    """``PriorityGenerator.get_priorities_without_lottery``'s arithmetic.

    The round priority, plus at listed programs the policy priority and the
    non-designation boost; under ``restrict-zone`` every program outside the
    student's zone eligibility is pushed 500 below everything else.
    """
    final = (
        round_priorities
        + np.multiply(policy_priorities, not_designation)
        + not_designation * non_designation_boost
    )
    if restrict_zone:
        zone_mask = zone_eligibility() if callable(zone_eligibility) else zone_eligibility
        return np.multiply(final, zone_mask) - (1 - zone_mask) * 500
    return final


def round_priority_boost(first_round: np.ndarray) -> np.ndarray:
    """Each earlier selected round outranks every later one by 1000."""
    return 1000 * (first_round.max() - first_round)


def program_type_eligibility(programs, students) -> np.ndarray:
    """(students, programs) bool: the student qualifies for the program's type
    (language pools, special programs; ``Students.get_qualified_programs_dict``)."""
    qualified = students.get_qualified_programs_dict()
    by_type = programs.program_type_to_indices
    known = set(programs.program_df["program_type"].unique())
    eligible = np.zeros((students.n, programs.num_programs), dtype=bool)
    for row, studentno in students.idx2studentno.items():
        for program_type in qualified[studentno]:
            if program_type in known:
                eligible[row, [index - 1 for index in by_type[program_type]]] = True
    return eligible


def distance_orderings(distances: pd.DataFrame, index=None):
    """Each student's language and GE programs, nearest first.

    Returns ``(lp_columns, lp_order, ge_columns, ge_order)``: the distance
    columns of non-GE and of GE programs, and per row of ``distances`` (or of
    ``index``) the column positions in increasing distance. This is numpy's
    default argsort row by row, as the simulator has always ordered them, done
    once over the whole matrix; programs at one school tie and keep the order
    that sort gives them.
    """
    if index is not None:
        distances = distances.loc[index]
    lp_columns = [c for c in distances.columns if c.split("-")[1] != "GE"]
    ge_columns = [c for c in distances.columns if c.split("-")[1] == "GE"]
    lp_order = np.argsort(distances[lp_columns].to_numpy(), axis=1)
    ge_order = np.argsort(distances[ge_columns].to_numpy(), axis=1)
    return lp_columns, lp_order, ge_columns, ge_order


# -- the market ------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AppliedPolicyMarket:
    """A :class:`PolicyMarket` under one zoning (see :meth:`PolicyMarket.apply`)."""

    preferences: np.ndarray
    pref_length: np.ndarray
    zone_eligibility: np.ndarray
    policy_priorities: np.ndarray
    priorities: np.ndarray


@dataclass(frozen=True, eq=False)
class PolicyMarket:
    """The zoning-independent part of a matching policy's market.

    Rows are students in the simulator's order (``Students.student_data``),
    columns programs in index order: column ``j`` is program index ``j + 1``.
    """

    settings: dict
    grade: str
    #: ``program_ids[j]`` has program index ``j + 1``.
    program_ids: tuple[str, ...]
    #: The assignment program table (``Programs.program_df``).
    program_table: pd.DataFrame
    citywide_schools: tuple[int, ...]
    #: Student numbers in row order (``Students.student_data.index``).
    student_index: pd.Index
    #: Per student: :data:`STUDENT_AREA_COLUMNS` present and ``frl``.
    student_attributes: pd.DataFrame
    #: (n, p) submitted program indices from the selected round, 0-padded.
    selected: np.ndarray
    #: (n,) program index of the GE program of the student's district
    #: attendance area at this grade; 0 when there is none.
    attendance_area_ge: np.ndarray
    #: (n, p) bool: the student qualifies for the program's type.
    qualifies: np.ndarray
    #: (n, p) the zoning-independent addend of ``Zones.zone_eligibility_matrix``:
    #: sibling-access, tk-access, CTIP_access and unrestricted zones.
    access: np.ndarray
    #: ``ctip_key(ctip)`` -> (in-zone, out-of-zone) policy priorities: what
    #: ``PriorityGenerator._set_policy_priorities`` gives with every program in
    #: the student's zone, and with none.
    tiers: dict[str, tuple[np.ndarray, np.ndarray]]
    #: (n,) the round priority (``PriorityGenerator._round_priorities``).
    round_boost: np.ndarray
    non_designation_boost: float
    restrict_zone: Any
    #: CSR rows (``designation_start``) of each student's designation order:
    #: the language programs of a type they ranked, nearest first, if they
    #: asked for language designation; then every GE program, nearest first.
    designation_start: np.ndarray
    designation: np.ndarray
    #: Reserve class per student, class count, strictness, and per GE program
    #: its class seat fractions; ``reserve_fractions`` is ``None`` when the
    #: reserves are sized by zone shares, which depend on the zoning.
    reserve_classes: np.ndarray
    num_classes: int
    strict_reserves: bool
    reserve_fractions: dict[str, tuple[float, ...]] | None
    #: (n, p) the utility model's base utilities (``UtilityModel`` ``mus``),
    #: when the policy enables it; ``None`` otherwise. A default, so markets
    #: pickled before the field existed (all of them submitted-list ones) load
    #: with ``None``.
    utility: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.student_index)

    @property
    def num_programs(self) -> int:
        return len(self.program_ids)

    @property
    def program_index(self) -> dict[str, int]:
        return {program_id: j + 1 for j, program_id in enumerate(self.program_ids)}

    def designation_order(self, row: int) -> np.ndarray:
        return self.designation[self.designation_start[row] : self.designation_start[row + 1]]

    # -- applying a zoning -------------------------------------------------------

    def zone_eligibility(self, zone_priority: np.ndarray) -> np.ndarray:
        """``Zones.zone_eligibility_matrix`` for this zone priority matrix."""
        return np.clip(_zone_matrix(zone_priority, self) + self.access, 0, 1)

    def policy_priorities(self, zone_priority: np.ndarray, ctip) -> np.ndarray:
        """``PriorityGenerator._set_policy_priorities(ctip, ...)`` for this zoning.

        The simulator's priorities are elementwise in the 0/1 zone priority,
        so each cell is its in-zone or its out-of-zone tier, bit for bit.
        """
        inside, outside = self.tiers[ctip_key(ctip)]
        return np.where(_zone_matrix(zone_priority, self) > 0, inside, outside)

    def appended_programs(
        self,
        zone_priority: np.ndarray | None,
        *,
        generated_zones: bool,
        citywide_program_ids=(),
    ) -> list[np.ndarray]:
        """What ``add_aa_schools`` appends to each list.

        The district attendance area's GE program, or under generated zones
        the GE program of every school the student's zone owns, in program
        order (``PreferenceGenerator._zone_owned_ge_programs``).
        """
        if not generated_zones:
            return [
                np.array([program], dtype=int) if program else np.zeros(0, dtype=int)
                for program in self.attendance_area_ge
            ]
        citywide = set(citywide_program_ids)
        owned = np.array(
            [
                j
                for j, code in enumerate(self.program_ids)
                if code.split("-")[1:] == ["GE", self.grade] and code not in citywide
            ],
            dtype=int,
        )
        in_zone = _zone_matrix(zone_priority, self)[:, owned] > 0
        return [owned[row] + 1 for row in in_zone]

    def preferences(
        self,
        eligibility: np.ndarray,
        appended: list[np.ndarray] | None,
        selected: np.ndarray | None = None,
        dtype=float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``initialize_real_preferences(designate=True)``: ``(prefs, pref_length)``.

        Each list is the submitted programs (or ``selected``, 0-padded
        program indices), then ``appended`` (``None``: no ``add_aa_schools``),
        then the designation order; only programs the student qualifies for
        and is zone-eligible for are kept. ``pref_length`` counts the kept
        submitted and appended programs.
        """
        n, width = self.n, self.num_programs
        eligible = self.qualifies & (np.asarray(eligibility) > 0)
        selected = self.selected if selected is None else np.asarray(selected)
        prefs = np.zeros((n, width), dtype=dtype)
        pref_length = np.zeros(n, dtype=int)
        lengths = np.count_nonzero(selected, axis=1)
        for i in range(n):
            row = selected[i]
            extra = None if appended is None else appended[i]
            if extra is not None and len(extra):
                row = row.copy()
                length = int(lengths[i])
                for program in extra:
                    if program in row[:length] or length >= width:
                        continue
                    row[length] = program
                    length += 1
            ranked = list(dict.fromkeys(int(p) for p in row if p != 0))
            allowed = eligible[i]
            kept = [p for p in ranked if allowed[p - 1]]
            pref_length[i] = len(kept)
            seen = set(ranked)
            kept += [
                int(p)
                for p in self.designation_order(i)
                if allowed[p - 1] and int(p) not in seen
            ]
            prefs[i, : len(kept)] = kept
        return prefs, pref_length

    def round_priorities(self) -> np.ndarray:
        return np.outer(self.round_boost, np.ones(self.num_programs))

    # -- the utility model -----------------------------------------------------

    def utility_draws(
        self,
        iteration: int,
        random_seed: int,
        *,
        gumbel_scale: float = 1.0,
        outside_option: bool = False,
    ) -> UtilityDraw:
        """The draws the simulator makes in one iteration of a utility-model policy.

        ``_run_single_iteration_of_policy`` seeds NumPy with
        ``iteration_seed(random_seed, iteration)``, draws the Gumbel utility
        noise and then, with the outside option, its draw
        (``UtilityModel.draw_utility_model_randomness``); nothing else draws
        before ``PriorityGenerator._mtb`` takes the MTB lottery. A private
        ``RandomState`` seeded alike gives the same numbers in that order.
        """
        if self.utility is None:
            raise ValueError("This policy market was built without the utility model.")
        if not np.isfinite(gumbel_scale) or gumbel_scale < 0:
            raise ValueError("gumbel_scale must be a finite non-negative number.")
        rng = np.random.RandomState(iteration_seed(random_seed, iteration))
        mus = self.utility
        n, p = mus.shape
        if gumbel_scale == 0:
            utilities = mus.copy()
        else:
            utilities = np.array(mus + rng.gumbel(0, gumbel_scale, (n, p)))
        outside = None
        if outside_option:
            outside = np.zeros(n) if gumbel_scale == 0 else rng.gumbel(0, gumbel_scale, n)
        lottery = rng.rand(n, self.num_programs)
        return UtilityDraw(utilities, outside, lottery)

    def truncated_utility_lists(
        self,
        eligible: np.ndarray,
        utilities: np.ndarray,
        outside: np.ndarray | None,
        list_length: int | None,
    ) -> np.ndarray:
        """``PreferenceGenerator._truncate_utility_model_preferences``.

        Each student's eligible programs by drawn utility, best first, cut to
        ``list_length`` (``None``: those drawn above the outside option).
        ``eligible`` is program-type and zone eligibility.
        """
        eligible = np.asarray(eligible).astype(bool)
        order = np.argsort(-utilities, axis=1) + 1
        if list_length is None:
            if outside is None:
                raise ValueError("The outside-option list length needs its draw.")
            ranked = np.count_nonzero(eligible & (utilities > outside[:, None]), axis=1)
        else:
            ranked = np.full(self.n, list_length)
        ranked = np.minimum(ranked, eligible.sum(axis=1)).astype(int)
        kept = np.take_along_axis(eligible, order - 1, axis=1)
        position = np.cumsum(kept, axis=1)
        kept &= position <= ranked[:, None]
        truncated = np.zeros_like(order)
        rows, columns = np.nonzero(kept)
        truncated[rows, position[rows, columns] - 1] = order[rows, columns]
        return truncated

    def utility_preferences(
        self,
        eligibility: np.ndarray,
        appended: list[np.ndarray] | None,
        utilities: np.ndarray,
        outside: np.ndarray | None,
        list_length: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``get_utility_model_preferences_after_truncation`` with designation:
        ``(prefs, pref_length)``, the truncated drawn lists completed as
        :meth:`preferences` completes the submitted ones."""
        eligible = self.qualifies & (np.asarray(eligibility) > 0)
        truncated = self.truncated_utility_lists(
            eligible, utilities, outside, list_length
        )
        return self.preferences(eligibility, appended, truncated, dtype=truncated.dtype)

    def apply(
        self,
        zone_priority: np.ndarray,
        ctip,
        *,
        add_aa_schools: bool,
        generated_zones: bool,
        citywide_program_ids=(),
    ) -> AppliedPolicyMarket:
        """Everything the simulator builds for one zoning, before the lottery."""
        eligibility = self.zone_eligibility(zone_priority)
        appended = (
            self.appended_programs(
                zone_priority,
                generated_zones=generated_zones,
                citywide_program_ids=citywide_program_ids,
            )
            if add_aa_schools
            else None
        )
        prefs, pref_length = self.preferences(eligibility, appended)
        base = self.policy_priorities(zone_priority, ctip)
        priorities = priorities_without_lottery(
            self.round_priorities(),
            base,
            not_designation_mask(prefs, pref_length, self.num_programs),
            self.non_designation_boost,
            self.restrict_zone,
            eligibility,
        )
        return AppliedPolicyMarket(prefs, pref_length, eligibility, base, priorities)

    def attendance_area_zone_priority(self) -> np.ndarray:
        """The status-quo zone priority: each student's own area GE program."""
        matrix = np.zeros((self.n, self.num_programs), dtype=int)
        rows = np.flatnonzero(self.attendance_area_ge)
        matrix[rows, self.attendance_area_ge[rows] - 1] = 1
        return matrix

    def matches(self, students) -> bool:
        """Whether ``students`` (a ``Students``) are this market's, in order."""
        return (
            students.n == self.n
            and students.num_programs == self.num_programs
            and students.student_data.index.equals(self.student_index)
            and tuple(students.programs.program_df["program_id"]) == self.program_ids
        )


def _zone_matrix(zone_priority, market: PolicyMarket) -> np.ndarray:
    zone_priority = np.asarray(zone_priority)
    if zone_priority.shape != (market.n, market.num_programs):
        raise ValueError(
            f"Zone priority matrix shape {zone_priority.shape} does not match "
            f"{(market.n, market.num_programs)}."
        )
    if np.any((zone_priority != 0) & (zone_priority != 1)):
        raise ValueError("A zone priority matrix must be 0/1.")
    return zone_priority


# -- the simulator's side -----------------------------------------------------


def zones_are_set(zones) -> bool:
    """Whether a market's ``Zones`` hold a zoning (``set_area_id2prog_list_dict``)."""
    return zones is not None and hasattr(zones, "area_id2prog_list")


def policy_market_zone_eligibility(market, policy_market: PolicyMarket) -> np.ndarray:
    """The market's ``Zones.zone_eligibility_matrix``, computed from the policy
    market and kept on the zones for every later reader."""
    zones = market.zones
    if zones._zone_eligibility_matrix is None:
        zones._zone_eligibility_matrix = policy_market.zone_eligibility(
            zones.zone_priority_matrix
        )
    return zones._zone_eligibility_matrix


# -- building ------------------------------------------------------------------


class _StaticZones:
    """Zones whose priority matrix is set directly (``get_base_policy_priorities``)."""

    def __init__(self) -> None:
        self._zone_priority_matrix = None

    @property
    def zone_priority_matrix(self):
        return self._zone_priority_matrix


def build_policy_market(
    data_scenario,
    settings: Mapping,
    market_data: tuple | None = None,
) -> PolicyMarket:
    """Build the market from ``(programs, schools, students)`` (loaded if None).

    ``settings`` are :func:`policy_market_settings`; nothing else of the
    policy config is read, which is what makes them a sound cache key.
    """
    from .priority_generator import PriorityGenerator
    from .school_choice_market import load_market_data

    programs, schools, students = (
        load_market_data(data_scenario) if market_data is None else market_data
    )
    config = copy.deepcopy(dict(settings))
    if "peng" in config["priority-weights"]:
        raise ValueError("A policy market does not model PENG priority boosts.")
    grade = data_scenario.filter("assignment", "grade")
    n, width = students.n, programs.num_programs
    shape = (n, width)
    market = SimpleNamespace(
        config=config,
        data_scenario=data_scenario,
        grade=grade,
        programs=programs,
        schools=schools,
        students=students,
        n=n,
        num_programs=width,
        zones=_StaticZones(),
    )
    program_ids = tuple(str(p) for p in programs.program_df["program_id"])
    if [programs.indices[p] for p in program_ids] != list(range(1, width + 1)):
        raise ValueError("Program indices must follow the program table's order.")
    index_of = programs.indices
    frame = students.student_data
    if [students.studentno2idx[s] for s in frame.index] != list(range(n)):
        raise ValueError("Assignment student rows are out of order.")

    generator = PriorityGenerator(market)
    tiers = {}
    for ctip in config["ctip-options"]:
        tiers[ctip_key(ctip)] = tuple(
            generator.get_base_policy_priorities(
                "policy-market", ctip, zone_priority_matrix=np.full(shape, value, dtype=int)
            )
            for value in (1, 0)
        )

    # Zone eligibility without the zone (Zones.zone_eligibility_matrix).
    restrict = config["restrict-zone"]
    access = np.zeros(shape, dtype=np.int16)
    access += int(config["sibling-access"]) * students.sibling(programs)
    if config["tk-access"]:
        access += students.promotion(index_of)
    access += int(not restrict) * 1
    ctip1 = frame["ctip1"].fillna(0).to_numpy(dtype=int)
    access += int(restrict == "CTIP_access") * ctip1[:, None]

    qualifies = program_type_eligibility(programs, students)

    # Designation order: wanted language programs, then GE, nearest first.
    lp_columns, lp_order, ge_columns, ge_order = distance_orderings(
        students.distance_data, frame.index
    )
    lp_index = np.array([index_of[c] for c in lp_columns], dtype=np.int32)
    ge_index = np.array([index_of[c] for c in ge_columns], dtype=np.int32)
    lp_type = np.array([c.split("-")[1] for c in lp_columns], dtype=object)
    language_designation = students.language_designation.to_numpy()
    utility_settings = config.get("utility-model")
    if utility_settings is not None and utility_settings["designate-lp-for-all"]:
        # The simulator designates language programs for everyone then.
        language_designation = np.ones(n, dtype=bool)
    program_types = frame["program_types"].to_numpy()
    rows = []
    for row in range(n):
        order = ge_index[ge_order[row]]
        if language_designation[row]:
            value = program_types[row]
            wanted = (
                set()
                if value is None or isinstance(value, float)
                else set(value)
            )
            lp = [k for k in lp_order[row] if lp_type[k] in wanted]
            order = np.concatenate([lp_index[np.asarray(lp, dtype=int)], order])
        rows.append(order)
    designation_start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(r) for r in rows], out=designation_start[1:])
    designation = (
        np.concatenate(rows).astype(np.int32) if rows else np.zeros(0, dtype=np.int32)
    )

    # Each student's district attendance-area GE program.
    attendance_area_ge = np.zeros(n, dtype=np.int64)
    for row, value in enumerate(frame["idschoolattendance"]):
        if pd.isna(value):
            continue
        program = index_of.get(f"{int(value)}-GE-{grade}")
        if program is not None:
            attendance_area_ge[row] = program

    selected = np.asarray(students.selected_preferences(programs.index_list))
    attributes = pd.DataFrame(
        {
            column: frame[column]
            for column in STUDENT_AREA_COLUMNS
            if column in frame.columns
        },
        index=frame.index,
    )
    attributes["frl"] = students.frl

    classes, num_classes, strict, fractions = _reserves(
        config, frame, programs.program_df, schools.citywide_schools
    )
    utility = None
    if utility_settings is not None:
        from .utility_model import UtilityModel

        utility = UtilityModel(
            data_scenario.source("assignment.estimate").path, programs, students
        )._load_base_utility_matrix(students.only_keep_rows, programs.only_keep_cols)
        utility = np.asarray(utility, dtype=float)
    return PolicyMarket(
        settings=dict(settings),
        grade=grade,
        program_ids=program_ids,
        program_table=programs.program_df.copy(),
        citywide_schools=tuple(int(s) for s in schools.citywide_schools),
        student_index=frame.index.copy(),
        student_attributes=attributes,
        selected=selected.astype(np.int32),
        attendance_area_ge=attendance_area_ge,
        qualifies=qualifies,
        access=access,
        tiers=tiers,
        round_boost=round_priority_boost(students.first_round),
        non_designation_boost=config["non_designation_boost"],
        restrict_zone=restrict,
        designation_start=designation_start,
        designation=designation,
        reserve_classes=classes,
        num_classes=num_classes,
        strict_reserves=strict,
        reserve_fractions=fractions,
        utility=utility,
    )


def _reserves(config, frame, program_table, citywide_schools):
    """Reserve classes and fractions, as ``GuardrailSetup`` sizes them.

    Classes split ``reserve-settings.column`` at its thresholds (missing
    values take the mean); every GE program -- or every citywide-school GE
    program under ``citywide_only`` -- reserves ``reserve_fraction``; and
    citywide-school GE programs take ``citywide-reserve-ratios`` unless
    ``citywide-separate-reserves`` is off. Without reserves (``guard-rails:
    -1``) everyone is class 0 of 1. Zone-share reserves (no
    ``reserve_fraction``) give ``None`` fractions.
    """
    n = len(frame)
    if int(config["guard-rails"]) == -1:
        return np.zeros(n, dtype=int), 1, False, {}
    settings = config["reserve-settings"] or {}
    column = settings["column"]
    if isinstance(column, dict):
        index = sum(weight * frame[name] for name, weight in column.items())
    else:
        index = frame[column]
    values = pd.to_numeric(index, errors="coerce").to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=np.nanmean(values))
    if "num_categories" in settings:
        count = int(settings["num_categories"])
        thresholds = np.percentile(values, 100 * np.arange(1, count) / count)
    elif "thresholds" in settings:
        thresholds = settings["thresholds"]
        if isinstance(thresholds, str) and thresholds.startswith("percentile:"):
            thresholds = np.atleast_1d(
                np.percentile(values, float(thresholds.split(":")[1]))
            )
        thresholds = np.asarray(thresholds, dtype=float)
    else:
        raise ValueError("reserve-settings needs num_categories or thresholds.")
    classes = np.searchsorted(thresholds, values)
    if not settings["lower_disadvantaged"]:
        classes = len(thresholds) - classes
    num_classes = len(thresholds) + 1
    strict = int(config["guard-rails"]) == 1

    fraction = settings.get("reserve_fraction", -1)
    if fraction == -1:
        return classes, num_classes, strict, None
    fraction = tuple(float(value) for value in fraction)
    ratios = tuple(float(value) for value in config["citywide-reserve-ratios"])
    if len(fraction) != num_classes or len(ratios) != num_classes:
        raise ValueError("Reserve fractions must give one share per class.")
    citywide = {int(value) for value in citywide_schools}
    separate = config["citywide-separate-reserves"]
    citywide_only = settings.get("citywide_only", False)
    fractions = {}
    for program_id, school_id in zip(
        program_table["program_id"], program_table["school_id"], strict=True
    ):
        parts = str(program_id).split("-")
        if len(parts) < 2 or parts[1] != "GE":
            continue
        citywide_ge = int(school_id) in citywide
        if separate and citywide_ge:
            fractions[str(program_id)] = ratios
        elif citywide_ge or not citywide_only:
            fractions[str(program_id)] = fraction
    return classes, num_classes, strict, fractions


# -- caching -------------------------------------------------------------------


def policy_market_namespace(data_scenario, settings: Mapping):
    """The cache namespace of one (data scenario, policy settings) market."""
    from loaders.cache import CacheStore

    from .school_choice_market import assignment_source_roles

    return CacheStore(data_scenario).namespace(
        POLICY_MARKET_ARTIFACT,
        {"settings": dict(settings)},
        schema_version=POLICY_MARKET_SCHEMA_VERSION,
        roles=assignment_source_roles(data_scenario),
        classification="restricted-derived",
    )


@contextmanager
def _build_lock(namespace):
    """Exclusive lock held while one process builds a namespace's market."""
    namespace.version_dir.mkdir(parents=True, exist_ok=True)
    path = namespace.version_dir / f".{namespace.key}.build.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o660)
    with os.fdopen(descriptor, "r+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _load(namespace, settings) -> PolicyMarket | None:
    market = namespace.load_pickle(POLICY_MARKET_PAYLOAD)
    if isinstance(market, PolicyMarket) and market.settings == dict(settings):
        return market
    return None


def load_policy_market(
    data_scenario,
    policy_config: Mapping,
    *,
    market_data: tuple | Callable[[], tuple] | None = None,
) -> PolicyMarket:
    """The cached :class:`PolicyMarket` of a data scenario and policy config.

    ``market_data`` is the ``(programs, schools, students)`` to build from on
    a miss (or a callable returning them); by default they are loaded.
    """
    settings = policy_market_settings(policy_config)
    namespace = policy_market_namespace(data_scenario, settings)
    market = _MEMO.get(namespace.key)
    if market is None:
        market = _load(namespace, settings)
    if market is None:
        with _build_lock(namespace):
            market = _load(namespace, settings)
            if market is None:
                data = market_data() if callable(market_data) else market_data
                market = build_policy_market(data_scenario, settings, data)
                namespace.save_pickle(POLICY_MARKET_PAYLOAD, market)
    _MEMO[namespace.key] = market
    _MEMO.move_to_end(namespace.key)
    while len(_MEMO) > _MEMO_SIZE:
        _MEMO.popitem(last=False)
    return market


def clear_policy_market_memo() -> None:
    """Forget the in-process markets (the disk cache is kept)."""
    _MEMO.clear()


__all__ = [
    "OUTSIDE_OPTION_LIST_LENGTH",
    "POLICY_MARKET_SCHEMA_VERSION",
    "POLICY_MARKET_SETTINGS",
    "AppliedPolicyMarket",
    "UtilityDraw",
    "PolicyMarket",
    "build_policy_market",
    "ctip_key",
    "iteration_seed",
    "distance_orderings",
    "load_policy_market",
    "not_designation_mask",
    "policy_market_namespace",
    "policy_market_settings",
    "priorities_without_lottery",
    "program_type_eligibility",
    "policy_market_whitelist",
    "unmodelled_policy_settings",
    "utility_list_length",
    "utility_model_enabled",
]
