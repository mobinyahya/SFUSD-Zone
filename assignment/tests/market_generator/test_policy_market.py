"""The shared policy market: what applying a zoning yields, and its cache."""

from __future__ import annotations

import hashlib
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment.market_generator import policy_market as pm
from loaders.cache import CacheNamespace

PROGRAMS = ("10-GE-KG", "10-SN-KG", "20-GE-KG", "30-GE-KG")
SETTINGS = pm.policy_market_settings(
    {
        "ctip-options": [1],
        "priority-weights": {"zone": 4, "sibling": 16},
        "restrict-zone": True,
        "sibling-access": True,
    }
)


def _toy_market(settings=SETTINGS) -> pm.PolicyMarket:
    """Two students, four programs; 30 is a citywide school.

    Student a ranked 20-GE then 10-SN, lives in 10's area, asked for language
    designation and has a sibling at 20. Student b ranked nothing and lives
    in 20's area.
    """
    n, p = 2, len(PROGRAMS)
    inside = np.full((n, p), 7.0)
    outside = np.full((n, p), 3.0)
    access = np.zeros((n, p), dtype=np.int16)
    access[0, 2] = 1  # a's sibling opens 20-GE
    designation = [
        np.array([2, 1, 3, 4], dtype=np.int32),  # a: 10-SN, then GE nearest first
        np.array([3, 4, 1], dtype=np.int32),  # b: GE only
    ]
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(d) for d in designation], out=start[1:])
    return pm.PolicyMarket(
        settings=dict(settings),
        grade="KG",
        program_ids=PROGRAMS,
        program_table=pd.DataFrame(
            {"program_id": PROGRAMS, "school_id": [10, 10, 20, 30]}
        ),
        citywide_schools=(30,),
        student_index=pd.Index(["a", "b"], name="studentno"),
        student_attributes=pd.DataFrame({"frl": [0.0, 1.0]}, index=["a", "b"]),
        selected=np.array([[3, 2, 0, 0], [0, 0, 0, 0]], dtype=np.int32),
        attendance_area_ge=np.array([1, 3]),
        qualifies=np.ones((n, p), dtype=bool),
        access=access,
        tiers={pm.ctip_key(1): (inside, outside)},
        round_boost=np.array([0, 1000]),
        non_designation_boost=100.0,
        restrict_zone=True,
        designation_start=start,
        designation=np.concatenate(designation),
        reserve_classes=np.zeros(n, dtype=int),
        num_classes=1,
        strict_reserves=False,
        reserve_fractions={},
    )


def test_applying_a_zoning_builds_the_simulator_lists_and_priorities():
    market = _toy_market()
    # One zone owning 10 and 20, and 30 open to everyone (restrict-zone gives
    # the citywide program to every zone).
    zone = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    applied = market.apply(zone, 1, add_aa_schools=True, generated_zones=False)
    # a: 20-GE, 10-SN listed, 10-GE appended (own area), then 30-GE designated.
    # b: 20-GE appended, then designations 30-GE, 10-GE.
    np.testing.assert_array_equal(
        applied.preferences, [[3, 2, 1, 4], [3, 4, 1, 0]]
    )
    np.testing.assert_array_equal(applied.pref_length, [3, 1])
    np.testing.assert_array_equal(applied.policy_priorities, np.full((2, 4), 7.0))
    # Listed programs: round + tier + boost; designations: the round alone.
    np.testing.assert_array_equal(
        applied.priorities, [[107, 107, 107, 0], [1000, 1000, 1107, 1000]]
    )

    # Zones {10} and {20, 30}: a may enter 10's programs and, by sibling
    # access, 20-GE at its out-of-zone tier; nothing at 30.
    zone = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    applied = market.apply(zone, 1, add_aa_schools=True, generated_zones=True)
    np.testing.assert_array_equal(applied.zone_eligibility, [[1, 1, 1, 0], [0, 0, 1, 1]])
    np.testing.assert_array_equal(applied.preferences, [[3, 2, 1, 0], [3, 4, 0, 0]])
    np.testing.assert_array_equal(applied.pref_length, [3, 2])
    np.testing.assert_array_equal(
        applied.policy_priorities, [[7, 7, 3, 3], [3, 3, 7, 7]]
    )
    np.testing.assert_array_equal(
        applied.priorities, [[107, 107, 103, -500], [-500, -500, 1107, 1107]]
    )


def test_generated_zones_append_every_owned_ge_program_in_program_order():
    market = _toy_market()
    zone = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    appended = market.appended_programs(
        zone, generated_zones=True, citywide_program_ids=["30-GE-KG"]
    )
    assert [list(a) for a in appended] == [[1, 3], []]
    appended = market.appended_programs(zone, generated_zones=False)
    assert [list(a) for a in appended] == [[1], [3]]


def test_a_zone_priority_matrix_must_be_zero_one_and_market_shaped():
    market = _toy_market()
    with pytest.raises(ValueError, match="0/1"):
        market.zone_eligibility(np.full((2, 4), 2))
    with pytest.raises(ValueError, match="shape"):
        market.zone_eligibility(np.ones((3, 4)))


def test_distance_orderings_match_the_row_by_row_argsort():
    rng = np.random.default_rng(0)
    columns = [f"{school}-{kind}-KG" for school in range(30) for kind in ("GE", "SN", "CE")]
    values = rng.integers(0, 6, size=(200, len(columns))).astype(float)
    # Programs at one school share a distance, so the order must survive ties.
    values = np.repeat(values[:, ::3], 3, axis=1)
    distances = pd.DataFrame(values, columns=columns, index=rng.permutation(200))
    lp_columns, lp_order, ge_columns, ge_order = pm.distance_orderings(distances)
    for name, cols, order in (("LP", lp_columns, lp_order), ("GE", ge_columns, ge_order)):
        frame = distances[cols]
        expected = frame.apply(
            lambda row: pd.Series(frame.columns[np.argsort(row.values)]), axis=1
        )
        np.testing.assert_array_equal(
            np.asarray(cols, dtype=object)[order], expected.to_numpy(), err_msg=name
        )


def test_settings_fill_the_simulator_defaults_and_drop_unused_reserves():
    settings = pm.policy_market_settings(
        {
            "ctip-options": [1],
            "priority-weights": {"zone": 4},
            "reserve-settings": {"column": "x"},
        }
    )
    assert settings["non_designation_boost"] == 100
    assert settings["distance-priority"] == {"step-size": 10}
    assert settings["reserve-settings"] == {}  # guard-rails -1: no reserves
    with pytest.raises(ValueError, match="priority-weights"):
        pm.policy_market_settings({"ctip-options": [1]})
    # Settings outside the market (outputs, zone files) do not key it.
    assert pm.policy_market_settings(
        {"ctip-options": [1], "priority-weights": {"zone": 4}, "iterations": 3}
    ) == pm.policy_market_settings({"ctip-options": [1], "priority-weights": {"zone": 4}})


def _namespace(root: Path, settings) -> CacheNamespace:
    key = hashlib.sha256(pm.ctip_key(settings).encode()).hexdigest()
    return CacheNamespace(
        root=root,
        artifact=pm.POLICY_MARKET_ARTIFACT,
        schema_version=pm.POLICY_MARKET_SCHEMA_VERSION,
        key=key,
        parameters={"settings": settings},
        source_manifest={"sources": {}},
        classification="restricted-derived",
    )


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Route the cache to ``tmp_path`` and count builds in a file."""
    builds = tmp_path / "builds.log"

    def build(data_scenario, settings, market_data=None):
        with builds.open("a") as stream:
            stream.write("build\n")
        time.sleep(0.3)
        return _toy_market(settings)

    monkeypatch.setattr(
        pm, "policy_market_namespace", lambda data, settings: _namespace(tmp_path, settings)
    )
    monkeypatch.setattr(pm, "build_policy_market", build)
    pm.clear_policy_market_memo()
    yield builds
    pm.clear_policy_market_memo()


def _builds(path: Path) -> int:
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_the_market_is_built_once_then_memoised_then_loaded(cache):
    config = {"ctip-options": [1], "priority-weights": {"zone": 4, "sibling": 16},
              "restrict-zone": True, "sibling-access": True}
    first = pm.load_policy_market(None, config)
    assert _builds(cache) == 1
    assert pm.load_policy_market(None, config) is first
    pm.clear_policy_market_memo()
    loaded = pm.load_policy_market(None, config)
    assert _builds(cache) == 1 and loaded is not first
    np.testing.assert_array_equal(loaded.designation, first.designation)
    pm.load_policy_market(None, {**config, "non_designation_boost": 64})
    assert _builds(cache) == 2


def _load_in_child(config):
    pm.load_policy_market(None, config)


def test_concurrent_processes_build_the_market_once(cache):
    config = {"ctip-options": [1], "priority-weights": {"zone": 4}}
    context = multiprocessing.get_context("fork")
    workers = [context.Process(target=_load_in_child, args=(config,)) for _ in range(3)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
    assert [worker.exitcode for worker in workers] == [0, 0, 0]
    assert _builds(cache) == 1


# -- the whitelist ---------------------------------------------------------------

POLICY_CONFIG = Path(__file__).resolve().parents[3] / (
    "assignment/configs/policy_configs/restrictive_aa_sib_ctip_reserves.yaml"
)
#: (setting, unmodelled value): both consumers leave the shared path.
UNMODELLED = [
    ("ties-options", ["STB"]),
    ("ties-options", ["MTB", "STB"]),
    ("citywide-or-lp", ["language_zones"]),
    (
        "restrict-zone-options",
        [
            {"restrict-zone": True, "citywide-or-lp": []},
            {"restrict-zone": False, "citywide-or-lp": []},
        ],
    ),
    ("restrict-zone-options", [{"restrict-zone": False, "citywide-or-lp": []}]),
    ("read-lotteries", True),
    # Zone-share reserves: no reserve_fraction.
    (
        "reserve-settings",
        {"column": "freelunch_prob", "num_categories": 2, "lower_disadvantaged": True},
    ),
    (
        "guard-rails-reserve-options",
        [
            {"guard-rails": 0, "reserve-settings": {}},
            {"guard-rails": -1, "reserve-settings": {}},
        ],
    ),
    # The utility model with what the shared path does not model: no or an
    # unknown list length, a bad Gumbel scale, an unknown key, or a non-bool
    # enable.
    ("utility-model", {"enable": True}),
    ("utility-model", {"enable": True, "list-length": "real_length"}),
    ("utility-model", {"enable": True, "list-length": 7}),
    ("utility-model", {"enable": True, "list-length": "7", "gumbel-scale": -1}),
    ("utility-model", {"enable": True, "list-length": "7", "gumbel-scale": True}),
    ("utility-model", {"enable": True, "list-length": "7", "shuffle": True}),
    ("utility-model", {"enable": "yes", "list-length": "7"}),
    ("assignment-algorithm", "TTC"),
    ("designation-ordering-type", "simple"),
    ("drop_below_aa", True),
    ("remove_non_aa_or_citywide", True),
    ("aa_ge_only", True),
    ("truncate-at-AA-GE", True),
    ("restrict-zone", "sometimes"),
    ("guard-rails", 2),
    ("priority-weights", {"zone": 4, "peng": 1}),
    ("priority-weights", {"zone": 4, "brown-ms": {"sibling": 1}}),
    ("ctip-options", ["5D"]),
]
#: Exact on the simulator's shared path, not modelled by the lottery DA.
MID_ONLY = [
    ("ctip-options", [1, 0]),
    ("policies", ["Con1", "Con2"]),
    ("overscribe_aa", True),
    ("aa_boost", 2),
    ("distance-priority", {"thresholds": [0.5, 3]}),
    ("distance-boost", {"low_income_boost": 0.2}),
    ("priority-weights", {"zone": 4, "distance": 1}),
]


def _policy(**overrides):
    import yaml

    config = yaml.safe_load(POLICY_CONFIG.read_text())
    config.update(overrides)
    return config


def _names(reasons):
    return [reason.split("=")[0] for reason in reasons]


def test_the_11_18_policy_is_on_the_whitelist_for_both_consumers():
    assert pm.unmodelled_policy_settings(_policy(), "simulator") == []
    assert pm.unmodelled_policy_settings(_policy(), "mid") == []
    # An options list restating the top-level settings is what the simulator
    # copies over them anyway; soft_reserve_boost only labels the output.
    config = _policy(soft_reserve_boost=3)
    config["guard-rails-reserve-options"] = [
        {"guard-rails": config["guard-rails"], "reserve-settings": config["reserve-settings"]}
    ]
    assert pm.unmodelled_policy_settings(config, "mid") == []


def _simulator_market(config):
    from types import SimpleNamespace

    from assignment.student_assignment.market_generator.school_choice_market import (
        SchoolChoiceMarket,
    )

    market = SchoolChoiceMarket.__new__(SchoolChoiceMarket)
    market.config = config
    market.zones = SimpleNamespace(area_id2prog_list={})
    return market


@pytest.mark.parametrize(("name", "value"), UNMODELLED)
def test_an_unmodelled_setting_routes_the_simulator_to_its_own_code(name, value):
    from assignment.student_assignment.market_generator.preference_generator import (
        PreferenceGenerator,
    )
    from assignment.student_assignment.market_generator.priority_generator import (
        PriorityGenerator,
    )

    config = _policy(**{name: value})
    assert _names(pm.unmodelled_policy_settings(config, "simulator")) == [name]
    market = _simulator_market(config)
    # policy_market() decides before touching any data.
    assert market.policy_market() is None
    assert PreferenceGenerator(market)._policy_market() is None
    assert PriorityGenerator(market)._policy_market() is None


@pytest.mark.parametrize(("name", "value"), MID_ONLY)
def test_settings_only_the_simulator_reproduces_keep_its_shared_path(name, value):
    config = _policy(**{name: value})
    assert pm.unmodelled_policy_settings(config, "simulator") == []
    assert _names(pm.unmodelled_policy_settings(config, "mid")) == [name]


@pytest.mark.parametrize(("name", "value"), UNMODELLED + MID_ONLY)
def test_mid_raises_naming_each_unmodelled_setting(name, value, monkeypatch):
    from types import SimpleNamespace

    from optimization.data import mid

    config = _policy(**{name: value})
    monkeypatch.setattr(mid, "policy_config", lambda optimization_config: config)

    def no_market(*args, **kwargs):
        raise AssertionError("an unmodelled policy must fail before loading data")

    monkeypatch.setattr(mid, "load_policy_market", no_market)
    optimization_config = SimpleNamespace(
        mid_utility_handling="omit_nonpositive",
        matching_policy="restrictive_aa_sib_ctip_reserves",
        mid_preference_source="stated",
        data_scenario=None,
    )
    with pytest.raises(ValueError, match=f"{name}=") as error:
        mid.build_mid_student_market(None, optimization_config)
    assert "does not model" in str(error.value)


def test_mid_names_every_unmodelled_setting_at_once():
    from optimization.data.mid import _require_modelled_policy

    config = _policy(**{"ties-options": ["STB"], "read-lotteries": True})
    with pytest.raises(ValueError, match="ties-options=.*read-lotteries="):
        _require_modelled_policy(config, "p")


# -- the utility model -----------------------------------------------------------

#: Utility-model blocks both consumers model.
UTILITY_MODELS = [
    {"enable": True, "list-length": "7"},
    {"enable": True, "list-length": "outside_option"},
    {
        "enable": True,
        "list-length": "3",
        "gumbel-scale": 0.5,
        "designate-lp-for-all": True,
        "save-path": "/tmp/utilities.npy",
    },
    {"enable": True, "list-length": "7", "gumbel-scale": 0},
]


@pytest.mark.parametrize("utility_model", UTILITY_MODELS)
def test_the_utility_model_is_on_the_whitelist_for_both_consumers(utility_model):
    config = _policy(**{"utility-model": utility_model})
    assert pm.utility_model_enabled(config)
    assert pm.unmodelled_policy_settings(config, "simulator") == []
    assert pm.unmodelled_policy_settings(config, "mid") == []


@pytest.mark.parametrize(
    "utility_model",
    [None, {"enable": None, "list-length": "7"}, {"enable": False, "list-length": "x"}],
)
def test_an_unset_or_disabled_utility_model_means_the_submitted_lists(utility_model):
    config = _policy(**{"utility-model": utility_model})
    assert not pm.utility_model_enabled(config)
    assert pm.unmodelled_policy_settings(config, "mid") == []
    # It does not key the market either: the submitted-list caches stay put.
    assert pm.policy_market_settings(config) == pm.policy_market_settings(_policy())


def test_the_utility_model_needs_designation():
    config = _policy(
        **{"utility-model": UTILITY_MODELS[0], "designate": False}
    )
    assert _names(pm.unmodelled_policy_settings(config, "simulator")) == [
        "utility-model"
    ]


def test_an_enabled_utility_model_keys_the_market_by_what_it_reads():
    settings = {
        repr(model): pm.policy_market_settings(_policy(**{"utility-model": model}))
        for model in UTILITY_MODELS
    }
    seven, outside, lp_for_all, _ = settings.values()
    # The list length and Gumbel scale act at draw time: one market for both.
    assert seven == outside
    assert seven["utility-model"] == {"designate-lp-for-all": False}
    assert lp_for_all["utility-model"] == {"designate-lp-for-all": True}
    assert "utility-model" not in pm.policy_market_settings(_policy())


def test_list_lengths():
    assert pm.utility_list_length("7") == 7
    assert pm.utility_list_length("outside_option") is None
    for value in ("real_length", "7.5", 7, None, "\u00b2"):
        with pytest.raises(ValueError, match="list-length"):
            pm.utility_list_length(value)


def test_the_iteration_seed_is_the_simulators():
    from assignment.student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

    for seed, iteration in ((2023, 0), (2023, 7), (0, 3), (17, 120)):
        assert pm.iteration_seed(seed, iteration) == MarketGenerator.iteration_seed(
            seed, iteration
        )


def _utility_market(rng, n=40, p=12):
    market = _toy_market()
    qualifies = rng.random((n, p)) < 0.8
    designation = [rng.permutation(p)[: rng.integers(1, p)] + 1 for _ in range(n)]
    start = np.zeros(n + 1, dtype=np.int64)
    np.cumsum([len(d) for d in designation], out=start[1:])
    selected = np.zeros((n, p), dtype=np.int32)
    return pm.PolicyMarket(
        **{
            **{name: getattr(market, name) for name in market.__dataclass_fields__},
            "program_ids": tuple(f"{10 * (j + 1)}-GE-KG" for j in range(p)),
            "student_index": pd.Index([str(i) for i in range(n)], name="studentno"),
            "selected": selected,
            "attendance_area_ge": rng.integers(0, p + 1, size=n),
            "qualifies": qualifies,
            "access": np.zeros((n, p), dtype=np.int16),
            "round_boost": np.zeros(n),
            "designation_start": start,
            "designation": np.concatenate(designation).astype(np.int32),
            "reserve_classes": np.zeros(n, dtype=int),
            "utility": rng.normal(-1.0, 2.0, size=(n, p)),
        }
    )


@pytest.mark.parametrize(("scale", "outside"), [(1.0, False), (0.7, True), (0, True)])
def test_utility_draws_replay_the_simulators_stream(scale, outside):
    from assignment.student_assignment.market_generator.utility_model import (
        UtilityModel,
    )

    market = _utility_market(np.random.default_rng(0))
    model = UtilityModel("estimate.npy", None, None)
    model._base_utility_cache_key = ("estimate.npy", None, None)
    model._base_utility_matrix = market.utility
    for iteration in (0, 4):
        np.random.seed(pm.iteration_seed(2023, iteration))
        model.draw_utility_model_randomness(
            iteration, gumbel_scale=scale, outside_option=outside
        )
        lottery = np.random.rand(market.n, market.num_programs)
        draw = market.utility_draws(
            iteration, 2023, gumbel_scale=scale, outside_option=outside
        )
        np.testing.assert_array_equal(draw.utilities, model.original_utilities)
        np.testing.assert_array_equal(draw.lottery, lottery)
        if outside:
            np.testing.assert_array_equal(draw.outside, model.outside_utilities)
        else:
            assert draw.outside is None


@pytest.mark.parametrize("length", ["0", "1", "4", "outside_option"])
def test_truncation_is_the_simulators(length):
    from types import SimpleNamespace

    from assignment.student_assignment.market_generator.preference_generator import (
        PreferenceGenerator,
    )

    rng = np.random.default_rng(1)
    market = _utility_market(rng)
    draw = market.utility_draws(0, 5, outside_option=True)
    eligible = market.qualifies & (rng.random(market.qualifies.shape) < 0.7)
    simulator = SimpleNamespace(
        n=market.n,
        num_programs=market.num_programs,
        config={"utility-model": {"enable": True, "list-length": length}},
        grade="KG",
        students=SimpleNamespace(student_data=None, attendance_area={}),
        umodel=SimpleNamespace(
            original_utilities=draw.utilities,
            original_preferences=np.argsort(-draw.utilities, axis=1) + 1,
            outside_utilities=draw.outside,
        ),
    )
    expected = PreferenceGenerator(simulator)._truncate_utility_model_preferences(
        eligible.astype(float)
    )
    actual = market.truncated_utility_lists(
        eligible, draw.utilities, draw.outside, pm.utility_list_length(length)
    )
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("utility_model", "source", "required"),
    [
        (UTILITY_MODELS[0], "stated", "estimate"),
        (None, "estimate", "stated"),
    ],
)
def test_mid_needs_the_preference_source_the_policy_matches(
    utility_model, source, required, monkeypatch
):
    """Submitted lists are ``stated``; lists drawn from the simulator's
    utility estimate are ``estimate``."""
    from types import SimpleNamespace

    from optimization.data import mid

    config = _policy(**{"utility-model": utility_model})
    monkeypatch.setattr(mid, "policy_config", lambda optimization_config: config)

    def no_market(*args, **kwargs):
        raise AssertionError("a mismatched source must fail before loading data")

    monkeypatch.setattr(mid, "load_policy_market", no_market)
    optimization_config = SimpleNamespace(
        mid_utility_handling="omit_nonpositive",
        matching_policy="restrictive_aa_sib_ctip_reserves",
        mid_preference_source=source,
        data_scenario=None,
    )
    with pytest.raises(ValueError, match=f"set mid_preference_source: {required}"):
        mid.build_mid_student_market(None, optimization_config)
