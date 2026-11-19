import gzip
import io
import json

import numpy as np
import pandas as pd
import pytest

from assignment.student_assignment import choice_ranks
from assignment.student_assignment.choice_ranks import ASSIGNMENT_SCHEMA_VERSION
from assignment.student_assignment.lottery_files import (
    LotteryFileError,
    lottery_file_paths,
    read_lotteries,
    read_saved_lottery,
    remove_lotteries,
    render_lottery,
    saved_lotteries,
    saved_lottery_exists,
    write_lotteries,
)


def _frame(seed, *, utility=False, all_assigned=False):
    rng = np.random.default_rng(seed)
    n = 7
    programno = rng.integers(0 if not all_assigned else 1, 4, n).astype(float)
    codes = {1.0: "101-GE-KG", 2.0: "101-SN-KG", 3.0: "202-GE-KG"}
    assigned = programno > 0
    rank = np.where(assigned, rng.integers(1, 6, n), np.nan)
    frame = pd.DataFrame(
        {
            "assignment_schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "studentno": np.arange(100, 100 + n),
            "programno": programno,
            "programcodes": [codes.get(value, np.nan) for value in programno],
            "rank_basis": "utility" if utility else "listed",
            "submitted_rank": rank,
            "rank_excluding_promotion": rank,
            "utility_rank": rank if utility else np.nan,
            "rank": rank,
            "mechanism_rank": rank,
            "designation": np.zeros(n, dtype=int),
            "overage_seat": rng.integers(0, 2, n).astype(bool),
            "In-Zone Rank": rank,
        }
    )
    if utility:
        frame["assigned_utility"] = rng.normal(size=n) / 3
    return frame


def _as_read(frame):
    """What ``pd.read_csv`` returned for a per-lottery file of ``frame``."""
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)))


@pytest.mark.parametrize("utility", [False, True])
def test_lotteries_rebuild_the_per_lottery_frames_exactly(tmp_path, utility):
    path = tmp_path / "variant_lotteries.csv.gz"
    frames = {lottery: _frame(lottery, utility=utility) for lottery in range(4)}
    frames[3] = _frame(3, utility=utility, all_assigned=True)

    write_lotteries(path, {k: render_lottery(v) for k, v in frames.items()})
    saved = read_lotteries(path)

    assert saved.lotteries == [0, 1, 2, 3]
    for lottery, frame in frames.items():
        pd.testing.assert_frame_equal(saved.frame(lottery), _as_read(frame))
    derived = saved.sidecar["derived_columns"]
    assert "programcodes" in derived and "assignment_schema_version" in derived
    assert ("utility_rank" in derived) is not utility
    with gzip.open(path, "rt") as stream:
        header = stream.readline().strip().split(",")
    assert header[0] == "lottery"
    assert not set(derived) & set(header)


def test_upserts_keep_sibling_lotteries_and_their_exact_text(tmp_path):
    path = tmp_path / "variant_lotteries.csv.gz"
    first = {0: _frame(0), 1: _frame(1)}
    write_lotteries(path, {k: render_lottery(v) for k, v in first.items()})
    before = read_lotteries(path).frame(0)

    write_lotteries(path, {2: render_lottery(_frame(2, all_assigned=True))})
    remove_lotteries(path, [1])

    saved = read_lotteries(path)
    assert saved.lotteries == [0, 2]
    pd.testing.assert_frame_equal(saved.frame(0), before)
    assert saved.lottery_text(0) == read_lotteries(path).lottery_text(0)

    remove_lotteries(path, [0, 2])
    assert not path.exists()
    assert saved_lotteries(path) == set()


def test_inconsistent_derivable_columns_stay_in_the_data(tmp_path):
    path = tmp_path / "variant_lotteries.csv.gz"
    mixed = _frame(0)
    mixed.loc[0, "rank_basis"] = "utility"
    mixed.loc[0, "utility_rank"] = mixed.loc[0, "rank"]
    write_lotteries(path, {0: render_lottery(mixed), 1: render_lottery(_frame(1))})

    saved = read_lotteries(path)
    assert {"rank_basis", "utility_rank"} <= set(saved.stored_columns)
    pd.testing.assert_frame_equal(saved.frame(0), _as_read(mixed))


def test_a_corrupt_data_file_is_rejected(tmp_path):
    path = tmp_path / "variant_lotteries.csv.gz"
    write_lotteries(path, {0: render_lottery(_frame(0))})
    path.write_bytes(gzip.compress(b"lottery,studentno\n0,1\n"))

    with pytest.raises(LotteryFileError, match="checksum"):
        read_lotteries(path)


def test_mismatched_columns_cannot_share_a_file(tmp_path):
    path = tmp_path / "variant_lotteries.csv.gz"
    write_lotteries(path, {0: render_lottery(_frame(0))})

    with pytest.raises(LotteryFileError, match="differ"):
        write_lotteries(path, {1: render_lottery(_frame(1, utility=True))})


def test_per_lottery_paths_read_either_layout(tmp_path):
    legacy = tmp_path / "variant" / "variant_iteration3.csv"
    legacy.parent.mkdir()
    data_path, sidecar_path = lottery_file_paths(legacy)
    assert data_path.name == "variant_lotteries.csv.gz"
    assert sidecar_path.name == "variant_lotteries.json"

    old = _frame(3).assign(assignment_schema_version=3)
    old.to_csv(legacy, index=False)
    assert saved_lottery_exists(legacy)
    pd.testing.assert_frame_equal(read_saved_lottery(legacy), _as_read(old))

    new = _frame(4)
    write_lotteries(data_path, {3: render_lottery(new)})
    pd.testing.assert_frame_equal(read_saved_lottery(legacy), _as_read(new))
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["schema_version"] == ASSIGNMENT_SCHEMA_VERSION
    assert sidecar["lotteries"] == [3]
    assert sidecar["rows"] == {"3": len(new)}


def test_legacy_schema_three_assignments_normalize_to_the_current_version():
    frame = _frame(0).assign(assignment_schema_version=3)
    normalized = choice_ranks.normalize_assignment_ranks(_as_read(frame))
    assert normalized["assignment_schema_version"].eq(ASSIGNMENT_SCHEMA_VERSION).all()
    with pytest.raises(ValueError, match="schema version"):
        choice_ranks.normalize_assignment_ranks(
            _as_read(frame.assign(assignment_schema_version=2))
        )


@pytest.mark.parametrize(
    "values",
    [
        [1, 2, 3],
        [np.int64(4), np.int32(1)],
        [1.0, 2.0],
        [np.float64(3.0)],
        ["1", "2"],
        [True, 2],
        [1, 1],
        [0, 1],
        [-1],
        [1.5],
        [np.nan],
        [np.inf],
        [None],
        ["x"],
        [],
    ],
)
def test_fast_rank_validation_matches_the_pandas_route(values):
    def outcome(function):
        try:
            return ("ok", function(list(values), "ranks"))
        except ValueError as exc:
            return ("error", str(exc))

    assert outcome(choice_ranks._validated_ranks) == outcome(
        choice_ranks._validated_ranks_slow
    )
