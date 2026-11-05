import pytest

from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec, level_unit


@pytest.mark.parametrize(
    "text,unit,depth",
    [
        ("BlockGroup_0", "BlockGroup", 0),
        ("Block_2", "Block", 2),
        ("Tract_0", "Tract", 0),
        ("attendance_area_1", "attendance_area", 1),
    ],
)
def test_parse_roundtrip(text, unit, depth):
    spec = LevelSpec.parse(text)
    assert spec.unit == unit
    assert spec.depth == depth
    assert str(spec) == text
    assert spec.filename == f"{text}.pickle"


def test_parse_passthrough():
    spec = LevelSpec("Block", 1)
    assert LevelSpec.parse(spec) is spec


def test_is_base():
    assert LevelSpec("BlockGroup", 0).is_base
    assert not LevelSpec("BlockGroup", 2).is_base


@pytest.mark.parametrize(
    "bad", ["BlockGroup", "Block_x", "Foo_0", "", "Block_.5", "Block_0.", "Block_0.5", "Block_-1"]
)
def test_malformed(bad):
    with pytest.raises(ValueError):
        LevelSpec.parse(bad)


def test_parent_follows_the_chain_by_default():
    assert LevelSpec("Block", 2).parent() == LevelSpec("Block", 1)
    assert LevelSpec("BlockGroup", 1).parent() == LevelSpec("BlockGroup", 0)
    with pytest.raises(ValueError):
        LevelSpec("Block", 0).parent()


def test_block_levels_are_a_coarsening_chain():
    targets = LEVEL_NODE_TARGETS["Block"]
    assert sorted(targets) == list(range(1, len(targets) + 1))
    assert all(targets[d] > targets[d + 1] for d in range(1, len(targets)))
    assert LevelSpec("Block", 1).parent() == LevelSpec("Block", 0)


def test_level_unit_reads_retired_level_names():
    assert level_unit("Block_0.5") == "Block"
    assert level_unit("attendance_area_1") == "attendance_area"
    assert level_unit(LevelSpec("BlockGroup", 1)) == "BlockGroup"
    with pytest.raises(ValueError):
        level_unit("Foo_1")
