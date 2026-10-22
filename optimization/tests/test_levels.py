import pytest

from optimization.levels import LEVEL_NODE_TARGETS, LevelSpec


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


def test_navigation():
    spec = LevelSpec("BlockGroup", 2)
    assert spec.base() == LevelSpec("BlockGroup", 0)
    assert spec.finer() == LevelSpec("BlockGroup", 1)
    assert spec.coarser() == LevelSpec("BlockGroup", 3)
    assert LevelSpec("BlockGroup", 0).is_base


@pytest.mark.parametrize(
    "bad", ["BlockGroup", "Block_x", "Foo_0", "", "Block_.5", "Block_0.", "Block_-1"]
)
def test_malformed(bad):
    with pytest.raises(ValueError):
        LevelSpec.parse(bad)


def test_parent_follows_the_chain_by_default():
    assert LevelSpec("Block", 2).parent() == LevelSpec("Block", 1)
    assert LevelSpec("BlockGroup", 1).parent() == LevelSpec("BlockGroup", 0)
    with pytest.raises(ValueError):
        LevelSpec("Block", 0).parent()


def test_block_0_5_is_coarsened_from_block_0():
    spec = LevelSpec.parse("Block_0.5")
    assert spec.depth == 0.5
    assert str(spec) == "Block_0.5"
    assert spec.filename == "Block_0.5.pickle"
    assert spec.parent() == LevelSpec("Block", 0)
    assert LevelSpec("Block", 1).parent() == LevelSpec("Block", 0)
    assert LEVEL_NODE_TARGETS["Block"][1] < LEVEL_NODE_TARGETS["Block"][0.5]
    with pytest.raises(ValueError):
        spec.finer()


def test_integral_float_depth_is_the_integer_level():
    assert LevelSpec("Block", 1.0) == LevelSpec("Block", 1)
    assert str(LevelSpec.parse("Block_1.0")) == "Block_1"
