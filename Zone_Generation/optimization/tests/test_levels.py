import pytest

from Zone_Generation.pipeline.levels import LevelSpec


@pytest.mark.parametrize(
    "text,unit,depth",
    [
        ("BlockGroup_0", "BlockGroup", 0),
        ("Block_2", "Block", 2),
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


@pytest.mark.parametrize("bad", ["BlockGroup", "Block_x", "Foo_0", ""])
def test_malformed(bad):
    with pytest.raises(ValueError):
        LevelSpec.parse(bad)
