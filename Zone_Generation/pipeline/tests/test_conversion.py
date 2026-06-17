from Zone_Generation.pipeline.data.conversion import (
    LevelConverter,
    base_area_assignment,
)
from Zone_Generation.pipeline.levels import LevelSpec
from Zone_Generation.pipeline.tests.synthetic import make_path_graphs

BG0 = LevelSpec("BlockGroup", 0)
BG1 = LevelSpec("BlockGroup", 1)


def test_base_area_assignment_expands_blocks():
    _, coarse = make_path_graphs()
    area = base_area_assignment(coarse, {0: 0, 1: 1})
    assert area == {10: 0, 11: 0, 12: 1, 13: 1}


def test_fine_to_coarse():
    base, coarse = make_path_graphs()
    conv = LevelConverter()
    fine_assignment = {0: 0, 1: 0, 2: 1, 3: 1}
    coarse_assignment = conv.between(base, fine_assignment, BG0, coarse, BG1)
    assert coarse_assignment == {0: 0, 1: 1}


def test_coarse_to_fine_roundtrip():
    base, coarse = make_path_graphs()
    conv = LevelConverter()
    coarse_assignment = {0: 0, 1: 1}
    fine = conv.between(coarse, coarse_assignment, BG1, base, BG0)
    assert fine == {0: 0, 1: 0, 2: 1, 3: 1}
    # round-trip back to coarse
    back = conv.between(base, fine, BG0, coarse, BG1)
    assert back == coarse_assignment
