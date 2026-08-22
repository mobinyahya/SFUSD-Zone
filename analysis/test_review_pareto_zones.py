from pathlib import Path

import pandas as pd

from analysis import review_pareto_zones as review


def _row(task_id: str, config_name: str, value: float = 1.0) -> dict:
    row = {
        "task_id": task_id,
        "config_name": config_name,
        "path": f"/runs/{task_id}",
    }
    for metric, minimize in review.OBJECTIVES.items():
        row[metric] = value if minimize else -value
    return row


def test_pareto_front_respects_minimize_and_maximize_directions():
    best = _row("best", "best-root:policy/variant", 1.0)
    dominated = _row("dominated", "dominated-root:policy/variant", 2.0)
    tradeoff = _row("tradeoff", "tradeoff-root:policy/variant", 1.0)
    tradeoff["normalized_cut_edges"] = 0.5
    tradeoff["avg_polsby_popper_score"] = -2.0

    frontier = review.pareto_front(pd.DataFrame([best, dominated, tradeoff]))

    assert set(frontier["task_id"]) == {"best", "tradeoff"}


def test_join_observations_supports_multiple_assignment_rows_per_zone():
    zones = pd.DataFrame(
        [
            {
                "task_id": "abc",
                "path": "/runs/abc",
                **{metric: 1.0 for metric in review.ZONE_METRICS},
            }
        ]
    )
    assignments = pd.DataFrame(
        [
            {
                "config_name": f"abc-root:{policy}/variant",
                **{metric: 1.0 for metric in review.ASSIGNMENT_METRICS},
            }
            for policy in ("one", "two")
        ]
    )

    joined = review.join_observations(zones, assignments, known_task_ids={"abc"})

    assert joined["task_id"].tolist() == ["abc", "abc"]
    assert joined["config_name"].tolist() == [
        "abc-root:one/variant",
        "abc-root:two/variant",
    ]


def test_write_outputs_keeps_only_approved_rows_and_visualizations(tmp_path: Path):
    frontier = pd.DataFrame(
        [
            _row("yes", "yes-root:policy/variant"),
            _row("no", "no-root:policy/variant"),
        ]
    )
    source_yes = tmp_path / "yes-source.png"
    source_no = tmp_path / "no-source.png"
    source_yes.write_bytes(b"yes")
    source_no.write_bytes(b"no")
    output = tmp_path / "analysis"
    stale = output / "viz" / "stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    approved = review.write_outputs(
        frontier,
        {"yes": True, "no": False},
        {"yes": source_yes, "no": source_no},
        output,
    )

    assert approved["task_id"].tolist() == ["yes"]
    exported = pd.read_csv(output / "pareto.csv")
    assert exported.columns.tolist() == review.OUTPUT_COLUMNS
    assert exported["task_id"].tolist() == ["yes"]
    assert (output / "viz" / "yes.png").read_bytes() == b"yes"
    assert not (output / "viz" / "no.png").exists()
    assert not stale.exists()
