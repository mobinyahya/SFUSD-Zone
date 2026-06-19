from click.testing import CliRunner

from sfusd_matching import DeferredAcceptance
from sfusd_matching.cli import cli


def test_deferred_acceptance_wrapper_runs_match():
    matcher = DeferredAcceptance(
        school_caps=[1, 1],
        student_priorities=[
            [2, 1],
            [1, 2],
        ],
        student_prefs=[
            [1, 2],
            [1, 2],
        ],
    )

    student_match, lowest_priority, student_proposal = matcher.run()

    assert student_match.tolist() == [1, 2]
    assert lowest_priority.tolist() == [2, 2]
    assert student_proposal.tolist() == [1, 2]


def test_sfusd_matching_cli_help():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "simulate" in result.output
