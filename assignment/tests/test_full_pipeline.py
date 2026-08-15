"""End-to-end smoke test of the alternative/baseline/selected estimates pipeline.

Runs the real tracked pipeline script (scripts/run_models_estimates.sh)
at tiny scale against the committed fake dataset in
tests/fixtures/fake_2223/:

    config generation -> run_custom_config.py (DA simulation)
    -> analyze_trends.py -> metrics_comparison.xlsx

Because every input lives in the repository (no /share/data, no real
student records), this test also proves that a fresh clone of the branch
is self-sufficient — see scripts/test_clean_checkout.sh.

Run: python -m pytest tests/test_full_pipeline.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SCRIPT = REPO_ROOT / "scripts" / "run_models_estimates.sh"
TEST_SETTINGS = REPO_ROOT / "scripts" / "settings" / "models_test.env"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "fake_2223"

# Run labels produced by the test settings (1 fake model x 1 k-value x
# in-sample year x 2 list-length variants + the status_quo_real reference).
EXPECTED_RUN_LABELS = [
    "status_quo_real_2223",
    "selectedfake_2223_k1_prog_gesplit_2223_ll0p8",
    "selectedfake_2223_k1_prog_gesplit_2223_ll7",
]
UTILITY_RUN_LABELS = EXPECTED_RUN_LABELS[1:]
# Test settings use ITER_START=0, ITER_END=2 -> iterations 0 and 1.
EXPECTED_ITERATIONS = [0, 1]
KEY_METRICS = [
    "Unassigned",
    "Distance Av (All Assigned)",
    "Prop Top 1 choice (All Assigned)",
    "Prop Top 3 choice (All Assigned)",
]


def _pipeline_env(tmp_path: Path) -> dict[str, str]:
    """Build the environment for a tiny pipeline run under tmp_path.

    Args:
        tmp_path (Path): Pytest-provided temporary directory.

    Returns:
        Dict[str, str]: Environment with all outputs redirected to
            tmp_path and the current interpreter as PYTHON_CMD.
    """
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_CMD": sys.executable,
            "RUNS_ROOT": str(tmp_path / "runs"),
            "CFG_DIR": str(tmp_path / "configs"),
            "ANALYSIS_CFG": str(tmp_path / "analysis.yaml"),
            "OUTPUT_DIR": str(tmp_path / "metrics"),
            "LOG_DIR": str(tmp_path / "logs"),
        }
    )
    return env


def _run_pipeline(env: dict[str, str], *extra_args: str):
    """Invoke the pipeline script with the test settings file.

    Args:
        env (Dict[str, str]): Environment for the subprocess.
        *extra_args (str): Additional CLI flags for the script.

    Returns:
        subprocess.CompletedProcess: The finished process.
    """
    return subprocess.run(
        ["bash", str(PIPELINE_SCRIPT), "--settings", str(TEST_SETTINGS)]
        + list(extra_args),
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_fixtures_present():
    """All committed fake-data files must exist."""
    expected_files = [
        FIXTURES_DIR / "student_2223_filtered.csv",
        FIXTURES_DIR / "programs_without_specialprogs_2223.csv",
        FIXTURES_DIR / "Cleaned" / "schools_rehauled_2223.csv",
        FIXTURES_DIR
        / "models"
        / "selectedfake_2223_k1_prog_gesplit"
        / "estimates_2223.csv",
        FIXTURES_DIR / "zones" / "concept1zones.csv",
    ]
    missing = [str(path) for path in expected_files if not path.is_file()]
    assert not missing, f"Missing fixture files: {missing}"


def test_dry_run(tmp_path):
    """The pipeline script must succeed in --dry-run mode."""
    result = _run_pipeline(_pipeline_env(tmp_path), "--dry-run")
    assert result.returncode == 0, result.stderr or result.stdout
    assert "[DRY-RUN]" in result.stdout


def test_generated_non_kg_grade_remains_a_string(tmp_path):
    env = _pipeline_env(tmp_path)
    env["GRADE"] = "06"

    result = _run_pipeline(env, "--no-simulate", "--no-analyze")

    assert result.returncode == 0, result.stderr or result.stdout
    config_path = tmp_path / "configs" / f"{EXPECTED_RUN_LABELS[0]}.yaml"
    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)
    assert config["grade"] == "06"


def test_full_pipeline_tiny(tmp_path):
    """Full pipeline on fake data: simulate, analyze, export Excel."""
    env = _pipeline_env(tmp_path)
    result = _run_pipeline(env)
    assert result.returncode == 0, result.stderr or result.stdout
    assert "simulation(s) failed" not in result.stdout, result.stdout

    # -- Step 1: one generated config per run label ------------------------
    for label in EXPECTED_RUN_LABELS:
        config_path = tmp_path / "configs" / f"{label}.yaml"
        assert config_path.is_file(), f"Missing generated config: {config_path}"

    # -- Step 2: assignment CSVs for every run and iteration ---------------
    for label in EXPECTED_RUN_LABELS:
        run_dir = tmp_path / "runs" / label
        assignment_csvs = sorted(run_dir.rglob("*_iteration*.csv"))
        found_iterations = {
            int(path.stem.rsplit("iteration", 1)[1]) for path in assignment_csvs
        }
        assert set(EXPECTED_ITERATIONS) <= found_iterations, (
            f"Run {label}: expected iterations {EXPECTED_ITERATIONS}, "
            f"found {sorted(found_iterations)}"
        )
        assignment_df = pd.read_csv(assignment_csvs[0])
        for column in ["studentno", "programno", "programcodes", "rank"]:
            assert column in assignment_df.columns

    # Utility-model runs save their drawn utility matrix.
    for label in UTILITY_RUN_LABELS:
        utility_path = tmp_path / "runs" / label / "utility_matrix.csv"
        assert utility_path.is_file(), f"Missing: {utility_path}"

    # -- Steps 3-5: Excel with the expected sheets and columns -------------
    excel_path = tmp_path / "metrics" / "metrics_comparison.xlsx"
    assert excel_path.is_file(), "analyze_trends did not produce the Excel"

    excel_file = pd.ExcelFile(excel_path)
    expected_sheets = {"Mean Values", "Std Values", "Mean ± Std", "2223"}
    assert expected_sheets <= set(excel_file.sheet_names), (
        f"Sheets found: {excel_file.sheet_names}"
    )

    df_mean = excel_file.parse("Mean Values", index_col=0)
    assert list(df_mean.columns) == EXPECTED_RUN_LABELS

    for metric in KEY_METRICS:
        assert metric in df_mean.index, f"Missing metric row: {metric}"
        values = df_mean.loc[metric].to_numpy(dtype=float)
        assert np.all(np.isfinite(values)), f"Non-finite values for {metric}"

    # Sanity: every fake student is matched or unassigned, never negative.
    assert bool((df_mean.loc["Unassigned"] >= 0).all())
    assert bool((df_mean.loc["Prop Top 1 choice (All Assigned)"] <= 1).all())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
