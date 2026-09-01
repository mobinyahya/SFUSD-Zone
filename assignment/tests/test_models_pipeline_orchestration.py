import os
import subprocess
import sys
from pathlib import Path

import yaml


ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ASSIGNMENT_ROOT / "scripts" / "run_models_estimates.sh"


def _write_settings(
    tmp_path: Path,
    *,
    test_specs: list[str],
    model_families: list[str] | None = None,
    variants: list[str] | None = None,
    workers: int = 2,
    iter_end: int = 1,
) -> tuple[Path, dict[str, Path]]:
    paths = {
        "runs": tmp_path / "runs",
        "configs": tmp_path / "configs",
        "analysis": tmp_path / "analysis.yaml",
        "output": tmp_path / "metrics",
        "logs": tmp_path / "logs",
        "students": tmp_path / "students",
        "programs": tmp_path / "programs",
        "schools": tmp_path / "schools",
        "models": tmp_path / "models",
        "zones": tmp_path / "zones",
    }
    families = " ".join(model_families or [])
    variant_values = " ".join(f'"{variant}"' for variant in (variants or []))
    spec_values = " ".join(f'"{spec}"' for spec in test_specs)
    settings = tmp_path / "settings.env"
    settings.write_text(
        f''': "${{PYTHON_CMD:={sys.executable}}}"
SFUSD_MODELS_DIR="{paths["models"]}"
STUDENT_DIR="{paths["students"]}"
PROGRAM_DIR="{paths["programs"]}"
SCHOOL_DATA_DIR="{paths["schools"]}"
SFUSD_DATA_DIR="{tmp_path / "sfusd"}"
ZONES_DIR="{paths["zones"]}"
RUNS_ROOT="{paths["runs"]}"
CFG_DIR="{paths["configs"]}"
ANALYSIS_CFG="{paths["analysis"]}"
OUTPUT_DIR="{paths["output"]}"
LOG_DIR="{paths["logs"]}"
ANALYSIS_NEW_CTIP_PATH=""
GRADE="KG"
RANDOM_SEED="2023"
ITER_START="0"
ITER_END="{iter_end}"
SIMULATION_WORKERS="{workers}"
TRAIN_YEAR="2223"
MODEL_SUFFIX="suffix"
MODEL_FAMILIES=({families})
K_VALUES=(1)
TEST_SPECS=({spec_values})
LIST_LENGTH_VARIANTS=({variant_values})
'''
    )
    return settings, paths


def _assignment_files(
    runs_root: Path,
    label: str,
    subconfig: str,
    iterations: list[int],
) -> None:
    output = runs_root / label / subconfig / "policy"
    output.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        (output / f"assignment_iteration{iteration}.csv").write_text("studentno\n")


def _run(settings: Path, *args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(PIPELINE), "--settings", str(settings), *args],
        cwd=ASSIGNMENT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_skip_existing_requires_every_expected_iteration(tmp_path):
    settings, paths = _write_settings(tmp_path, test_specs=["2223:22:is"], iter_end=2)
    run_dir = paths["runs"] / "status_quo_real_2223" / "status_quo_real"
    run_dir.mkdir(parents=True)
    (run_dir / "unrelated.txt").write_text("partial")

    incomplete = _run(settings, "--no-generate", "--no-analyze", "--skip-existing")
    assert incomplete.returncode != 0
    assert "simulation config(s) are missing" in incomplete.stdout

    _assignment_files(paths["runs"], "status_quo_real_2223", "status_quo_real", [0])
    still_incomplete = _run(
        settings, "--no-generate", "--no-analyze", "--skip-existing"
    )
    assert still_incomplete.returncode != 0

    _assignment_files(paths["runs"], "status_quo_real_2223", "status_quo_real", [1])
    complete = _run(settings, "--no-generate", "--no-analyze", "--skip-existing")
    assert complete.returncode == 0, complete.stderr or complete.stdout
    assert "SKIP (already done): status_quo_real_2223" in complete.stdout


def test_simulation_failures_keep_labels_after_skips_and_respect_bound(tmp_path):
    variants = ["bad:1", "good1:1", "good2:1"]
    settings, paths = _write_settings(
        tmp_path,
        test_specs=["2223:22:is"],
        model_families=["fake"],
        variants=variants,
        workers=2,
    )
    _assignment_files(paths["runs"], "status_quo_real_2223", "status_quo_real", [0])

    labels = [
        "fake_2223_k1_suffix_2223_bad",
        "fake_2223_k1_suffix_2223_good1",
        "fake_2223_k1_suffix_2223_good2",
    ]
    paths["configs"].mkdir(parents=True)
    for label in labels:
        (paths["configs"] / f"{label}.yaml").write_text("subconfigs: []\n")

    active_dir = tmp_path / "active"
    active_dir.mkdir()
    observations = tmp_path / "observations.txt"
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        """import os
import pathlib
import sys
import time

config_path = pathlib.Path(sys.argv[sys.argv.index("--config-path") + 1])
label = config_path.stem
active_dir = pathlib.Path(os.environ["ACTIVE_DIR"])
marker = active_dir / str(os.getpid())
marker.write_text(label)
with pathlib.Path(os.environ["OBSERVATIONS"]).open("a") as output:
    output.write(f"{len(list(active_dir.iterdir()))}\\n")
time.sleep(0.4)
try:
    if label.endswith("_bad"):
        raise SystemExit(9)
    output_dir = pathlib.Path(os.environ["RUNS_ROOT"]) / label / "status_quo" / "policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "assignment_iteration0.csv").write_text("studentno\\n")
finally:
    marker.unlink(missing_ok=True)
"""
    )
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_CMD": f"{sys.executable} {runner}",
            "RUNS_ROOT": str(paths["runs"]),
            "ACTIVE_DIR": str(active_dir),
            "OBSERVATIONS": str(observations),
        }
    )

    result = _run(
        settings,
        "--no-generate",
        "--no-analyze",
        "--skip-existing",
        env=env,
    )

    assert result.returncode != 0
    failure_summary = result.stdout.split("simulation(s) failed", 1)[1]
    assert f"- {labels[0]}" in failure_summary
    assert "- status_quo_real_2223" not in failure_summary
    observed = [int(value) for value in observations.read_text().splitlines()]
    assert max(observed) == 2


def test_analysis_failure_is_nonzero_and_config_uses_year_school_data(tmp_path):
    settings, paths = _write_settings(
        tmp_path, test_specs=["2223:22:is", "2324:23:oos"]
    )
    for year in ["2223", "2324"]:
        _assignment_files(
            paths["runs"],
            f"status_quo_real_{year}",
            "status_quo_real",
            [0],
        )

    failing_runner = tmp_path / "fail_analysis.py"
    failing_runner.write_text("raise SystemExit(7)\n")
    env = os.environ.copy()
    env["PYTHON_CMD"] = f"{sys.executable} {failing_runner}"

    result = _run(settings, "--no-generate", "--no-simulate", env=env)

    assert result.returncode != 0
    assert "analyze_trends failed" in result.stdout
    analysis = yaml.safe_load(paths["analysis"].read_text())
    by_label = {run["label"]: run for run in analysis["runs"]}
    for year in ["2223", "2324"]:
        assert by_label[f"status_quo_real_{year}"]["schools_data"] == str(
            paths["schools"] / f"schools_rehauled_{year}.csv"
        )
    assert "schools_data" not in {
        key: value for key, value in analysis.items() if key != "runs"
    }
