"""Shared pytest configuration.

Two session-level guarantees so the suite runs identically on any machine
(including a clean checkout with only git-tracked files):

1. The temp-file directories that the legacy unit tests
   (tests/utils_for_tests.py) write their generated CSVs into exist.
2. The Configerator loads a pristine user config generated from
   base_config.yaml + local_path_config.yaml, instead of the developer's
   personal configs/<user>.config.yaml (whose custom data paths would
   break the tests).
"""

import os
from pathlib import Path

import pytest

from assignment.student_assignment.definitions import CONFIGS_DIR

TEST_FILES_DIR = Path(__file__).parent / "test_files"
PYTEST_CONFIG_USER = "_pytest"
PYTEST_CONFIG_PATH = Path(CONFIGS_DIR) / f"{PYTEST_CONFIG_USER}.config.yaml"
CONFIG_USER_ENV = "SFUSD_ASSIGNMENT_CONFIG_USER"


@pytest.fixture(scope="session", autouse=True)
def ensure_test_file_dirs() -> None:
    """Create tests/test_files/Data/Cleaned before any test runs."""
    (TEST_FILES_DIR / "Data" / "Cleaned").mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session", autouse=True)
def hermetic_user_config():
    """Force Configerator to build a fresh pytest-only user config.

    The explicit config user makes every test session start from base_config +
    path config, regardless of the developer's personal config file.
    """
    original_user = os.environ.get(CONFIG_USER_ENV)
    os.environ[CONFIG_USER_ENV] = PYTEST_CONFIG_USER
    if PYTEST_CONFIG_PATH.exists():
        PYTEST_CONFIG_PATH.unlink()
    yield
    if original_user is None:
        os.environ.pop(CONFIG_USER_ENV, None)
    else:
        os.environ[CONFIG_USER_ENV] = original_user
    PYTEST_CONFIG_PATH.unlink(missing_ok=True)
