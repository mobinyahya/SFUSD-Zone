from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from loaders import load_scenario


@pytest.fixture
def data_config_factory(tmp_path: Path):
    def factory(
        *,
        sources: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        cache_root: Path | None = None,
        data_root: Path | None = None,
    ) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "roots": {"cache": str(cache_root or tmp_path / "cache")}
        }
        if data_root is not None:
            overrides["roots"]["data"] = str(data_root)
        if sources:
            overrides["sources"] = sources
        if filters:
            overrides["filters"] = filters
        return {"scenario": "legacy", "overrides": overrides}

    return factory


@pytest.fixture
def scenario_factory(data_config_factory):
    def factory(**kwargs):
        return load_scenario(data_config_factory(**kwargs), environ={})

    return factory
