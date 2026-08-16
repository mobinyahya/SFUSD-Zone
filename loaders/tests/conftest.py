from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from loaders.config import DataScenario, load_scenario


@pytest.fixture
def scenario_factory(tmp_path: Path):
    def factory(
        sources: dict[str, Any],
        filters: dict[str, Any] | None = None,
        *,
        scenario_id: str = "synthetic",
        cache_root: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> DataScenario:
        normalized_filters: dict[str, Any] = {}
        defaults = {
            "optimization": {
                "years": ["1819"],
                "grades": ["KG"],
                "student_population": "enrolled",
                "rounds": "all",
                "special_programs": "include",
                "program_population": "GE",
                "capacity_scenario": "programs",
                "include_k8": False,
                "include_citywide": False,
                "include_mission_bay": False,
                "geography_vintage": "2010",
            },
            "assignment": {
                "year": "1819",
                "grades": ["KG"],
                "student_population": "applicant",
                "rounds": "all",
                "special_programs": "include",
                "capacity_profile": "default",
                "capacity_scenario": "programs",
                "include_mission_bay": False,
                "geography_vintage": "2010",
            },
        }
        for group, values in (filters or {}).items():
            normalized_filters[group] = {**defaults[group], **values}
        scenario_path = tmp_path / f"{scenario_id}.yaml"
        scenario_path.write_text(
            yaml.safe_dump(
                {
                    "id": scenario_id,
                    "sources": sources,
                    "filters": normalized_filters,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        run_overrides: dict[str, Any] = {
            "roots": {"cache": str(cache_root or tmp_path / "cache")},
            "sources": sources,
        }
        if overrides:
            for key, value in overrides.items():
                if key == "roots":
                    run_overrides["roots"].update(value)
                else:
                    run_overrides[key] = value
        return load_scenario(
            {"scenario": str(scenario_path), "overrides": run_overrides},
            environ={},
        )

    return factory
