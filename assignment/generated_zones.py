"""Run assignment policy subconfigs against a generated zoning plan."""

from __future__ import annotations

import copy
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

from loaders import load_scenario

from .run_custom_config import (
    _run_market_generator,
    _write_provenance_config,
    load_custom_config,
)
from .student_assignment.configerator import Configerator
from .student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


GENERATED_ZONE_POLICY = "generated-zones"
GENERATED_ZONE_FILENAME = "assignment_zones.csv"
SKIP_MARKER_FILENAME = ".assignment-skipped"


def write_generated_zones(
    area_assignment: Mapping[int, int], path: str | Path
) -> dict[int, int]:
    """Write assignment's row-per-zone CSV from ``{area_id: zone_id}``."""
    if not area_assignment:
        raise ValueError("Cannot write generated zones from an empty assignment.")

    zones: dict[int, list[int]] = {}
    for area_id, zone_id in area_assignment.items():
        zones.setdefault(int(zone_id), []).append(int(area_id))

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    zone_id_map = {}
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for row_index, zone_id in enumerate(sorted(zones)):
            zone_id_map[zone_id] = row_index
            writer.writerow(sorted(zones[zone_id]))
    return zone_id_map


def resolve_generated_zone_configs(
    config_path: str | Path,
    *,
    zone_file: str | Path,
    assignment_folder: str | Path,
    zone_building_blocks: str,
    geography_vintage: str | None = None,
) -> tuple[dict, list[dict]]:
    """Resolve every policy subconfig and inject one generated zone source."""
    base = load_custom_config(
        config_path,
        assignment_folder=assignment_folder,
        absolute_assignment_folder=True,
    )
    return resolve_generated_zone_config(
        base,
        zone_file=zone_file,
        assignment_folder=assignment_folder,
        zone_building_blocks=zone_building_blocks,
        geography_vintage=geography_vintage,
    )


def resolve_generated_zone_batch_configs(
    config_path: str | Path,
    targets: list[dict],
    *,
    assignment_folder: str | Path,
) -> tuple[dict, list[dict]]:
    """Resolve the target-by-policy product into one assignment batch."""
    base = load_custom_config(
        config_path,
        assignment_folder=assignment_folder,
        absolute_assignment_folder=True,
    )
    return resolve_generated_zone_batch_config(
        base,
        targets,
        assignment_folder=assignment_folder,
    )


def resolve_generated_zone_batch_config(
    base_config: dict,
    targets: list[dict],
    *,
    assignment_folder: str | Path,
) -> tuple[dict, list[dict]]:
    """Inject every generated zone target as a unique policy subconfig."""
    if not targets:
        raise ValueError("Generated-zone assignment requires at least one target.")
    target_ids = [str(target["id"]) for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Generated-zone target IDs must be unique.")

    output_path = Path(assignment_folder).expanduser().resolve()
    base = copy.deepcopy(base_config)
    base.setdefault("paths", {})["assignment-folder"] = str(output_path)

    scenario = load_scenario(base["data"])
    assignment_vintage = scenario.filter("assignment", "geography_vintage")
    for target in targets:
        geography_vintage = target.get("geography_vintage")
        if geography_vintage is not None and geography_vintage != assignment_vintage:
            raise ValueError(
                "Optimization and assignment geography vintages differ: "
                f"{geography_vintage!r} != {assignment_vintage!r}."
            )

    configurator = Configerator.from_config(base)
    policy_names = list(configurator.config.get("subconfigs", []))
    if not policy_names:
        raise ValueError("Generated-zone assignment requires at least one subconfig.")
    templates = []
    for policy_name in policy_names:
        configurator.load_subconfig_by_name(policy_name)
        config = copy.deepcopy(configurator.config)
        config["subconfigs"] = []
        config["paths"]["assignment-folder"] = str(output_path)
        templates.append({"name": policy_name, "config": config})

    resolved = []
    for target, target_id in zip(targets, target_ids, strict=True):
        zone_path = Path(target["zone_file"]).expanduser().resolve()
        for template in templates:
            config = copy.deepcopy(template["config"])
            _inject_generated_zone(
                config, zone_path, str(target["zone_building_blocks"])
            )
            policy_name = template["name"]
            name = f"{target_id}:{policy_name}"
            config["subconfig-name"] = name
            resolved.append(
                {
                    "name": name,
                    "policy": policy_name,
                    "target": target_id,
                    "config": config,
                }
            )

    base["subconfigs"] = [entry["name"] for entry in resolved]
    return base, resolved


def resolve_generated_zone_config(
    base_config: dict,
    *,
    zone_file: str | Path,
    assignment_folder: str | Path,
    zone_building_blocks: str,
    geography_vintage: str | None = None,
) -> tuple[dict, list[dict]]:
    """Resolve generated-zone subconfigs from an already anchored base config."""
    zone_path = Path(zone_file).expanduser().resolve()
    output_path = Path(assignment_folder).expanduser().resolve()
    base = copy.deepcopy(base_config)
    base.setdefault("paths", {})["assignment-folder"] = str(output_path)
    _inject_generated_zone(base, zone_path, zone_building_blocks)

    scenario = load_scenario(base["data"])
    assignment_vintage = scenario.filter("assignment", "geography_vintage")
    if geography_vintage is not None and assignment_vintage != geography_vintage:
        raise ValueError(
            "Optimization and assignment geography vintages differ: "
            f"{geography_vintage!r} != {assignment_vintage!r}."
        )

    configurator = Configerator.from_config(base)
    names = list(configurator.config.get("subconfigs", []))
    if not names:
        raise ValueError("Generated-zone assignment requires at least one subconfig.")

    resolved = []
    for name in names:
        configurator.load_subconfig_by_name(name)
        config = copy.deepcopy(configurator.config)
        config["subconfigs"] = []
        config["paths"]["assignment-folder"] = str(output_path)
        _inject_generated_zone(config, zone_path, zone_building_blocks)
        resolved.append({"name": name, "config": config})
    return base, resolved


def run_generated_zone_assignment(
    config_path: str | Path,
    area_assignment: Mapping[int, int],
    *,
    assignment_folder: str | Path,
    zone_building_blocks: str,
    geography_vintage: str | None = None,
    workers: int = 1,
) -> None:
    """Run all assignment subconfigs and assignment-owned metrics."""
    output_path = Path(assignment_folder).expanduser().resolve()
    zone_file = output_path / GENERATED_ZONE_FILENAME
    write_generated_zones(area_assignment, zone_file)
    base, resolved = resolve_generated_zone_configs(
        config_path,
        zone_file=zone_file,
        assignment_folder=output_path,
        zone_building_blocks=zone_building_blocks,
        geography_vintage=geography_vintage,
    )
    _run_generated_zone_configs(base, resolved, output_path, workers)


def run_generated_zone_assignments(
    config_path: str | Path,
    targets: list[dict],
    *,
    assignment_folder: str | Path,
    workers: int = 1,
) -> None:
    """Run all generated zone targets as one root-level assignment batch."""
    output_path = Path(assignment_folder).expanduser().resolve()
    base, resolved = resolve_generated_zone_batch_configs(
        config_path,
        targets,
        assignment_folder=output_path,
    )
    _run_generated_zone_configs(base, resolved, output_path, workers)


def _run_generated_zone_configs(
    base: dict,
    resolved: list[dict],
    output_path: Path,
    workers: int,
) -> None:
    _write_provenance_config(base)
    reports = _run_resolved_configs(resolved, workers=max(1, int(workers)))
    if base.get("export-aggregate-metrics", False):
        combined = MarketGenerator.combine_aggregate_metric_reports(reports)
        MarketGenerator.write_aggregate_metric_reports(output_path, combined)


def _inject_generated_zone(
    config: dict, zone_file: Path, zone_building_blocks: str
) -> None:
    sources = (
        config.setdefault("data", {})
        .setdefault("overrides", {})
        .setdefault("sources", {})
    )
    sources.setdefault("assignment.zones", {})[GENERATED_ZONE_POLICY] = str(zone_file)
    config["policies"] = [GENERATED_ZONE_POLICY]
    config["zone-building-blocks"] = zone_building_blocks
    config["reuse_assignments"] = config.get("reuse_assignments", True)


def _run_resolved_config(config: dict):
    market = MarketGenerator(
        config=config,
        assignment_path=config["paths"]["assignment-folder"],
        write_config=False,
        write_aggregate_metrics=False,
    )
    return _run_market_generator(market)


def _run_resolved_configs(resolved: list[dict], workers: int) -> list[dict]:
    configs = [copy.deepcopy(entry["config"]) for entry in resolved]
    utility_owners = [
        index
        for index, config in enumerate(configs)
        if config.get("utility-model", {}).get("save-path")
    ]
    for index in utility_owners[:-1]:
        configs[index]["utility-model"].pop("save-path", None)

    if workers == 1 or len(configs) == 1:
        reports = []
        market = None
        for config in configs:
            if market is None:
                market = MarketGenerator(
                    config=config,
                    assignment_path=config["paths"]["assignment-folder"],
                    write_config=False,
                    write_aggregate_metrics=False,
                )
            else:
                market.reconfigure(
                    config,
                    config["paths"]["assignment-folder"],
                    write_config=False,
                )
            report = _run_market_generator(market)
            if report is not None:
                reports.append(report)
        return reports

    reports_by_index = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(configs))) as executor:
        futures = {
            executor.submit(_run_resolved_config, config): index
            for index, config in enumerate(configs)
        }
        for future in as_completed(futures):
            report = future.result()
            if report is not None:
                reports_by_index[futures[future]] = report
    return [reports_by_index[index] for index in sorted(reports_by_index)]
