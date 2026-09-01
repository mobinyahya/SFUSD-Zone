"""@author: Edouard Rabasse
@date: 01-09-2026
This script allows passing a custom configuration YAML file AND overriding specific variables.
The CLI is: python run_custom_config.py --config-path config.yaml --sample sample001 --frac frac0.40.

Parallelism: use --workers N to simulate subconfigs concurrently. Each
subconfig is submitted independently so a failure does not prevent another
subconfig from running and can be attributed precisely. Each worker gets its
own in-memory configuration and numpy random state, so results are identical
to a sequential run (each subconfig resets np.random.seed internally).
"""

import copy
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import warnings
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
import yaml

# Support direct execution from assignment/ while importing top-level loaders/.
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from loaders import anchor_data_config

if __package__:
    from .student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )
else:
    from student_assignment.market_generator.school_choice_market_generator import (
        MarketGenerator,
    )

_WORKER_MARKET_GENERATOR = None


def _run_market_generator(market: MarketGenerator):
    result = market.simulate()
    if isinstance(result, Generator):
        market.execute_generator(result)
        return None
    return result


def resolve_variables(item, root_config):
    """Recursively replaces ${var} in strings using values from root_config."""
    if isinstance(item, dict):
        return {k: resolve_variables(v, root_config) for k, v in item.items()}
    elif isinstance(item, list):
        return [resolve_variables(v, root_config) for v in item]
    elif isinstance(item, str):
        pattern = re.compile(r"\$\{([^\}]+)\}")

        def replace(match):
            key = match.group(1)
            if key in root_config:
                return str(root_config[key])
            else:
                warnings.warn(f"Could not resolve variable ${{{key}}}", stacklevel=2)
                return match.group(0)

        return pattern.sub(replace, item)
    else:
        return item


def load_custom_config(
    config_path: str | pathlib.Path,
    *,
    sample: str | None = None,
    frac: str | None = None,
    assignment_folder: str | pathlib.Path | None = None,
    absolute_assignment_folder: bool = False,
) -> dict:
    """Load substitutions and anchor a custom assignment configuration."""
    config_path = pathlib.Path(config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config {config_path} must be a YAML mapping.")
    if sample:
        raw_config["sample"] = sample
    if frac:
        raw_config["frac"] = frac

    custom_config = resolve_variables(raw_config, raw_config)
    if isinstance(custom_config.get("data"), dict):
        custom_config["data"] = anchor_data_config(
            custom_config["data"], config_path.parent
        )
    if assignment_folder is not None:
        custom_config.setdefault("paths", {})["assignment-folder"] = str(
            assignment_folder
        )
    if absolute_assignment_folder:
        try:
            output = custom_config["paths"]["assignment-folder"]
        except KeyError as exc:
            raise ValueError(
                "Config must define paths.assignment-folder when saving assignments."
            ) from exc
        custom_config["paths"]["assignment-folder"] = str(
            pathlib.Path(output).expanduser().resolve()
        )
    return custom_config


def _write_provenance_config(
    custom_config: dict, *, clear_aggregate_metrics: bool = True
) -> None:
    """Write the complete resolved config once, before workers start."""
    if not custom_config.get("save-assignment", True):
        return

    try:
        assignment_folder = custom_config["paths"]["assignment-folder"]
    except KeyError as exc:
        raise ValueError(
            "Config must define paths.assignment-folder when saving assignments."
        ) from exc

    output_dir = pathlib.Path(assignment_folder).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir = output_dir / "aggregate_metrics"
    if clear_aggregate_metrics and aggregate_dir.exists():
        shutil.rmtree(aggregate_dir)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=".config.json.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        json.dump(custom_config, temp_file, indent=4)
        temp_path = pathlib.Path(temp_file.name)

    os.replace(temp_path, output_dir / "config.json")


def _run_subconfig_worker(
    custom_config: dict,
    subconfig_name: str,
    write_shared_utility_output: bool,
):
    """Run one subconfig in an isolated worker process.

    Must be a top-level function to be picklable by ProcessPoolExecutor.
    Each worker process has its own configuration and numpy random state, so
    results are deterministic and independent of execution order.

    Args:
        custom_config: Fully resolved base configuration dict.
        subconfig_name: Name of the subconfig to run (maps to a file in
            SUBCONFIGS_DIR).
        write_shared_utility_output: Whether this subconfig owns the shared
            utility-model save path. Exactly one parallel task owns it.
    """
    single_config = copy.deepcopy(custom_config)
    single_config["subconfigs"] = [subconfig_name]
    if not write_shared_utility_output:
        single_config.get("utility-model", {}).pop("save-path", None)

    global _WORKER_MARKET_GENERATOR
    if _WORKER_MARKET_GENERATOR is None:
        _WORKER_MARKET_GENERATOR = MarketGenerator(
            config=single_config,
            write_config=False,
            write_aggregate_metrics=False,
        )
    else:
        _WORKER_MARKET_GENERATOR.reconfigure(
            single_config,
            write_config=False,
        )
    return _run_market_generator(_WORKER_MARKET_GENERATOR)


@click.command()
@click.option(
    "--config-path",
    "--config_path",
    "config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the base configuration file.",
)
@click.option(
    "--sample",
    default=None,
    help="Override the sample variable (e.g., sample001)",
)
@click.option(
    "--frac", default=None, help="Override the frac variable (e.g., frac0.40)"
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    help="Number of parallel worker processes. Each subconfig runs in its own "
    "worker batch. workers=1 is the original sequential behaviour.",
)
def generate(config_path, sample, frac, workers):
    if workers < 1:
        raise click.BadParameter("--workers must be >= 1")

    custom_config = load_custom_config(
        config_path,
        sample=sample,
        frac=frac,
    )

    subconfigs_list = custom_config.get("subconfigs", [])
    duplicate_subconfigs = sorted(
        {name for name in subconfigs_list if subconfigs_list.count(name) > 1}
    )
    if duplicate_subconfigs:
        raise ValueError(
            f"Config contains duplicate subconfigs: {duplicate_subconfigs}."
        )

    # Workers must never race to replace the root provenance file. The parent
    # writes the complete resolved config, while every MarketGenerator below is
    # explicitly told not to write a per-worker variant.
    _write_provenance_config(custom_config)

    # 4. Run simulations — sequentially or in parallel
    if workers == 1 or not subconfigs_list:
        m = MarketGenerator(config=custom_config, write_config=False)
        _run_market_generator(m)
    else:
        worker_count = min(workers, len(subconfigs_list))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_subconfig_worker,
                    custom_config,
                    subconfig_name,
                    index == len(subconfigs_list) - 1,
                ): (index, subconfig_name)
                for index, subconfig_name in enumerate(subconfigs_list)
            }
            failures = []
            aggregate_reports = {}
            for future in as_completed(futures):
                index, subconfig_name = futures[future]
                try:
                    aggregate_reports[index] = future.result()
                except Exception as exc:
                    failures.append((index, subconfig_name, exc))

        if failures:
            failures.sort(key=lambda failure: failure[0])
            failed_names = [name for _, name, _ in failures]
            failure_details = "; ".join(
                f"{name}: {type(exc).__name__}: {exc}" for _, name, exc in failures
            )
            raise RuntimeError(
                f"{len(failures)} subconfig(s) failed: {failed_names}. "
                f"{failure_details}"
            )
        if custom_config.get("export-aggregate-metrics", False):
            reports = MarketGenerator.combine_aggregate_metric_reports(
                [aggregate_reports[index] for index in sorted(aggregate_reports)]
            )
            MarketGenerator.write_aggregate_metric_reports(
                custom_config["paths"]["assignment-folder"],
                reports,
            )


if __name__ == "__main__":
    generate()
