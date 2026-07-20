"""@author: Edouard Rabasse
@date: 01-09-2026
This script allows passing a custom configuration YAML file AND overriding specific variables.
The CLI is: python run_custom_config.py --config-path config.yaml --sample sample001 --frac frac0.40.

Parallelism: use --workers N to simulate subconfigs concurrently. Each worker
process handles a batch of subconfigs so market data is loaded once per batch.
Each worker gets its own Configerator singleton and numpy random state, so
results are identical to a sequential run (each subconfig resets np.random.seed
internally).
"""

import os
import re
import sys
import warnings
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
import yaml

# Ensure we can import your project modules
sys.path.append(os.getcwd())

from student_assignment.configerator import Configerator
from student_assignment.market_generator.school_choice_market_generator import (
    MarketGenerator,
)


def _run_market_generator(market: MarketGenerator) -> None:
    result = market.simulate()
    if isinstance(result, Generator):
        market.execute_generator(result)


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
                warnings.warn(
                    f"Could not resolve variable ${{{key}}}", stacklevel=2
                )
                return match.group(0)

        return pattern.sub(replace, item)
    else:
        return item


def _chunk_subconfigs(subconfigs: list[str], workers: int) -> list[list[str]]:
    if not subconfigs:
        return []
    worker_count = min(workers, len(subconfigs))
    base_size, extra = divmod(len(subconfigs), worker_count)
    chunks = []
    start = 0
    for i in range(worker_count):
        end = start + base_size + int(i < extra)
        chunks.append(subconfigs[start:end])
        start = end
    return chunks


def _run_subconfig_worker(
    custom_config: dict, subconfig_names: list[str]
) -> None:
    """Run a batch of subconfigs in an isolated worker process.

    Must be a top-level function to be picklable by ProcessPoolExecutor.
    Each worker process has its own Configerator singleton and numpy random
    state, so results are deterministic and independent of execution order.

    Args:
        custom_config: Fully resolved base configuration dict.
        subconfig_names: Names of the subconfigs to run (map to files in
            SUBCONFIGS_DIR).
    """
    # Reset the singleton so this process starts from a clean slate.
    # Required when the parent process used fork and inherited an instance.
    Configerator.instance = None

    single_config = {**custom_config, "subconfigs": subconfig_names}

    c = Configerator()
    c._config = single_config
    c._original_config = single_config
    c.subconfigs = iter(subconfig_names)

    m = MarketGenerator()
    _run_market_generator(m)


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

    # 1. Load the raw YAML
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # 2. APPLY OVERRIDES HERE
    if sample:
        raw_config["sample"] = sample
    if frac:
        raw_config["frac"] = frac

    # 3. Resolve variables using the updated raw_config
    custom_config = resolve_variables(raw_config, raw_config)

    subconfigs_list = custom_config.get("subconfigs", [])

    # 4. Run simulations — sequentially or in parallel
    if workers == 1:
        c = Configerator()
        c._config = custom_config
        c._original_config = custom_config
        c.subconfigs = iter(subconfigs_list)

        m = MarketGenerator()
        _run_market_generator(m)
    else:
        subconfig_chunks = _chunk_subconfigs(subconfigs_list, workers)
        if not subconfig_chunks:
            return
        with ProcessPoolExecutor(max_workers=len(subconfig_chunks)) as executor:
            futures = {
                executor.submit(
                    _run_subconfig_worker, custom_config, subconfig_chunk
                ): subconfig_chunk
                for subconfig_chunk in subconfig_chunks
            }
            failed = []
            failure_details = []
            for future in as_completed(futures):
                subconfig_chunk = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed.extend(subconfig_chunk)
                    failure_details.append(f"{subconfig_chunk}: {exc}")

        if failed:
            raise RuntimeError(
                f"{len(failed)} subconfig(s) failed: {failed}. "
                + "; ".join(failure_details)
            )


if __name__ == "__main__":
    generate()
