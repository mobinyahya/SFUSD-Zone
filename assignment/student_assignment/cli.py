"""CLI entry point for SFUSD matching commands."""

from pathlib import Path
from typing import Any

import click
import yaml
from loaders import anchor_data_config


@click.group()
def cli() -> None:
    """Run SFUSD matching commands."""


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Student-assignment YAML config to run.",
)
@click.option(
    "--assignments-dir",
    "--assignment-folder",
    "assignments_dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for raw assignment CSV outputs. Defaults to paths.assignment-folder.",
)
def simulate(config_path: Path, assignments_dir: Path | None) -> None:
    """Run a student-assignment simulation."""

    from .market_generator.school_choice_market_generator import MarketGenerator

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise click.ClickException(f"Config {config_path} must be a YAML mapping.")
    if isinstance(config.get("data"), dict):
        config["data"] = anchor_data_config(config["data"], config_path.parent)

    paths: dict[str, Any] = dict(config.get("paths") or {})
    if assignments_dir is None:
        assignment_folder = paths.get("assignment-folder")
        if not assignment_folder:
            raise click.ClickException(
                "Provide --assignments-dir or set paths.assignment-folder in the config."
            )
        assignments_dir = Path(str(assignment_folder)).expanduser()
    paths["assignment-folder"] = str(assignments_dir.resolve())
    config["paths"] = paths

    assignments_dir.mkdir(parents=True, exist_ok=True)
    market = MarketGenerator(
        config=config,
        assignment_path=str(assignments_dir),
    )
    market.simulate()


if __name__ == "__main__":
    cli()
