#!/usr/bin/env python3
"""Find, review, and export Pareto-optimal benchmark zone plans."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment.run_custom_config import load_custom_config  # noqa: E402
from benchmark.config import SimulationSweep, VisualizationRunConfig  # noqa: E402
from benchmark.results import discover_run_dirs, load_run_result  # noqa: E402
from benchmark.runner import (  # noqa: E402
    MANIFEST_FILENAME,
    load_manifest,
    load_solutions,
    write_json,
)
from benchmark.visualize import render_task_visualizations  # noqa: E402


# True means minimize; False means maximize.
OBJECTIVES = {
    "normalized_cut_edges": True,
    "avg_polsby_popper_score": False,
    "avg_reock_score": False,
    "#Schools above 10% district FRL": True,
    "#Schools above 15% district FRL": True,
    "Dissimilarity (High FRL)": True,
    "Dissimilarity (White)": True,
    "# Racial majority schools": True,
    "Unassigned": True,
    "Designated": True,
    "Prop Top 1 choice (All Students)": False,
    "Prop Top 3 choice (All Students)": False,
}
ZONE_METRICS = list(OBJECTIVES)[:3]
ASSIGNMENT_METRICS = list(OBJECTIVES)[3:]
OUTPUT_COLUMNS = ["task_id", "config_name", "path", *OBJECTIVES]
FEASIBLE_STATUSES = {"FEASIBLE", "OPTIMAL"}
DECISIONS_FILENAME = "pareto_reviews.json"


def pareto_front(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows not dominated across the configured objectives."""
    _require_columns(frame, OBJECTIVES, "Pareto observations")
    values = frame[list(OBJECTIVES)].apply(pd.to_numeric, errors="coerce")
    invalid = values.isna().any(axis=1)
    if invalid.any():
        names = frame.loc[invalid, "config_name"].astype(str).head(5).tolist()
        raise ValueError(f"Pareto observations have missing metrics: {names}")

    costs = values.to_numpy(dtype=float, copy=True)
    maximize = [not minimize for minimize in OBJECTIVES.values()]
    costs[:, maximize] *= -1
    dominated = np.zeros(len(costs), dtype=bool)
    for index, candidate in enumerate(costs):
        not_worse = (costs <= candidate).all(axis=1)
        better = (costs < candidate).any(axis=1)
        dominated[index] = bool((not_worse & better).any())

    return (
        frame.loc[~dominated, OUTPUT_COLUMNS]
        .sort_values(["normalized_cut_edges", "task_id", "config_name"])
        .reset_index(drop=True)
    )


def load_observations(config_path: str | Path) -> tuple[pd.DataFrame, Path, str | None]:
    """Load and join optimization and root-level assignment metrics."""
    sweep = SimulationSweep.from_yaml(str(config_path))
    if not sweep.matching.enabled or not sweep.matching.config:
        raise ValueError("The benchmark config must have matching.enabled: true.")
    assignment_config = load_custom_config(sweep.matching.config)
    if assignment_config.get("export-aggregate-metrics") is not True:
        raise ValueError(
            "The matching config must have export-aggregate-metrics: true."
        )

    root = Path(sweep.execution.output_dir).expanduser().resolve()
    zone_metrics, known_task_ids = _load_zone_metrics(root)
    assignment_metrics = _load_assignment_metrics(root)
    observations = join_observations(
        zone_metrics, assignment_metrics, known_task_ids=known_task_ids
    )
    return observations, root, sweep.visualization.artifact_dir


def _load_zone_metrics(root: Path) -> tuple[pd.DataFrame, set[str]]:
    run_dirs = discover_run_dirs(str(root))
    if not run_dirs:
        raise FileNotFoundError(f"No benchmark runs found under {root}.")

    rows = []
    known_task_ids = set()
    for run_dir in run_dirs:
        manifest = load_manifest(run_dir)
        task_id = str(manifest.get("task_id") or "")
        if not task_id:
            raise ValueError(f"Benchmark manifest has no task_id: {run_dir}")
        if task_id in known_task_ids:
            raise ValueError(f"Duplicate benchmark task_id: {task_id}")
        known_task_ids.add(task_id)

        result = load_run_result(run_dir)
        status = str(result.get("status") or manifest.get("status") or "").upper()
        if status not in FEASIBLE_STATUSES:
            continue
        metrics = result.get("metrics") or {}
        missing = [metric for metric in ZONE_METRICS if metric not in metrics]
        if missing:
            raise ValueError(f"Run {task_id} is missing zone metrics: {missing}")
        rows.append(
            {
                "task_id": task_id,
                "path": str(Path(run_dir).resolve()),
                **{metric: metrics[metric] for metric in ZONE_METRICS},
            }
        )

    if not rows:
        raise ValueError("The benchmark has no feasible runs to compare.")
    frame = pd.DataFrame(rows)
    _require_numeric(frame, ZONE_METRICS, "Zone metrics")
    return frame, known_task_ids


def _load_assignment_metrics(root: Path) -> pd.DataFrame:
    candidates = [
        root / "aggregate_metrics" / "metrics_citywide.csv",
        root / "assignments" / "aggregate_metrics" / "metrics_citywide.csv",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            "Could not find matching aggregate metrics at "
            + " or ".join(str(path) for path in candidates)
        )
    if len(existing) > 1:
        raise ValueError(f"Multiple matching aggregate metric files found: {existing}")

    try:
        frame = pd.read_csv(existing[0], usecols=["config_name", *ASSIGNMENT_METRICS])
    except ValueError as exc:
        raise ValueError(
            f"Assignment metrics are missing required columns in {existing[0]}: {exc}"
        ) from exc
    if frame["config_name"].duplicated().any():
        duplicates = frame.loc[
            frame["config_name"].duplicated(keep=False), "config_name"
        ].head(5)
        raise ValueError(
            f"Assignment metrics contain duplicate config_name rows: {duplicates.tolist()}"
        )
    _require_numeric(frame, ASSIGNMENT_METRICS, "Assignment metrics")
    return frame


def join_observations(
    zone_metrics: pd.DataFrame,
    assignment_metrics: pd.DataFrame,
    *,
    known_task_ids: set[str],
) -> pd.DataFrame:
    """Join generated-zone assignment rows to their benchmark task."""
    assignment = assignment_metrics.copy()
    assignment["task_id"] = (
        assignment["config_name"]
        .astype("string")
        .str.extract(r"^(.+)-root:", expand=False)
    )
    assignment = assignment.dropna(subset=["task_id"]).copy()
    if assignment.empty:
        raise ValueError("Assignment metrics contain no root generated-zone rows.")

    unknown = sorted(set(assignment["task_id"]) - known_task_ids)
    if unknown:
        raise ValueError(
            f"Assignment metrics reference unknown benchmark tasks: {unknown[:5]}"
        )
    missing = sorted(set(zone_metrics["task_id"]) - set(assignment["task_id"]))
    if missing:
        raise ValueError(
            f"Feasible benchmark tasks are missing assignment metrics: {missing[:5]}"
        )

    joined = assignment.merge(
        zone_metrics,
        on="task_id",
        how="inner",
        validate="many_to_one",
    )
    return joined[OUTPUT_COLUMNS]


def ensure_visualizations(
    frontier: pd.DataFrame,
    *,
    artifact_dir: str | None,
) -> dict[str, Path]:
    """Return a final PNG for each unique frontier task, rendering if absent."""
    images = {}
    settings = VisualizationRunConfig(
        enabled=True, stages="final", artifact_dir=artifact_dir
    )
    task_paths = frontier.drop_duplicates("task_id").set_index("task_id")["path"]
    for task_id, run_path_value in task_paths.items():
        run_path = Path(str(run_path_value))
        manifest = load_manifest(str(run_path))
        image = _existing_visualization(run_path, manifest)
        if image is None:
            solutions, config, manifest = load_solutions(str(run_path))
            results = render_task_visualizations(
                solutions, config, run_path, settings, manifest
            )
            write_json(str(run_path / MANIFEST_FILENAME), manifest)
            figures = [path for result in results for path in result.figure_paths]
            if len(figures) != 1:
                raise ValueError(
                    f"Expected one final visualization for {task_id}, found {len(figures)}."
                )
            image = figures[0]
        images[str(task_id)] = image
    return images


def _existing_visualization(run_path: Path, manifest: dict) -> Path | None:
    final_stage = manifest.get("final_stage")
    if final_stage:
        expected = run_path / f"visualization_{final_stage}.png"
        if expected.is_file() and expected.stat().st_size:
            return expected

    record = manifest.get("visualization") or {}
    for artifact in record.get("artifacts") or []:
        if final_stage and artifact.get("stage") != final_stage:
            continue
        for figure in artifact.get("figures") or []:
            path = run_path / str(figure)
            if path.is_file() and path.stat().st_size:
                return path
    return None


def load_decisions(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        decisions = json.load(stream)
    if not isinstance(decisions, dict) or any(
        not isinstance(key, str) or not isinstance(value, bool)
        for key, value in decisions.items()
    ):
        raise ValueError(f"Invalid Pareto review decisions in {path}.")
    return decisions


def save_decisions(path: Path, decisions: dict[str, bool]) -> None:
    write_json(str(path), decisions)


def write_outputs(
    frontier: pd.DataFrame,
    decisions: dict[str, bool],
    images: dict[str, Path],
    output_dir: Path,
) -> pd.DataFrame:
    """Write approved frontier rows and synchronize their PNG copies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    approved_ids = {task_id for task_id, approved in decisions.items() if approved}
    approved = frontier[frontier["task_id"].isin(approved_ids)][OUTPUT_COLUMNS].copy()

    csv_path = output_dir / "pareto.csv"
    temporary = output_dir / f".{csv_path.name}.tmp"
    approved.to_csv(temporary, index=False)
    os.replace(temporary, csv_path)

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    expected = {f"{task_id}.png" for task_id in approved["task_id"].unique()}
    for existing in viz_dir.glob("*.png"):
        if existing.name not in expected:
            existing.unlink()
    for task_id in approved["task_id"].unique():
        destination = viz_dir / f"{task_id}.png"
        source = images[str(task_id)]
        if (
            not destination.is_file()
            or destination.stat().st_size != source.stat().st_size
        ):
            shutil.copy2(source, destination)
    return approved


class ReviewApp:
    def __init__(
        self,
        frontier: pd.DataFrame,
        images: dict[str, Path],
        output_dir: Path,
        decisions: dict[str, bool],
    ):
        self.frontier = frontier
        self.images = images
        self.output_dir = output_dir
        self.decisions = decisions
        self.tasks = list(dict.fromkeys(frontier["task_id"].astype(str)))
        self.config_names = (
            frontier.groupby("task_id")["config_name"].apply(list).to_dict()
        )

    @property
    def decisions_path(self) -> Path:
        return self.output_dir / DECISIONS_FILENAME

    def next_task(self) -> str | None:
        return next((task for task in self.tasks if task not in self.decisions), None)

    def record(self, task_id: str, approved: bool) -> None:
        if task_id not in self.tasks:
            raise ValueError(f"Unknown Pareto task: {task_id}")
        self.decisions[task_id] = approved
        save_decisions(self.decisions_path, self.decisions)
        write_outputs(self.frontier, self.decisions, self.images, self.output_dir)


class ReviewHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, app: ReviewApp, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/":
            self._send(_review_page(self.app).encode(), "text/html; charset=utf-8")
            return
        if route == "/pareto.csv":
            self._send(
                (self.app.output_dir / "pareto.csv").read_bytes(),
                "text/csv; charset=utf-8",
            )
            return
        if route.startswith("/image/"):
            task_id = unquote(route.removeprefix("/image/")).removesuffix(".png")
            image = self.app.images.get(task_id)
            if image is not None:
                self._send(image.read_bytes(), "image/png")
                return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/decision":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("Request is too large.")
            form = parse_qs(self.rfile.read(length).decode())
            task_id = form["task_id"][0]
            decision = form["decision"][0]
            if decision not in {"yes", "no"}:
                raise ValueError("Decision must be yes or no.")
            self.app.record(task_id, decision == "yes")
        except (KeyError, IndexError, UnicodeDecodeError, ValueError) as exc:
            self.send_error(400, str(exc))
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _review_page(app: ReviewApp) -> str:
    reviewed = sum(task in app.decisions for task in app.tasks)
    approved = sum(app.decisions.get(task) is True for task in app.tasks)
    rejected = sum(app.decisions.get(task) is False for task in app.tasks)
    task_id = app.next_task()
    if task_id is None:
        body = (
            f"<h1>Review complete</h1><p>{approved} approved, {rejected} rejected.</p>"
            '<p><a href="/pareto.csv">Download pareto.csv</a></p>'
        )
    else:
        configs = app.config_names[task_id]
        body = f"""
        <h1>Is this zone plan acceptable?</h1>
        <p><strong>{html.escape(task_id)}</strong> ({len(configs)} Pareto assignment row(s))</p>
        <img src="/image/{html.escape(task_id)}.png" alt="Zone visualization">
        <form method="post" action="/decision">
          <input type="hidden" name="task_id" value="{html.escape(task_id)}">
          <button class="yes" name="decision" value="yes">Yes</button>
          <button class="no" name="decision" value="no">No</button>
        </form>
        """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pareto zone review</title>
  <style>
    body {{ font: 16px sans-serif; margin: 20px auto; max-width: 1000px; padding: 0 12px; }}
    img {{ display: block; max-height: 72vh; max-width: 100%; margin: 16px 0; }}
    button {{ border: 0; color: white; cursor: pointer; font-size: 22px; margin-right: 12px; padding: 12px 36px; }}
    .yes {{ background: #287a38; }} .no {{ background: #a52b2b; }}
  </style>
</head>
<body>
  <p>Reviewed {reviewed} of {len(app.tasks)} zone plans. Yes: {approved}. No: {rejected}.</p>
  {body}
</body>
</html>"""


def _require_columns(frame: pd.DataFrame, columns, description: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{description} are missing columns: {missing}")


def _require_numeric(frame: pd.DataFrame, columns, description: str) -> None:
    converted = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        invalid = converted.columns[converted.isna().any()].tolist()
        raise ValueError(
            f"{description} contain missing or non-numeric values: {invalid}"
        )
    frame[list(columns)] = converted


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Benchmark sweep YAML.")
    parser.add_argument("--port", type=int, default=8000, help="Local HTTP port.")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open the review page."
    )
    parser.add_argument(
        "--reset-reviews", action="store_true", help="Discard saved yes/no decisions."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    observations, root, artifact_dir = load_observations(args.config)
    frontier = pareto_front(observations)
    images = ensure_visualizations(frontier, artifact_dir=artifact_dir)
    output_dir = root / "analysis"
    decisions_path = output_dir / DECISIONS_FILENAME
    if args.reset_reviews:
        decisions_path.unlink(missing_ok=True)
    decisions = load_decisions(decisions_path)
    write_outputs(frontier, decisions, images, output_dir)

    app = ReviewApp(frontier, images, output_dir, decisions)
    server = HTTPServer(("127.0.0.1", args.port), partial(ReviewHandler, app=app))
    url = f"http://127.0.0.1:{server.server_port}/"
    print(
        f"Found {len(frontier)} Pareto rows across {len(app.tasks)} zone plans. "
        f"Review at {url}"
    )
    print(f"Approved outputs are synchronized under {output_dir}.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped Pareto zone review.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
