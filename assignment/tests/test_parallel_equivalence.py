"""
Regression test: verifies that parallel execution produces identical CSV
outputs to sequential execution.

Requires real SFUSD data — cannot run without it.

Usage:
    python tests/test_parallel_equivalence.py \
        --config-path configs/erabasse.config.yaml \
        --n-subconfigs 2 \
        --n-iters 2 \
        --workers 4

    # Keep temp dirs for manual inspection:
    python tests/test_parallel_equivalence.py \
        --config-path configs/erabasse.config.yaml \
        --keep-outputs
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "run_custom_config.py"
DEFAULT_N_SUBCONFIGS = 2
DEFAULT_N_ITERS = 2
DEFAULT_WORKERS = 2


def make_minimal_config(
    base_path: str,
    output_dir: str,
    n_subconfigs: int,
    n_iters: int,
) -> str:
    """Write a trimmed copy of the base config to a temp YAML file.

    Args:
        base_path: Path to the real base config YAML.
        output_dir: Where simulated assignments should be saved.
        n_subconfigs: How many subconfigs to keep (first N).
        n_iters: Number of iterations (iterations.end will be set to this).

    Returns:
        Path to the temp YAML file (caller must delete it).

    Raises:
        ValueError: If the base config has fewer subconfigs than requested.
    """
    with open(base_path) as f:
        config = yaml.safe_load(f)

    all_subs = config.get("subconfigs", [])
    if len(all_subs) < n_subconfigs:
        raise ValueError(
            f"Config only has {len(all_subs)} subconfigs, "
            f"requested {n_subconfigs}."
        )

    config["subconfigs"] = all_subs[:n_subconfigs]
    config["iterations"] = {"start": 0, "end": n_iters}
    config["paths"]["assignment-folder"] = str(output_dir)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, mode="w", prefix="parallel_test_cfg_"
    )
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.close()
    return tmp.name


def run_simulation(
    config_path: str, workers: int
) -> subprocess.CompletedProcess:
    """Invoke run_custom_config.py as a subprocess.

    Args:
        config_path: Path to the config YAML.
        workers: Number of parallel workers (1 = sequential).

    Returns:
        CompletedProcess with stdout/stderr captured.
    """
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--config-path",
        config_path,
        "--workers",
        str(workers),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def collect_csvs(directory: pathlib.Path) -> dict[str, pd.DataFrame]:
    """Recursively collect all assignment CSVs, keyed by relative path.

    Root provenance files are not CSVs and are therefore excluded; their
    assignment-folder values legitimately differ between the two runs.

    Args:
        directory: Root directory to search.

    Returns:
        Dict mapping relative path string to DataFrame.
    """
    result = {}
    for csv_path in sorted(directory.rglob("*.csv")):
        rel = str(csv_path.relative_to(directory))
        result[rel] = pd.read_csv(csv_path)
    return result


def compare_outputs(
    seq_dir: pathlib.Path,
    par_dir: pathlib.Path,
) -> list[str]:
    """Compare two output directories, returning a list of error messages.

    Args:
        seq_dir: Sequential run output directory.
        par_dir: Parallel run output directory.

    Returns:
        Empty list if all CSVs match, otherwise a list of error descriptions.
    """
    errors: list[str] = []
    seq_csvs = collect_csvs(seq_dir)
    par_csvs = collect_csvs(par_dir)

    seq_keys = set(seq_csvs)
    par_keys = set(par_csvs)

    for missing in sorted(seq_keys - par_keys):
        errors.append(f"Missing in parallel output: {missing}")
    for extra in sorted(par_keys - seq_keys):
        errors.append(f"Unexpected file in parallel output: {extra}")

    for key in sorted(seq_keys & par_keys):
        try:
            pd.testing.assert_frame_equal(seq_csvs[key], par_csvs[key])
        except AssertionError as exc:
            errors.append(f"Mismatch in {key}:\n    {exc}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify parallel == sequential for run_custom_config.py."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to a real base config YAML (e.g. configs/erabasse.config.yaml).",
    )
    parser.add_argument(
        "--n-subconfigs",
        type=int,
        default=DEFAULT_N_SUBCONFIGS,
        help=f"Number of subconfigs to use (default: {DEFAULT_N_SUBCONFIGS}).",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=DEFAULT_N_ITERS,
        help=f"Number of iterations to run (default: {DEFAULT_N_ITERS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Workers for the parallel run (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Do not delete temp output directories after the test.",
    )
    args = parser.parse_args()

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="parallel_equiv_test_"))
    seq_dir = tmpdir / "sequential"
    par_dir = tmpdir / "parallel"
    seq_dir.mkdir()
    par_dir.mkdir()

    print(f"[INFO] Temp outputs: {tmpdir}")
    print(
        f"[INFO] Test parameters: "
        f"{args.n_subconfigs} subconfigs, "
        f"{args.n_iters} iterations, "
        f"{args.workers} workers"
    )

    seq_config_path = None
    par_config_path = None

    try:
        seq_config_path = make_minimal_config(
            args.config_path, str(seq_dir), args.n_subconfigs, args.n_iters
        )
        par_config_path = make_minimal_config(
            args.config_path, str(par_dir), args.n_subconfigs, args.n_iters
        )

        # --- Sequential run ---
        print("\n[INFO] Running SEQUENTIAL (workers=1)...")
        result = run_simulation(seq_config_path, workers=1)
        if result.returncode != 0:
            print(f"[ERROR] Sequential run failed (exit {result.returncode}):")
            print(result.stderr[-3000:])
            sys.exit(1)
        seq_files = list(seq_dir.rglob("*.csv"))
        print(f"[INFO] Sequential complete — {len(seq_files)} CSV(s) written.")

        # --- Parallel run ---
        print(f"\n[INFO] Running PARALLEL (workers={args.workers})...")
        result = run_simulation(par_config_path, workers=args.workers)
        if result.returncode != 0:
            print(f"[ERROR] Parallel run failed (exit {result.returncode}):")
            print(result.stderr[-3000:])
            sys.exit(1)
        par_files = list(par_dir.rglob("*.csv"))
        print(f"[INFO] Parallel complete — {len(par_files)} CSV(s) written.")

        # --- Compare ---
        print("\n[INFO] Comparing outputs...")
        errors = compare_outputs(seq_dir, par_dir)

        if errors:
            print(f"\n[FAIL] {len(errors)} difference(s) found:")
            for err in errors:
                print(f"  • {err}")
            sys.exit(1)
        else:
            print(
                f"\n[PASS] All {len(seq_files)} CSV file(s) are "
                f"bit-for-bit identical between sequential and parallel runs."
            )

    finally:
        for tmp_cfg in filter(None, [seq_config_path, par_config_path]):
            try:
                os.unlink(tmp_cfg)
            except OSError:
                pass

        if args.keep_outputs:
            print(f"[INFO] Outputs kept at: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print("[INFO] Temp outputs cleaned up.")


if __name__ == "__main__":
    main()
