"""
Pipeline orchestrator for the Amihud / Free-Float liquidity project.

Usage:
    python run_pipeline.py              # run only missing steps
    python run_pipeline.py --force      # rerun everything
    python run_pipeline.py --dashboard  # launch dashboard after pipeline
"""

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Pipeline definition
# Each step is (name, command, output_file_or_glob, requires_bloomberg)
# ---------------------------------------------------------------------------
STEPS = [
    (
        "1. Download 2025 OHLCV",
        [sys.executable, "download_ohlcv.py"],
        DATA_DIR / "us_equities_2025_ohlcv.parquet",
        False,
    ),
    (
        "2. Download 2024 look-back OHLCV",
        [sys.executable, "-m", "src.download_2024"],
        DATA_DIR / "us_equities_2024_ohlcv.parquet",
        False,
    ),
    (
        "3. Bloomberg free-float",
        [sys.executable, "-m", "src.bbg_free_float"],
        DATA_DIR / "bbg_free_float_monthly_2025.parquet",
        True,
    ),
    (
        "4. Compute rolling Amihud",
        [sys.executable, "-m", "src.calc_rolling_amihud"],
        DATA_DIR / "us_equities_2025_rolling_amihud_252d.parquet",
        False,
    ),
    (
        "5. Merge data",
        [sys.executable, "-m", "src.merge_data"],
        DATA_DIR / "amihud_with_free_float.parquet",
        False,
    ),
]


def _bloomberg_available() -> bool:
    """Return True if the blpapi package can be imported."""
    try:
        importlib.import_module("blpapi")
        return True
    except ImportError:
        return False


def _output_exists(output_path: Path) -> bool:
    return output_path.exists()


def _run_step(name: str, cmd: list[str]) -> bool:
    """Run a subprocess. Returns True on success."""
    project_root = Path(__file__).resolve().parent
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print(f"\n  [FAILED] {name} exited with code {result.returncode}")
        return False
    print(f"\n  [OK] {name}")
    return True


def _status_table() -> None:
    """Print a summary table of data artefacts."""
    files = [
        ("2025 OHLCV", DATA_DIR / "us_equities_2025_ohlcv.parquet"),
        ("2024 OHLCV", DATA_DIR / "us_equities_2024_ohlcv.parquet"),
        ("Bloomberg FF", DATA_DIR / "bbg_free_float_monthly_2025.parquet"),
        ("Rolling Amihud", DATA_DIR / "us_equities_2025_rolling_amihud_252d.parquet"),
        ("Merged dataset", DATA_DIR / "amihud_with_free_float.parquet"),
    ]
    print(f"\n{'='*60}")
    print("  Data Status")
    print(f"{'='*60}")
    print(f"  {'Artefact':<25} {'Status':<10} {'Path / Info'}")
    print(f"  {'-'*25} {'-'*10} {'-'*30}")
    root = Path(__file__).resolve().parent
    for label, path in files:
        exists = path.exists()
        info = str(path.relative_to(root)) if exists else "missing"
        marker = "OK" if exists else "MISSING"
        print(f"  {label:<25} {marker:<10} {info}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Amihud / Free-Float data pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all steps even if output files already exist.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Plotly Dash dashboard after the pipeline finishes.",
    )
    args = parser.parse_args()

    has_bbg = _bloomberg_available()
    if not has_bbg:
        print(
            "[WARN] blpapi not installed -- Bloomberg free-float step will be skipped.\n"
            "       The dashboard will still work but with reduced FF analysis."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    failed = False
    for name, cmd, output_path, needs_bbg in STEPS:
        # Skip Bloomberg step when blpapi is unavailable
        if needs_bbg and not has_bbg:
            print(f"\n  [SKIP] {name} (Bloomberg not available)")
            continue

        # Skip steps whose output already exists (unless --force)
        if not args.force and _output_exists(output_path):
            print(f"  [SKIP] {name} (output already exists)")
            continue

        ok = _run_step(name, cmd)
        if not ok:
            failed = True
            print(f"\n  Pipeline stopped at: {name}")
            break

    _status_table()

    if failed:
        sys.exit(1)

    if args.dashboard:
        print("Launching dashboard at http://127.0.0.1:8050 ...")
        project_root = Path(__file__).resolve().parent
        subprocess.run(
            [sys.executable, "plotly_dash.py"],
            cwd=str(project_root),
        )


if __name__ == "__main__":
    main()
