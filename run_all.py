"""Run every figure script in sequence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import sector_bb84_qkd_figures


SCRIPTS = [
    "01_bloch_five_rounds.py",
    "02_psb_open_path.py",
    "03_psb_geodesic_closure.py",
    "04_lambda_gamma_evolution.py",
    "sector_bb84_qkd_figures.py"
]


def main() -> None:
    root = Path(__file__).parent

    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        completed = subprocess.run(
            [sys.executable, str(root / script)],
            cwd=root,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"{script} failed with exit code {completed.returncode}"
            )


if __name__ == "__main__":
    main()
