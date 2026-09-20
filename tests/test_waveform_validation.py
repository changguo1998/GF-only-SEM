from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = PROJECT_ROOT / "examples" / "_shared" / "verify_waveform.py"


def test_waveform_validator_accepts_matching_tensor(tmp_path: Path) -> None:
    result_path = tmp_path / "comparison.npz"
    reference = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    np.savez(
        result_path,
        time=np.array([0.0, 0.5, 1.0]),
        reference_displacement=reference,
        sem_displacement=reference,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(result_path),
            "--end-time-s",
            "1.0",
            "--min-correlation",
            "0.99",
            "--max-fitted-rel-l2",
            "0.01",
            "--min-scale",
            "0.8",
            "--max-scale",
            "1.2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PASS" in completed.stdout


def test_waveform_validator_rejects_threefold_amplitude_error(tmp_path: Path) -> None:
    result_path = tmp_path / "comparison.npz"
    reference = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    np.savez(
        result_path,
        time=np.array([0.0, 0.5, 1.0]),
        reference_displacement=reference,
        sem_displacement=3.0 * reference,
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(result_path),
            "--end-time-s",
            "1.0",
            "--min-correlation",
            "0.99",
            "--max-fitted-rel-l2",
            "0.01",
            "--min-scale",
            "0.8",
            "--max-scale",
            "1.2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "FAIL" in completed.stdout
    assert "scale=3.000000" in completed.stdout
