"""CLI tests for gf_greenquery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def library_root(greenfun_library) -> Path:
    """Fixture providing a synthetic greenfun library root path."""
    return greenfun_library.root


def test_cli_help():
    """gf_greenquery --help prints usage and exits 0."""
    from greenfun.query import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_query_strain(library_root: Path):
    """Basic strain query via CLI produces correct output."""
    from greenfun.query import main

    # Query the first source at its own location (should get exact vertex match)
    exit_code = main(
        [
            str(library_root),
            "--source",
            "50",
            "50",
            "0",
            "--receiver",
            "100",
            "0",
            "0",
            "--quantity",
            "strain",
        ]
    )
    assert exit_code == 0


def test_cli_query_displacement(library_root: Path):
    """Displacement query via CLI."""
    from greenfun.query import main

    exit_code = main(
        [
            str(library_root),
            "--source",
            "50",
            "50",
            "0",
            "--receiver",
            "100",
            "0",
            "0",
            "--quantity",
            "displacement",
        ]
    )
    assert exit_code == 0


def test_cli_query_both(library_root: Path):
    """Both quantities via CLI."""
    from greenfun.query import main

    exit_code = main(
        [
            str(library_root),
            "--source",
            "50",
            "50",
            "0",
            "--receiver",
            "100",
            "0",
            "0",
            "--quantity",
            "both",
        ]
    )
    assert exit_code == 0


def test_cli_output_npz(library_root: Path, tmp_path: Path):
    """CLI --output writes valid .npz file."""
    from greenfun.query import main

    out_path = tmp_path / "result.npz"
    exit_code = main(
        [
            str(library_root),
            "--source",
            "50",
            "50",
            "0",
            "--receiver",
            "100",
            "0",
            "0",
            "--quantity",
            "both",
            "--output",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert out_path.exists()

    data = np.load(out_path)
    assert "time" in data
    assert "source_xyz_m" in data
    assert "receiver_xyz_m" in data
    assert "sem_source_xyz_m" in data
    assert "strain" in data
    assert "displacement" in data


def test_cli_rebuild_index(library_root: Path):
    """--rebuild-index flag forces index rebuild."""
    from greenfun.query import main

    exit_code = main(
        [
            str(library_root),
            "--source",
            "50",
            "50",
            "0",
            "--receiver",
            "100",
            "0",
            "0",
            "--quantity",
            "strain",
            "--rebuild-index",
        ]
    )
    assert exit_code == 0


def test_cli_invalid_quantity(library_root: Path):
    """Invalid quantity should fail."""
    from greenfun.query import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                str(library_root),
                "--source",
                "0",
                "0",
                "0",
                "--receiver",
                "0",
                "0",
                "0",
                "--quantity",
                "invalid_option",
            ]
        )
