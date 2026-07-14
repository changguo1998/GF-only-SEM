"""GreenQuery dataclass and CLI entry point."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass
class GreenQuery:
    """Result of a Green's function query.

    Attributes:
        source_xyz: Real source coordinate [x, y, z] in meters.
        receiver_xyz: Real receiver (station) coordinate [x, y, z] in meters.
        sem_source_xyz: SEM source location [x, y, z] that was matched via KDTree.
        time: Time axis array [nt].
        strain: Strain Green tensor [nt, 6, 3] or None if not requested/available.
        displacement: Displacement Green tensor [nt, 3, 3] or None.
        velocity: Velocity Green tensor [nt, 3, 3] or None.
        acceleration: Acceleration Green tensor [nt, 3, 3] or None.
        n_tiles_used: Number of tiles that contributed to this query.
        interpolation_used: True if the result was interpolated from vertices,
            False if the query source exactly matched a recorded vertex.
    """

    source_xyz: npt.NDArray[np.float64]
    receiver_xyz: npt.NDArray[np.float64]
    sem_source_xyz: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float32] | None = None
    displacement: npt.NDArray[np.float32] | None = None
    velocity: npt.NDArray[np.float32] | None = None
    acceleration: npt.NDArray[np.float32] | None = None
    n_tiles_used: int = 0
    interpolation_used: bool = False

    def summary(self) -> str:
        """Return a compact text summary of this query result."""
        lines = [
            f"source_xyz         : {self.source_xyz}",
            f"receiver_xyz       : {self.receiver_xyz}",
            f"sem_source_xyz     : {self.sem_source_xyz}",
            f"time               : {self.time.shape}, [{self.time[0]:.4f} .. {self.time[-1]:.4f}] s",
            f"n_tiles_used       : {self.n_tiles_used}",
            f"interpolation_used : {self.interpolation_used}",
        ]
        if self.strain is not None:
            lines.append(f"strain             : {self.strain.shape}")
        if self.velocity is not None:
            lines.append(f"velocity           : {self.velocity.shape}")
        if self.acceleration is not None:
            lines.append(f"acceleration       : {self.acceleration.shape}")
        if self.displacement is not None:
            lines.append(f"displacement       : {self.displacement.shape}")
        return "\n".join(lines)


def format_green_tensor(data: npt.NDArray[np.float32] | None, label: str) -> str:
    """Format a Green tensor for display."""
    if data is None:
        return f"{label}: None"
    return f"{label}: {data.shape}  range [{data.min():.6e} .. {data.max():.6e}]"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Query Green's function library by source/receiver coordinates."
    )
    parser.add_argument(
        "library_root", type=str, help="Path to the greenfun library root directory"
    )
    parser.add_argument(
        "--source",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Real source coordinates [m]",
    )
    parser.add_argument(
        "--receiver",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Real receiver (station) coordinates [m]",
    )
    parser.add_argument(
        "--quantity",
        type=str,
        choices=["strain", "displacement", "velocity", "acceleration", "both"],
        default="strain",
        help="Green's quantity to return (default: strain)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Write result to .npz file (optional)"
    )
    parser.add_argument(
        "--rebuild-index", action="store_true", help="Force rebuild of the library index cache"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for gf_greenquery."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Lazy import to avoid circular dependencies at package level.
    from greenfun.library import GreenFunctionLibrary

    lib = GreenFunctionLibrary(root=args.library_root, rebuild_index=args.rebuild_index)

    result = lib.query(source_xyz=args.source, receiver_xyz=args.receiver, quantity=args.quantity)

    if args.output:
        out_path = Path(args.output)
        data: dict[str, np.ndarray] = {
            "time": np.asarray(result.time),
            "source_xyz_m": np.asarray(result.source_xyz),
            "receiver_xyz_m": np.asarray(result.receiver_xyz),
            "sem_source_xyz_m": np.asarray(result.sem_source_xyz),
        }
        if result.strain is not None:
            data["strain"] = np.asarray(result.strain)
        if result.displacement is not None:
            data["displacement"] = np.asarray(result.displacement)
        if result.velocity is not None:
            data["velocity"] = np.asarray(result.velocity)
        if result.acceleration is not None:
            data["acceleration"] = np.asarray(result.acceleration)
        np.savez_compressed(out_path, **data)
        print(f"Wrote {out_path}")
    else:
        print(result.summary())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
