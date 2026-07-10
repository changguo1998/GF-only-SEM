"""Tests for TrilinearInterpolator."""

from __future__ import annotations

import numpy as np
import pytest

from greenfun.interpolator import TrilinearInterpolator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _regular_3d_grid(
    nx: int = 4,
    ny: int = 4,
    nz: int = 4,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> np.ndarray:
    """Return an ``(nx*ny*nz, 3)`` array of regularly-spaced 3-D points."""
    xs = x0 + dx * np.arange(nx)
    ys = y0 + dy * np.arange(ny)
    zs = z0 + dz * np.arange(nz)
    mesh = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrilinearInterpolator:
    """Suite for TrilinearInterpolator."""

    def test_linear_field_exact(self) -> None:
        """Trilinear interpolation recovers an analytic linear field exactly."""
        # 5³ grid, spacing 2 → vertices at 0, 2, 4, 6, 8 in each axis
        grid_coords = _regular_3d_grid(nx=5, ny=5, nz=5, dx=2.0, dy=2.0, dz=2.0)
        n_vert = grid_coords.shape[0]

        # Linear field: f(x,y,z) = 1 + 3x + 2y + 0.5z
        a, b, c, d = 1.0, 3.0, 2.0, 0.5
        field = a + b * grid_coords[:, 0] + c * grid_coords[:, 1] + d * grid_coords[:, 2]

        interp = TrilinearInterpolator(grid_coords)

        # Query at several interior points
        test_points = [
            np.array([3.0, 5.0, 7.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([7.0, 3.0, 5.0]),
            np.array([4.5, 2.5, 6.5]),
        ]
        for pt in test_points:
            expected = a + b * pt[0] + c * pt[1] + d * pt[2]
            result = interp.interpolate(pt, field)
            assert np.abs(result - expected) < _EPS, (
                f"Linear field mismatch at {pt}: got {result}, expected {expected}"
            )

    def test_cube_center_weights(self) -> None:
        """All 8 corners get equal weight at the center of a cube."""
        # Single cube cell: vertices at (±1, ±1, ±1)
        corners = np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        interp = TrilinearInterpolator(corners)

        # Assign unique values per vertex so we can check weighting
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

        center = np.array([0.0, 0.0, 0.0])
        result = interp.interpolate(center, values)
        expected = values.mean()  # uniform weights → mean
        assert np.abs(result - expected) < _EPS, (
            f"Center weight mismatch: got {result}, expected {expected}"
        )

    def test_out_of_range_raises(self) -> None:
        """Point outside the vertex cloud raises ValueError."""
        coords = _regular_3d_grid(nx=3, ny=3, nz=3)
        interp = TrilinearInterpolator(coords)

        far_point = np.array([1000.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="outside"):
            interp.interpolate(far_point, np.ones(coords.shape[0]))

    def test_on_vertex_exact_match(self) -> None:
        """Query at a vertex returns that vertex's value exactly."""
        coords = _regular_3d_grid(nx=5, ny=5, nz=5, dx=2.0, dy=2.0, dz=2.0)
        n_vert = coords.shape[0]
        values = np.arange(n_vert, dtype=np.float64) * 10.0

        interp = TrilinearInterpolator(coords)

        for idx in [0, 12, 24, 27, 62, 124]:
            pt = coords[idx]
            result = interp.interpolate(pt, values)
            expected = values[idx]
            assert np.abs(result - expected) < _EPS, (
                f"On-vertex mismatch at index {idx} (point {pt}): "
                f"got {result}, expected {expected}"
            )

    def test_trailing_dimensions_preserved(self) -> None:
        """Trailing tensor dimensions are unchanged after interpolation."""
        coords = _regular_3d_grid(nx=4, ny=4, nz=4)
        n_vert = coords.shape[0]

        # Values with trailing dims: [n_vert, 2, 3]
        rng = np.random.default_rng(1234)
        values = rng.standard_normal((n_vert, 2, 3)).astype(np.float64)

        interp = TrilinearInterpolator(coords)
        pt = np.array([1.3, 1.7, 2.1])
        result = interp.interpolate(pt, values)

        assert result.shape == (2, 3), f"Expected shape (2, 3), got {result.shape}"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_point_at_cell_face_center(self) -> None:
        """Point at the center of a cell face (2-D plane) still works."""
        coords = _regular_3d_grid(nx=6, ny=6, nz=6, dx=1.5, dy=1.5, dz=1.5)
        n_vert = coords.shape[0]
        field = 2.0 + 0.5 * coords[:, 0] - 1.0 * coords[:, 1] + 3.0 * coords[:, 2]

        interp = TrilinearInterpolator(coords)

        # Face center at z = 0.75 (midpoint of cell in z), x=1.5, y=1.5
        pt = np.array([1.5, 1.5, 0.75])
        expected = 2.0 + 0.5 * 1.5 - 1.0 * 1.5 + 3.0 * 0.75
        result = interp.interpolate(pt, field)
        assert np.abs(result - expected) < _EPS

    def test_scalar_1d_values(self) -> None:
        """Interpolation with 1-D values returns a scalar."""
        coords = _regular_3d_grid(nx=4, ny=4, nz=4)
        values = np.ones(coords.shape[0])
        interp = TrilinearInterpolator(coords)
        result = interp.interpolate(np.array([1.5, 1.5, 1.5]), values)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 0  # 0-D array (scalar equivalent)

    def test_wrong_shape_point_raises(self) -> None:
        """Point with wrong shape raises ValueError."""
        coords = _regular_3d_grid(nx=3, ny=3, nz=3)
        interp = TrilinearInterpolator(coords)
        with pytest.raises(ValueError, match="shape"):
            interp.interpolate(np.array([1.0, 2.0]), np.ones(coords.shape[0]))

    def test_wrong_values_count_raises(self) -> None:
        """Values with mismatched first dimension raises ValueError."""
        coords = _regular_3d_grid(nx=4, ny=4, nz=4)
        interp = TrilinearInterpolator(coords)
        with pytest.raises(ValueError, match="first dimension"):
            interp.interpolate(np.array([1.0, 2.0, 3.0]), np.ones(10))
