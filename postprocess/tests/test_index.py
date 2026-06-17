"""Tests for gf_post.index — KD-tree spatial index over element centroids."""

import numpy as np
import pytest

from gf_post.index import ElementIndex


class TestElementIndex:
    def test_build_index(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as reader:
            coords = reader.coords
            is_pml = reader.is_pml
        idx = ElementIndex(coords, is_pml)
        assert hasattr(idx, "_tree")
        assert hasattr(idx, "_indices")

    def test_query_single_point(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as reader:
            coords = reader.coords
            is_pml = reader.is_pml
        idx = ElementIndex(coords, is_pml)
        point = np.array([0.5, 0.5, 0.5])
        indices, dist = idx.query(point, k=1)
        assert indices.shape == (1,)
        assert dist.shape == (1,)
        # The unit cube element should contain (0.5, 0.5, 0.5)
        assert indices[0] == 0

    def test_query_multi_point(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as reader:
            coords = reader.coords
            is_pml = reader.is_pml
        idx = ElementIndex(coords, is_pml)
        points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])
        indices, dist = idx.query(points, k=1)
        assert indices.shape == (3,)
        assert dist.shape == (3,)
        assert np.all(indices == 0)

    def test_query_k_nearest(self, synthetic_mesh_2elem_path):
        with GeometryReader(synthetic_mesh_2elem_path) as reader:
            coords = reader.coords
            is_pml = reader.is_pml
        idx = ElementIndex(coords, is_pml)
        # Query near element 0 center (3D point -> shape (1, 2))
        point = np.array([0.5, 0.5, 0.5])
        indices, dist = idx.query(point, k=2)
        assert indices.shape == (1, 2)
        assert dist.shape == (1, 2)
        # Closest should be element 0
        assert indices[0, 0] == 0

        # Query near element 1 center
        point = np.array([1.5, 0.5, 0.5])
        indices, dist = idx.query(point, k=2)
        assert indices[0, 0] == 1  # Closest to element 1

    def test_pml_exclusion(self, synthetic_mesh_2elem_path):
        with GeometryReader(synthetic_mesh_2elem_path) as reader:
            coords = reader.coords
        # Mark element 0 as PML
        is_pml = np.array([1, 0], dtype=np.int8)
        idx = ElementIndex(coords, is_pml)
        assert len(idx._indices) == 1
        assert idx._indices[0] == 1

        # Query at element 0 center — should return element 1 as closest
        point = np.array([0.5, 0.5, 0.5])
        indices, dist = idx.query(point, k=1)
        assert indices[0] == 1  # Only element 1 is indexed


from gf_post.reader import GeometryReader