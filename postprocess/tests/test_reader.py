"""Tests for gf_post.reader — RecordReader, GeometryReader, merge_records."""

import numpy as np
from gf_post.reader import GeometryReader, RecordReader, merge_records


class TestRecordReader:
    def test_context_manager(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            assert rr.nsteps == 2
            assert rr.source_direction == 0

    def test_dt(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            assert rr.dt == 0.01

    def test_local_element_ids(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            ids = rr.local_element_ids
            np.testing.assert_array_equal(ids, [1])

    def test_read_strain_shape(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            strain = rr.read_strain(0)
            assert strain.shape == (1, 3, 3, 3, 6)

    def test_read_strain_values(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            strain = rr.read_strain(0)
            # First timestep: xx component = 1.0 everywhere
            assert np.isclose(strain[0, :, :, :, 0], 1.0).all()
            # yy = 2.0, zz = 3.0
            assert np.isclose(strain[0, :, :, :, 1], 2.0).all()
            assert np.isclose(strain[0, :, :, :, 2], 3.0).all()

    def test_read_all_strain_shape(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            all_strain = rr.read_all_strain()
            assert all_strain.shape == (2, 1, 3, 3, 3, 6)

    def test_read_all_strain_values(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            all_strain = rr.read_all_strain()
            # Timestep 0: xx=1.0, Timestep 1: xx=2.0
            assert np.isclose(all_strain[0, 0, :, :, :, 0], 1.0).all()
            assert np.isclose(all_strain[1, 0, :, :, :, 0], 2.0).all()

    def test_n_records(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            assert rr.n_records == 2

    def test_record_interval(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            assert rr.record_interval == 1


class TestGeometryReader:
    def test_context_manager(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            assert gr.n_cell == 1
            assert gr.ngll == 3

    def test_coords_shape(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            assert gr.coords.shape == (1, 3, 3, 3, 3)

    def test_coords_range(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            coords = gr.coords
            # Unit cube [0,1]^3
            assert np.all(coords >= 0)
            assert np.all(coords <= 1.0 + 1e-10)

    def test_dxi_dx_shape(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            assert gr.dxi_dx.shape == (1, 3, 3, 3, 3, 3)

    def test_dxi_dx_unit_cube(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            dxi_dx = gr.dxi_dx
            # For unit cube, dxi/dx = diag(2,2,2) everywhere
            for d in range(3):
                assert np.isclose(dxi_dx[0, :, :, :, d, d], 2.0).all()

    def test_is_pml(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            assert gr.is_pml.shape == (1,)
            assert gr.is_pml[0] == 0

    def test_n_cell(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            assert gr.n_cell == 1


class TestMergeRecords:
    def test_merge_single_rank(self, synthetic_record_path):
        merged, info = merge_records([synthetic_record_path], n_cell=1)
        assert merged.shape == (2, 1, 3, 3, 3, 6)
        assert info["dt"] == 0.01
        assert info["nsteps"] == 2

    def test_merge_two_ranks(self, synthetic_multirank_records):
        merged, info = merge_records(synthetic_multirank_records, n_cell=2)
        assert merged.shape == (2, 2, 3, 3, 3, 6)
        # Element 1 gets val = t+1, Element 2 gets val = t+1+10
        assert np.isclose(merged[0, 0, :, :, :, 0], 1.0).all()
        assert np.isclose(merged[0, 1, :, :, :, 0], 11.0).all()
        assert np.isclose(merged[1, 0, :, :, :, 0], 2.0).all()
        assert np.isclose(merged[1, 1, :, :, :, 0], 12.0).all()

    def test_merge_info(self, synthetic_multirank_records):
        _, info = merge_records(synthetic_multirank_records, n_cell=2)
        assert info["dt"] == 0.01
        assert info["nsteps"] == 2
