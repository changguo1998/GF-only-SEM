"""Tests for gf_post.reader — RecordReader, GeometryReader, ConfigReader."""

import h5py
import numpy as np
from gf_post.reader import ConfigReader, GeometryReader, RecordReader, merge_vertex_records


class TestRecordReader:
    def test_context_manager(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            assert rr.n_snapshots == 2
            assert rr.source_direction == "x"

    def test_vertex_ids(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            vids = rr.vertex_ids
            np.testing.assert_array_equal(vids, [1])

    def test_read_strain_shape(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            strain = rr.read_strain(0)
            assert strain.shape == (1, 6)

    def test_read_strain_values(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            strain = rr.read_strain(0)
            assert np.isclose(strain[0, 0], 1.0)
            assert np.isclose(strain[0, 1], 2.0)
            assert np.isclose(strain[0, 2], 3.0)

    def test_read_all_strain_shape(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            all_strain = rr.read_all_strain()
            assert all_strain.shape == (2, 1, 6)

    def test_read_all_strain_values(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            all_strain = rr.read_all_strain()
            assert np.isclose(all_strain[0, 0, 0], 1.0)
            assert np.isclose(all_strain[1, 0, 0], 2.0)

    def test_n_snapshots(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            assert rr.n_snapshots == 2

    def test_n_vertices(self, synthetic_record_path):
        with RecordReader(synthetic_record_path) as rr:
            assert rr.n_vertices == 1


class TestGeometryReader:
    def test_context_manager(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            assert gr.n_vertex == 8

    def test_vertex_coords_shape(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            assert gr.vertex_coords.shape == (8, 3)

    def test_domain_bounds(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            bounds = gr.domain_bounds
            assert bounds["xmin"] == 0.0
            assert bounds["xmax"] == 1.0
            assert bounds["zmin"] == 0.0


class TestConfigReader:
    def test_tile_fields(self, synthetic_config_path):
        cfg = ConfigReader(synthetic_config_path)
        assert cfg.nx_elements == 16
        assert cfg.ny_elements == 16
        assert cfg.tilex_elements == [5, 5]
        assert cfg.tiley_elements == [5, 5]
        assert cfg.pml_thickness["xmin"] == 3
        assert cfg.pml_thickness["xmax"] == 3
        assert cfg.solver_dt == 0.01
        assert cfg.nsteps == 2
        cfg.close()

    def test_record_depth(self, synthetic_config_path):
        cfg = ConfigReader(synthetic_config_path)
        assert cfg.record_depth_max_m == 1.0
        assert cfg.record_depth_actual_m == 1.0
        cfg.close()


class TestMergeVertexRecords:
    def test_merge_single_rank(self, synthetic_record_path):
        merged, mask = merge_vertex_records([synthetic_record_path], n_vertex=8)
        assert merged.shape == (2, 8, 6)
        assert mask[0]  # vertex 0 (1-based ID 1) recorded
        assert not mask[1]  # vertex 1 not recorded
        # Check values
        assert np.isclose(merged[0, 0, 0], 1.0)
        assert np.isclose(merged[1, 0, 0], 2.0)

    def test_merge_two_ranks(self, synthetic_multirank_records):
        merged, mask = merge_vertex_records(synthetic_multirank_records, n_vertex=8)
        assert merged.shape == (2, 8, 6)
        assert mask[0] and mask[1]  # vertices 0 and 1 recorded
        # Vertex 0 (rank 0): val = t+1
        assert np.isclose(merged[0, 0, 0], 1.0)
        assert np.isclose(merged[1, 0, 0], 2.0)
        # Vertex 1 (rank 1): val = t+1+10
        assert np.isclose(merged[0, 1, 0], 11.0)
        assert np.isclose(merged[1, 1, 0], 12.0)

    def test_merge_overlap_warns(self, tmp_path, capsys):
        """Two ranks recording the same vertex emit a warning; last value wins."""
        n_snapshots = 2
        paths = []
        for rank in [0, 1]:
            path = tmp_path / f"record_{rank}.h5"
            paths.append(path)
            with h5py.File(path, "w") as f:
                f.attrs["source_direction"] = "x"
                f.attrs["basis"] = "mesh_vertices"
                f.attrs["excludes_pml"] = True
                f.create_dataset("vertex_ids", data=np.array([1], dtype=np.int64))
                strain = np.zeros((n_snapshots, 1, 6), dtype=np.float64)
                vertex_value = float(rank) + 1.0  # rank0 → 1.0, rank1 → 2.0
                for t in range(n_snapshots):
                    strain[t, 0, :] = [vertex_value, vertex_value, vertex_value, 0.0, 0.0, 0.0]
                f.create_dataset("strain", data=strain, maxshape=(None, 1, 6))

        merged, mask = merge_vertex_records(paths, n_vertex=8)
        captured = capsys.readouterr()
        assert "recorded by multiple ranks" in captured.err
        # Last rank wins → value 2.0
        assert np.isclose(merged[0, 0, 0], 2.0)
        assert mask[0]
