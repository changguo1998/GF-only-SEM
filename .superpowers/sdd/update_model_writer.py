"""Script to update model_writer.py with recording map support."""

import re

with open("preprocess/model_writer.py", "r") as f:
    content = f.read()

# 1. Update write_model signature + docstring
content = content.replace(
    'def write_model(\n    model_path: str,\n    topology: TopologyData,\n    fields: dict[str, npt.NDArray],\n    boundary_tag: npt.NDArray[np.int64],\n    domain_bounds: dict[str, float],\n    partition_result: dict | None = None,\n) -> None:\n    """Extend model.h5 with field data and write partition files.',
    'def write_model(\n    model_path: str,\n    topology: TopologyData,\n    fields: dict[str, npt.NDArray],\n    boundary_tag: npt.NDArray[np.int64],\n    domain_bounds: dict[str, float],\n    partition_result: dict | None = None,\n    recording_map: dict | None = None,\n) -> None:\n    """Extend model.h5 with field data and write partition files.',
)

# 2. Update call to _write_partition_files
content = content.replace(
    "    if partition_result is not None:\n        _write_partition_files(model_path, topology, fields, boundary_tag, partition_result)",
    "    if partition_result is not None:\n        _write_partition_files(model_path, topology, fields, boundary_tag, partition_result,\n                                recording_map=recording_map)",
)

# 3. Update _write_partition_files signature
content = content.replace(
    "def _write_partition_files(\n    model_path: str,\n    topology: TopologyData,\n    fields: dict[str, npt.NDArray],\n    boundary_tag: npt.NDArray[np.int64],\n    partition_result: dict,\n) -> None:",
    "def _write_partition_files(\n    model_path: str,\n    topology: TopologyData,\n    fields: dict[str, npt.NDArray],\n    boundary_tag: npt.NDArray[np.int64],\n    partition_result: dict,\n    recording_map: dict | None = None,\n) -> None:",
)

# 4. Add recording map writing after exchange patterns
# Find the line after exchange writing and insert recording map code
recording_code = """
            # Write recording map if present
            if recording_map is not None:
                per_rank_rec = recording_map.get("per_rank_recording", {}).get(r)
                if per_rank_rec is not None and len(per_rank_rec.get("vertex_ids", [])) > 0:
                    rec_grp = f.create_group("recording")
                    rec_grp.attrs["basis"] = "mesh_vertices"
                    rec_grp.attrs["record_depth_max_m"] = recording_map.get("record_depth_actual_m", 0.0)
                    rec_grp.attrs["record_depth_actual_m"] = recording_map.get("record_depth_actual_m", 0.0)
                    rec_grp.attrs["green_tile_size_m"] = recording_map.get("green_tile_size_m", 0.0)
                    rec_grp.attrs["excludes_pml"] = True
                    _write_dataset(rec_grp, "save_element_mask",
                                   np.array(per_rank_rec["save_element_mask"], dtype=bool), dtype="bool")
                    _write_dataset(rec_grp, "vertex_ids",
                                   np.array(per_rank_rec["vertex_ids"], dtype=np.int64), dtype="int64")
                    _write_dataset(rec_grp, "source_element_local_index",
                                   np.array(per_rank_rec["source_element_local_index"], dtype=np.int32), dtype="int32")
                    _write_dataset(rec_grp, "source_corner_index",
                                   np.array(per_rank_rec["source_corner_index"], dtype=np.int8), dtype="int8")
"""

# Insert after the exchange-writing block (after "recv_dof" line)
insert_point = '                    _write_dataset(ng, "recv_dof", recv_arr, dtype="int32")\n\n            # Write recording map if present'
# Find and replace the end of exchange block + start of recording map
# Actually, we want to insert after the exchange block closes.
# Let's find the line after exchange writing and before the function returns.
# The pattern: exchange block ends, then there's just the end of the for r loop.
# Let's insert before the r-loop closes (the function has no explicit return, just ends after the loop).

# Find the closing of the for r in range(n_ranks) loop
# It ends with the _write_dataset calls for exchange, then a blank line before the function ends.
# Let's insert after the exchange block and before the next iteration.

# Pattern: exchange block ends with recv_dof write, then loop continues
old = '                    _write_dataset(ng, "recv_dof", recv_arr, dtype="int32")'
new = old + recording_code
content = content.replace(old, new)

with open("preprocess/model_writer.py", "w") as f:
    f.write(content)

print("Done")
