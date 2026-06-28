# Task 2: Forward Solver — Restart Writer + Recording Map Integration

## Summary

Add restart/resume capability to the forward solver and change `RecordWriter` to use the preprocess-built recording map (shallow mesh vertices only, not full GLL).

## Files to Create/Modify

1. **Create** `forward/include/gf/restart.hpp` — `RestartWriter` + `RestartReader` class declarations
1. **Create** `forward/src/restart.cpp` — implementation
1. `forward/include/gf/record.hpp` — change `RecordWriter` to use recording map
1. `forward/src/record.cpp` — implement shallow vertex recording
1. `forward/include/gf/io.hpp` — add recording map to `RankData`, add `restart_dt_s`/`restart_stride` to `ConfigData`
1. `forward/src/io.cpp` — read `/recording/` group from partition, read restart config
1. `forward/include/gf/types.hpp` — add `RecordingMap` struct
1. `forward/include/gf/solver.hpp` — add `--resume` flag
1. `forward/src/solver.cpp` — integrate restart writes, resume, shallow recording
1. `forward/src/main.cpp` — parse `--resume` flag
1. `forward/CMakeLists.txt` — add `src/restart.cpp`

## Design

### RecordingMap struct (types.hpp)

```cpp
struct RecordingMap {
    bool has_recording = false;           // false if no /recording/ group in partition
    std::vector<int64_t> vertex_ids;      // global mesh vertex IDs [n_vertices]
    std::vector<int32_t> src_elem_local;  // local element index [n_vertices]
    std::vector<int8_t> src_corner;       // corner index 0-7 [n_vertices]
};
```

Add to `RankData`:

```cpp
RecordingMap recording;
```

### ConfigData additions (types.hpp)

```cpp
double record_depth_max_m = 0.0;
double record_depth_actual_m = 0.0;
double green_tile_size_m = 0.0;
double restart_dt_s = 0.0;       // 0 = no restart writes
int restart_stride = 0;          // 0 = no restart writes
bool resume_mode = false;        // if true, restore from restart file
```

### RecordWriter changes

**Before:** writes `/strain` with shape `[n_snapshots, n_elem_local, NGLL, NGLL, NGLL, 6]`

**After:** writes `/strain` with shape `[n_snapshots, n_record_vertices, 6]`

Schema:

```
wavefields/{direction}/record_{r}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string
│   ├── basis                   : "mesh_vertices"
│   ├── record_depth_max_m      : float64
│   ├── record_depth_actual_m   : float64
│   └── excludes_pml            : bool
├── vertex_ids                  : int64[n_record_vertices]
└── strain                      : float32[n_snapshots, n_record_vertices, 6]
```

Constructor changes:

```cpp
RecordWriter(const std::string& output_dir, const std::string& source_direction, int rank,
             const RecordingMap& rec_map, int ngll,
             CompressionConfig compression, bool use_float32 = false);
```

- No more `n_local_elem` / `element_ids` — use `rec_map.vertex_ids.size()` for n_vertices
- Write `vertex_ids` dataset
- Write new attrs: `basis`, `record_depth_max_m`, `record_depth_actual_m`, `excludes_pml`

`write_step()` now takes `strain` as `[n_record_vertices * 6]` (shallow vertices only).

### RestartWriter class (restart.hpp)

```cpp
class RestartWriter {
public:
    RestartWriter(const std::string& output_dir, const std::string& source_direction,
                  int rank, int n_local_elem, int ngll);
    ~RestartWriter();
    void write(int step, double time_s,
               const std::vector<double>& displacement,
               const std::vector<double>& velocity,
               const std::vector<double>& acceleration,
               const std::vector<double>& pml_damping);
    void close();
private:
    // ... hdf5 handles, filepath
};
```

Schema:

```
restart/{direction}/restart_{r}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string
│   ├── step                    : int32
│   ├── time_s                  : float64
│   └── ngll                    : int32
├── displacement                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── velocity                    : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── acceleration                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
└── pml_damping                 : float64[n_elem_local, NGLL, NGLL, NGLL]  (current damping field)
```

Overwrites the same file each time (latest-only).

### RestartReader (restart.hpp)

```cpp
struct RestartState {
    int step;
    double time_s;
    std::vector<double> displacement;
    std::vector<double> velocity;
    std::vector<double> acceleration;
    std::vector<double> pml_damping;
};

RestartState read_restart(const std::string& output_dir, const std::string& source_direction, int rank);
```

### Solver changes (solver.cpp)

1. After reading partition, check if `recording.has_recording` is true
1. Initialize RecordWriter with `rec_map` (shallow vertices)
1. At snapshot steps:
   - Compute strain at all GLL nodes (current full computation)
   - **Extract only recording-map vertices** using `src_elem_local` + `src_corner` indices
   - Write shallow strain via `record.write_step()`
1. Compute `restart_stride` from config: if `restart_dt_s > 0`, `restart_stride = round(restart_dt_s / solver_dt)`
1. Initialize RestartWriter if `restart_stride > 0`
1. At restart steps (`step > 0 && step % restart_stride == 0`), call `restart.write()`
1. If `resume_mode`:
   - Read restart file via `RestartReader`
   - Restore u, v, a, pml_damping arrays
   - Start loop from `restored_step + 1`

### Config reading (io.cpp)

Add these to `/simulation/` attribute reads:

```cpp
read_attr_double(sim_grp, "record_depth_max_m", cfg.record_depth_max_m);
read_attr_double(sim_grp, "record_depth_actual_m", cfg.record_depth_actual_m);
read_attr_double(sim_grp, "green_tile_size_m", cfg.green_tile_size_m);
read_attr_double(sim_grp, "restart_dt_s", cfg.restart_dt_s);
read_attr_int(sim_grp, "restart_stride", cfg.restart_stride);
```

If `restart_dt_s` is missing or 0, no restart writes occur.

### Partition reading (io.cpp)

Read `/recording/` group:

```cpp
hid_t rec_grp = H5Gopen2(fid, "/recording", H5P_DEFAULT);
if (rec_grp >= 0) {
    data.recording.has_recording = true;
    data.recording.vertex_ids = read_dataset_int64(fid, "/recording/vertex_ids");
    data.recording.src_elem_local = read_dataset_int32(fid, "/recording/source_element_local_index");
    data.recording.src_corner = read_dataset_int8(fid, "/recording/source_corner_index");
    H5Gclose(rec_grp);
}
```

Note: HDF5 doesn't have `int8` natively — write/read as `H5T_NATIVE_INT8` (char/byte). In C++, `int8_t` maps to `signed char`.

### main.cpp

Add `--resume` flag:

```
gf_solver --direction x [--resume]
```

If `--resume`, set `cfg.resume_mode = true`.

### Constraints

- RecordWriter still uses extendible time dimension
- Restart overwrites the same file path (no versioning)
- Restart writes happen AFTER Newmark correct (state is at step end)
- If no `/recording/` group in partition, `recording.has_recording = false` and solver falls back to full-GLL recording (backward compat)
- Strain extraction at vertices: from the full GLL strain array, grab the corner node (index `src_corner[elem][n]`) for each recorded vertex

## Tests

The forward module uses Catch2 tests in `tests/`. For this task:

- Extend existing forward tests if they exist
- At minimum, verify compilation succeeds and the new classes construct/destruct cleanly

## Build

Add `src/restart.cpp` to `forward/CMakeLists.txt`:

```cmake
add_library(
    libgf STATIC
    ...
    src/restart.cpp)
```
