# Compression Module — Technical Design

> Parent: [docs/design-decisions.md](../../design-decisions.md)
> Implementation plan: [docs/superpowers/plans/2026-06-08-compress.md](../plans/2026-06-08-compress.md)

## Goal

Header-only C++ compression utility library that wraps HDF5 dataset creation with configurable compression filters, precision control, and chunking. Integrated into the forward solver's record write path.

## Architecture

Header-only C++17 library (`INTERFACE` CMake target) providing three independent policy components plus a convenience `write_checkpoint()` function that composes them. The forward solver links this library.

```
CompressionFilter  ─┐
PrecisionPolicy    ─┼─→ CheckpointWriter.write_checkpoint() → HDF5 datasets
ChunkingStrategy   ─┘
```

A standalone benchmarking tool compares speed and size tradeoffs. Catch2 tests verify round-trip correctness and float32 precision tolerance.

## Technology

- C++17 (compile features: `cxx_std_17`)
- HDF5 (C API from C++, via `HDF5::HDF5` target)
- Catch2 (testing)
- CMake (INTERFACE library target)

## Policy Components

### CompressionFilter

Enum + property list builder for HDF5 compression:

| Method | Description | Zlib Config |
|--------|------------|-------------|
| `None` | No compression | — |
| `LZF` | HDF5 built-in LZF (fast, modest ratio) | — |
| `Zlib` | Gzip/deflate (slower, better ratio) | `level` 1–9, default 6 |

### PrecisionPolicy

Template-based selection of HDF5 float type:

| Precision | HDF5 Type |
|-----------|-----------|
| `float32` | `H5T_NATIVE_FLOAT` |
| `float64` | `H5T_NATIVE_DOUBLE` |

### ChunkingStrategy

Computes chunk dimensions from the strain tensor shape to balance I/O throughput vs memory.

## Record Output Format

One strain record file per MPI rank per run direction. Strain dataset is extendible along the time axis. Restart is a separate latest-only file, not part of the strain record.

```
wavefields/{direction}/record_{r}.h5
├── attrs:
│   ├── rank                    : int32
│   ├── source_direction        : string          # "x", "y", or "z"
│   ├── basis                   : "mesh_vertices"
│   ├── record_depth_max_m      : float64
│   ├── record_depth_actual_m   : float64
│   └── excludes_pml            : bool
├── vertex_ids                  : int64[n_record_vertices]
└── strain                      : {precision}[n_snapshots, n_record_vertices, 6]
                                      # εxx,εyy,εzz,εxy,εxz,εyz
```

Latest-only restart lives separately:

```
restart/{direction}/restart_{r}.h5
├── displacement                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── velocity                    : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── acceleration                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
└── pml_memory_*                : float64[...]
```

### Write Pattern

1. **First snapshot**: create record file, create `vertex_ids` (fixed-size), create `strain` with dim 0 as `H5S_UNLIMITED`, write initial shallow mesh-vertex strain slice.
1. **Subsequent snapshots**: extend strain dim 0 by 1, write new shallow mesh-vertex strain slice.
1. **Restart overwrite**: when `step % restart_stride == 0`, overwrite `restart/{direction}/restart_{r}.h5` with full-volume state needed for exact resume.

### Data Shape

- 6 strain components (full symmetric tensor: εxx, εyy, εzz, εxy, εxz, εyz)
- Vertex-first layout: `[n_snapshots, n_record_vertices, 6]`
- `vertex_ids` maps local record index to global mesh vertex ID (1-based, from partition\_{r}.h5 `/recording/`)
- Strain precision is configurable: float32 (default, for production) or float64 (for validation). Set via config.py `snapshot_precision`. Restart state is always float64.

## Chunking

Chunked along time and recorded vertex dimensions:

```
chunk_dims = {1, min(chunk_size, n_record_vertices), 6}
```

Dim 0 chunk = 1 (one snapshot at a time, matching write pattern). `chunk_size` configurable (default: 64 or larger for production I/O).

## Namespace

`gf`

## File Layout

```
compress/
├── CMakeLists.txt                  — INTERFACE library target
├── include/gf/
│   ├── CompressionFilter.h         — enum + property list builder
│   ├── PrecisionPolicy.h           — float32/float64 template
│   ├── ChunkingStrategy.h          — chunk dimension computation
│   └── CheckpointWriter.h          — high-level write_checkpoint() with extendible dataset
├── benchmark/
│   ├── CMakeLists.txt
│   └── CompressionBenchmark.cpp    — speed vs size on sample strain data
└── tests/
    └── test_compress.cpp           — round-trip + precision tolerance
```
