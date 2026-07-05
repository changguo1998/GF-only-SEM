# Compression Module — Technical Design

> Parent: [../design-decisions.md](../design-decisions.md)

## Goal

Header-only C++ helpers for HDF5 compression, precision, and chunking. Used by forward record writes.

## Architecture

Header-only C++17 `INTERFACE` library. Provides compression filter, precision policy, and chunking strategy.

```
CompressionFilter  ─┐
PrecisionPolicy    ─┼─→ HDF5 property lists for record writes
ChunkingStrategy   ─┘
PrecisionPolicy    ─┼─→ CheckpointWriter.write_checkpoint() → HDF5 datasets
ChunkingStrategy   ─┘
```

Catch2 tests verify round-trip correctness and float32 precision tolerance.

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

One strain record per rank per direction. `strain` extends in time. Restart is separate and latest-only.

```
wavefields/{direction}/record_{r}_{step}.h5
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

Latest-only restart file:

```
restart/{direction}/restart_{r}.h5
├── displacement                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── velocity                    : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
├── acceleration                : float64[n_elem_local, NGLL, NGLL, NGLL, 3]
└── pml_memory_*                : float64[...]
```

## Chunking

Chunked along time and recorded vertex dimensions:

```
chunk_dims = {1, min(chunk_size, n_record_vertices), 6}
```

Dim 0 chunk = 1 because writes are one snapshot. `chunk_size` is configurable.

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
├── benchmark/
│   ├── CMakeLists.txt
│   └── CompressionBenchmark.cpp    — speed vs size on sample strain data
└── tests/
    └── test_compress.cpp           — round-trip + precision tolerance
```
