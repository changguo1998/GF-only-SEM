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
| `LZF`  | HDF5 built-in LZF (fast, modest ratio) | — |
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

One file per MPI rank per run direction. Strain dataset is extendible along the time axis; restart state is overwritten each snapshot:

```
wavefields/{direction}/record_{r}.h5
├── attrs:
│   ├── rank               : int32
│   ├── source_direction   : string          # "x", "y", or "z"
│   └── ngll                : int32          # N+1
├── local_element_ids       : int64[n_elem_local]    ← 1-based global element IDs
├── elem_centroids          : float32[n_elem_local, 3]   # element centroid (x, y, z)
├── strain                  : {precision}[n_snapshots, n_elem_local, NGLL, NGLL, NGLL, 6]
│                                                       └── εxx,εyy,εzz,εxy,εxz,εyz
└── /restart/               ← overwritten each snapshot, not extendible
    ├── displacement        : float64[n_elem_local, NGLL, NGLL, NGLL, 3]   — u
    ├── velocity            : float64[n_elem_local, NGLL, NGLL, NGLL, 3]   — v
    └── acceleration        : float64[n_elem_local, NGLL, NGLL, NGLL, 3]   — a
```

### Write Pattern

1. **First snapshot**: create file, create `local_element_ids` (fixed-size), create `strain` with dim 0 as `H5S_UNLIMITED`, create `/restart/` datasets (fixed-size, overwritten each snapshot). Write initial strain slice + restart state.
2. **Subsequent snapshots**: extend strain dim 0 by 1, write new slice. Overwrite `/restart/` datasets with latest (u, v, a).

### Data Shape

- 6 strain components (full symmetric tensor: εxx, εyy, εzz, εxy, εxz, εyz)
- Element-first layout: `[n_snapshots, n_elem_local, NGLL, NGLL, NGLL, 6]`
- NGLL = N+1 (N=3 test: NGLL=4; N=5 prod: NGLL=6)
- `local_element_ids` maps local index to global element ID (1-based, from partition_{r}.h5)
- Strain precision is configurable: float32 (default, for production) or float64 (for validation). Set via config.py `snapshot_precision`. Restart state (u,v,a) is always float64.

## Chunking

Chunked along the element dimension:

```
chunk_dims = {1, min(chunk_size, n_elem_local), NGLL, NGLL, NGLL, 6}
```

Dim 0 chunk = 1 (one step at a time, matching write pattern). `chunk_size` configurable (default: 64).

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