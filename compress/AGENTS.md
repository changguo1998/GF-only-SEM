# compress/ — C++ HDF5 Compression Utilities

## Purpose

Header-only C++17 `INTERFACE` library for HDF5 record compression.
Forward uses it in the record writer.

## Components

| Header | Role |
|--------|------|
| `CompressionFilter.h` | Compression method: none, LZF, or zlib. Builds HDF5 property lists. |
| `PrecisionPolicy.h` | Selects float32 or float64 output. |
| `ChunkingStrategy.h` | Computes HDF5 chunks. Time chunk = 1. Element chunk is configurable. |
| `CheckpointWriter.h` | Convenience config plus file/dataset helpers. |

## CMake

Target: `gf_compress`. No compiled sources.
Link with `target_link_libraries(gf PRIVATE gf_compress)`.

## Deferred

- Compression benchmark tool under `compress/benchmark/`.
- LZF may be missing in some HDF5 builds. Runtime handles this by skipping it.

## Design

See [`../docs/design/compress.md`](../docs/design/compress.md).
