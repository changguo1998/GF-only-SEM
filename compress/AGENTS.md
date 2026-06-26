# compress/ — C++ HDF5 Compression Utilities

## Purpose

Header-only C++17 library (`INTERFACE` CMake target) providing HDF5 compression
policy components used by the forward solver's record writer.

## Components

| Header | Responsibility |
|--------|---------------|
| `CompressionFilter.h` | `CompressionMethod` enum (None, LZF, Zlib) + `ZlibConfig` + `CompressionConfig` struct + property list builder |
| `PrecisionPolicy.h` | float32 (cast from float64) vs float64 precision selection |
| `ChunkingStrategy.h` | HDF5 chunk dimension computation (element-first, chunk_size configurable, time dim chunk = 1) |
| `CheckpointWriter.h` | Convenience `CheckpointConfig` struct + file + dataset creation helpers |

## CMake

`INTERFACE` library target `gf_compress`. No compiled sources.
Forward solver links via `target_link_libraries(gf PRIVATE gf_compress)`.

## Deferred Items

- Compression benchmark tool (`compress/benchmark/` — not implemented)
- LZF not available on all HDF5 installs (handled at runtime with graceful skip)

## Design Doc

`docs/superpowers/design/compress.md`
