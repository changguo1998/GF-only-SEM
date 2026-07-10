# greenfun/ — AGENTS.md

## Purpose

Green's function reader with reciprocity-based source/receiver query. Loads
pre-computed strain Green's tensors written by `gf_postprocess` and supports
querying the strain field at arbitrary source locations via reciprocity.

## Language

Python 3

## Dependencies

- h5py
- numpy
- scipy

## Key Classes

| Class | Responsibility |
|-------|----------------|
| `GreenFunctionLibrary` | Read and index Green's function tiles from HDF5 |
| `SourceRun` | Encapsulate a single source location and its directional components |
| `GreenQuery` | Reciprocity-based query: strain at a target receiver from an arbitrary source |
| `TrilinearInterpolator` | Interpolate Green's tensor values at arbitrary spatial positions |

## CLI

```bash
gf_greenquery
```

Console script entry point registered in root `pyproject.toml`:

```
gf_greenquery = "greenfun.query:main"
```

## Location

Root-level `greenfun/` package.

## Design Doc

[`../docs/design/greenfun.md`](../docs/design/greenfun.md)