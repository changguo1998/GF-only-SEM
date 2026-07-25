# tests/ — Shared Test Infrastructure

## Purpose

Shared tests for Python and C++ modules.

## Tiers

| Tier | Trigger | Scope |
|------|---------|-------|
| Unit | commit hook / CI | Functions and classes |
| Integration | CI | Small forward runs |
| Slow | manual | Analytical benchmarks |
| Profile | manual | Performance and scaling |

## Layout

```
tests/
├── conftest.py                 — shared pytest fixtures
├── test_*.cpp                  — Catch2 C++ tests
├── preprocess/                 — Python preprocess tests
├── tools/                      — GMSH→HDF5 tests
├── postprocess/                — Python postprocess tests
└── ../examples/                — per-model test case pipelines
```

## Commands

```bash
python -m pytest tests -q
ctest --test-dir build --output-on-failure
./build/tests/test_gll "[GLL]" --reporter compact
bash examples/halfspace/compare.sh
```
