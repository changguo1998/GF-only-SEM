# tests/ — Shared Test Infrastructure

## Purpose

Shared tests for Python and C++ modules.

## Tiers

| Tier | Trigger | Scope |
| --- | --- | --- |
| Unit | commit hook / CI | Functions and classes |
| Integration | CI | Small forward runs |
| Slow | manual | Analytical benchmarks |
| Profile | manual | Performance and scaling |

完整测试清单、代表参数组合和验收门限见
[`docs/testing.md`](../docs/testing.md)。有限 Q 剪切、体积与时间步收敛测试已注册到 CTest。

## Layout

```
tests/
├── conftest.py                 — shared pytest fixtures
├── greenfun/                   — Green-function library and CLI tests
├── preprocess/                 — Python preprocess tests
├── tools/                      — GMSH→HDF5 tests
├── test_*.py                   — cross-module scientific regressions
├── test_*.cpp                  — Catch2 C++ tests
└── ../examples/                — per-model test case pipelines
```

## Commands

```bash
.venv/bin/python -m pytest tests -q
ctest --test-dir build --output-on-failure
./build/tests/test_gll "[GLL]" --reporter compact
bash examples/halfspace/compare.sh
```
