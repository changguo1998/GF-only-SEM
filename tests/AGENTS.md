# tests/ — Shared Test Infrastructure

## Purpose

Test suites for all modules, organized by module and test tier.

## Test Tiers

| Tier | Trigger | Scope |
|------|---------|-------|
| Unit | `git commit` hook / CI | Individual functions and classes |
| Integration | CI | Small forward run (N=3, few steps) |
| Slow | Manual | Analytical benchmarks (homogeneous half-space, layered medium) |
| Profile | Manual | Performance profiling and scalability |

## Test Layout

```
tests/
├── conftest.py                    — Shared pytest fixtures
├── test_*.cpp                     — 48 Catch2 C++ tests (GLL, element, assembly,
│                                      Newmark, PML, source, exchange, IO, record,
│                                      compress, integration)
├── preprocess/                    — 74 Python tests
│   ├── conftest.py
│   └── test_*.py
├── tools/
│   └── test_gmsh_to_hdf5*.py
├── postprocess/                   — 46 Python tests
└── examples/halfspace/            — End-to-end pipeline (via run.sh, not pytest)
```

## Running Tests

```bash
# Python
python -m pytest tests -q

# C++ via CTest
ctest --test-dir build --output-on-failure

# Specific C++ test
./build/tests/test_gll "[GLL]" --reporter compact

# End-to-end pipeline (separate from pytest)
bash examples/halfspace/run.sh
```