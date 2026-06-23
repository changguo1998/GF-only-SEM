# tests/ — Shared Test Infrastructure

## Purpose

Test suites for all modules, organized by module and test tier.

## Test Tiers

| Tier | Trigger | Scope |
|------|---------|-------|
| Unit | `git commit` hook / CI | Individual functions and classes |
| Integration | CI | Small forward run (N=3, few steps), multi-element workflow |
| Slow | Manual (`GF_RUN_SLOW=1`) | 500k-element halfspace production workflow |
| Benchmark | Manual | Analytical benchmarks (homogeneous half-space, layered medium) |
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
├── workflows/
│   ├── conftest.py
│   ├── regular_hex_mesh.py        — Mesh generator helper
│   ├── test_multi_elem_workflow.py
│   └── test_halfspace_workflow.py — 500k elements (GF_RUN_SLOW=1)
├── tools/
│   └── test_gmsh_to_hdf5*.py
└── postprocess/                   — 46 Python tests
```

## Running Tests

```bash
# Python (all except 500k halfspace)
python -m pytest tests -q --ignore=tests/workflows/test_halfspace_workflow.py

# Python (with 500k halfspace)
GF_RUN_SLOW=1 python -m pytest tests -q

# C++ via CTest
ctest --test-dir build --output-on-failure

# Specific C++ test
./build/tests/test_gll "[GLL]" --reporter compact
```