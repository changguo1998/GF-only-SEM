#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
BUILD_DIR="${CASE_DIR}/.build"
MEMORY_LIMIT_GB="${GF_MEM_LIMIT_GB:-60}"
POSTPROCESS_RANKS="${POSTPROCESS_RANKS:-4}"
MEMLIMIT="${PROJECT_ROOT}/scripts/with_mem_limit.sh"

export PATH="${PROJECT_ROOT}/bin:${PATH}"
mkdir -p "${BUILD_DIR}"

echo "=== Build pure C++ mesh generator ==="
h5c++ -std=c++17 -O2 "${CASE_DIR}/mesh_gen.cpp" -o "${BUILD_DIR}/mesh_gen"
h5c++ -std=c++17 -O2 "${CASE_DIR}/verify_output.cpp" -o "${BUILD_DIR}/verify_output"

echo "=== Build case-specific C++ preprocessor ==="
MAIN_CMAKE_CACHE="${PROJECT_ROOT}/build/CMakeCache.txt"
if [ ! -f "${MAIN_CMAKE_CACHE}" ]; then
	echo "ERROR: main build cache not found; run scripts/build.sh first" >&2
	exit 1
fi
EIGEN3_DIR=$(sed -n 's/^Eigen3_DIR:[^=]*=//p' "${MAIN_CMAKE_CACHE}")
EIGEN_INCLUDE_DIR="$(cd "${EIGEN3_DIR}/../../.." && pwd)/include/eigen3"
METIS_INCLUDE_DIR=$(sed -n 's/^METIS_INCLUDE_DIR:[^=]*=//p' "${MAIN_CMAKE_CACHE}")
METIS_LIBRARY=$(sed -n 's/^METIS_LIBRARY:[^=]*=//p' "${MAIN_CMAKE_CACHE}")
MPIEXEC_EXECUTABLE=$(sed -n 's/^MPIEXEC_EXECUTABLE:[^=]*=//p' "${MAIN_CMAKE_CACHE}")
cmake --fresh -S "${PROJECT_ROOT}/preprocess/cpp" -B "${BUILD_DIR}/preprocess" \
    -DGF_USER_CONFIG="${CASE_DIR}/config_user_layer_surface.cpp" \
    -DGF_PREPROCESS_RUNTIME_DIR="${BUILD_DIR}" \
    -DEigen3_DIR="${EIGEN3_DIR}" \
    -DGF_EIGEN_INCLUDE_DIR="${EIGEN_INCLUDE_DIR}" \
    -DMETIS_INCLUDE_DIR="${METIS_INCLUDE_DIR}" \
    -DMETIS_LIBRARY="${METIS_LIBRARY}"
cmake --build "${BUILD_DIR}/preprocess" --target gf_preprocess -j"$(nproc)"

cd "${CASE_DIR}"
rm -rf config.h5 model.h5 partitions wavefields greenfun log restart_*.h5
mkdir -p log wavefields/x wavefields/y wavefields/z

echo "=== Generate layer-aligned topology in C++ ==="
"${BUILD_DIR}/mesh_gen" model.h5

echo "=== Run complete C++ preprocessing ==="
"${BUILD_DIR}/gf_preprocess" run model.h5 \
    --N 4 --cfl-safety 0.5 --nx 22 --ny 22 --n-ranks 16 \
    --pml-xmin 3 --pml-xmax 3 --pml-ymin 3 --pml-ymax 3 \
    --pml-zmin 0 --pml-zmax 3

echo "=== Run CUDA elastic solver for three force directions ==="
for direction in x y z; do
    echo "--- direction=${direction} ---"
    "${MEMLIMIT}" "${MEMORY_LIMIT_GB}" -- \
        "${PROJECT_ROOT}/bin/gf_solver_elastic_cuda" --direction "${direction}"
done

echo "=== Run C++ MPI postprocessor ==="
"${MEMLIMIT}" "${MEMORY_LIMIT_GB}" -- "${MPIEXEC_EXECUTABLE}" -n "${POSTPROCESS_RANKS}" \
    "${PROJECT_ROOT}/bin/gf_postprocess_mpi" model.h5 config.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/

tile_count=$(find greenfun -maxdepth 1 -type f -name 'tile_*.h5' | wc -l)
if [ "${tile_count}" -ne 16 ]; then
    echo "ERROR: expected 16 Green-function tiles, got ${tile_count}" >&2
    exit 1
fi

"${BUILD_DIR}/verify_output" greenfun

echo "PASS: pure C++ surface-source pipeline generated ${tile_count} Green-function tiles"
