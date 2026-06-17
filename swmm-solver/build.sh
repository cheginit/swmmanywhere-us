#!/bin/bash
set -euo pipefail

# On Apple, SWMM's CMake (src/solver/CMakeLists.txt) unconditionally fetches and
# builds LLVM OpenMP from source via FetchContent.  Replace that with a plain
# find_package so the conda-provided llvm-openmp is used instead: no network
# during the build and a toolchain-consistent libomp.  find_package(OpenMP)
# defines the same OpenMP::OpenMP_C target the solver links against.
if [[ "$(uname)" == "Darwin" ]]; then
  echo 'find_package(OpenMP REQUIRED C)' >cmake/openmp.cmake
  export CFLAGS="${CFLAGS:-} -D_FORTIFY_SOURCE=2"
  export CXXFLAGS="${CXXFLAGS:-} -D_FORTIFY_SOURCE=2"
fi

cmake -S . -B build -G "Unix Makefiles" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="$PREFIX" \
  -D BUILD_TESTS=OFF

cmake --build build -j"${CPU_COUNT:-2}"
cmake --install build
