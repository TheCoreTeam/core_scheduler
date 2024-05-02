cmake_minimum_required(VERSION 3.11)
project(dllm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DLLM_ENABLE_TESTS "" ON)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(third_party/fmt)

set(SPDLOG_FMT_EXTERNAL ON)
add_subdirectory(third_party/spdlog)

set(CUTLASS_ENABLE_HEADERS_ONLY ON)
set(CUTLASS_ENABLE_EXAMPLES OFF)
set(CUTLASS_ENABLE_TOOLS OFF)
set(CUTLASS_ENABLE_LIBRARY OFF)
set(CUTLASS_ENABLE_PROFILER OFF)
set(CUTLASS_ENABLE_PERFORMANCE OFF)
set(CUTLASS_ENABLE_TESTS OFF)
set(CUTLASS_INSTALL_TESTS ON)
set(CUTLASS_ENABLE_GTEST_UNIT_TESTS OFF)
add_subdirectory(third_party/cutlass)

add_subdirectory(src)

if (DLLM_ENABLE_TESTS)
    add_subdirectory(third_party/googletest)

    set(EIGEN_BUILD_DOC OFF)
    set(BUILD_TESTING OFF)
    add_subdirectory(third_party/eigen)

    add_subdirectory(test)
endif ()
