cmake_minimum_required(VERSION 3.25)
project(dllm LANGUAGES CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(BUILD_TESTING OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES "80;86;90;90a")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

option(DLLM_ENABLE_TESTS "" ON)
if (DLLM_ENABLE_TESTS)
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif ()
option(DLLM_ENABLE_EXAMPLES "" ON)
option(DLLM_ENABLE_FLASH_ATTENTION "" ON)

find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)
find_package(CuTe REQUIRED)
find_package(NCCL REQUIRED)
find_package(MPI REQUIRED)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

#add_subdirectory(third_party/fmt)

#set(SPDLOG_FMT_EXTERNAL ON)
#add_subdirectory(third_party/spdlog)

#set(CUTLASS_ENABLE_HEADERS_ONLY ON)
#set(CUTLASS_ENABLE_EXAMPLES OFF)
#set(CUTLASS_ENABLE_TOOLS OFF)
#set(CUTLASS_ENABLE_LIBRARY OFF)
#set(CUTLASS_ENABLE_PROFILER OFF)
#set(CUTLASS_ENABLE_PERFORMANCE OFF)
#set(CUTLASS_ENABLE_TESTS OFF)
#set(CUTLASS_INSTALL_TESTS ON)
#set(CUTLASS_ENABLE_GTEST_UNIT_TESTS OFF)
#add_subdirectory(third_party/cutlass)

add_subdirectory(src)

set(DLLM_EIGEN_ADDED OFF)
if (DLLM_ENABLE_TESTS)
    find_package(GTest REQUIRED)
    find_package(Catch2 REQUIRED)
    find_package(Eigen3 REQUIRED)
    set(DLLM_EIGEN_ADDED ON)

    add_subdirectory(test)
endif ()

if (DLLM_ENABLE_EXAMPLES)
    find_package(Eigen3 REQUIRED)

    add_subdirectory(example)
endif ()
