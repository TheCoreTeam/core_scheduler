cmake_minimum_required(VERSION 3.25)
project(dllm LANGUAGES CXX CUDA)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT IPOResult)
if (IPOResult)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(STATUS "IPO / LTO not supported: <${IPOResult}>")
endif ()
set(BUILD_TESTING OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES "80;86;90a")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

option(DLLM_ENABLE_TESTS "" ON)
option(DLLM_ENABLE_EXAMPLES "" ON)
option(DLLM_ENABLE_FLASH_ATTENTION "" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDNN REQUIRED)
#find_package(CuTe REQUIRED)
find_package(NCCL REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

message(STATUS $ENV{CONDA_PREFIX})
if (DEFINED ENV{CONDA_PREFIX})
    list(APPEND MPI_HOME ENV{CONDA_PREFIX}/bin/)
endif ()
find_package(MPI REQUIRED)

add_subdirectory(src)

set(DLLM_EIGEN_ADDED OFF)
if (DLLM_ENABLE_TESTS)
    find_package(GTest REQUIRED)
    find_package(Eigen3 REQUIRED)
    set(DLLM_EIGEN_ADDED ON)

    add_subdirectory(test)
endif ()

if (DLLM_ENABLE_EXAMPLES)
    find_package(Eigen3 REQUIRED)

    add_subdirectory(example)
endif ()
