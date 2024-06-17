# Copyright (c) 2024 The Core Team
#
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0146 OLD)
project(CoreScheduler LANGUAGES C CXX CUDA)

option(CS_ENABLE_TESTS "" ON)
option(CS_ENABLE_EXAMPLES "" ON)
option(CS_ENABLE_BUILTIN_TOOLCHAIN "" ON)
if (CS_ENABLE_BUILTIN_TOOLCHAIN)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchain.cmake)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# Enable sanitizer compilation
# Env variable: ASAN_OPTIONS=detect_odr_violation=0:protect_shadow_gap=0
# add_compile_options(-fsanitize=address,undefined)
# link_libraries(-fsanitize=address,undefined)

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

find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDNN REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(NCCL REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

if (DEFINED ENV{CONDA_PREFIX})
    list(APPEND MPI_HOME ENV{CONDA_PREFIX}/bin/)
endif ()
find_package(MPI COMPONENTS C REQUIRED)

add_subdirectory(src)

if (CS_ENABLE_TESTS)
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif ()

if (CS_ENABLE_EXAMPLES)
    add_subdirectory(example)
endif ()

if (EXISTS "${CMAKE_BINARY_DIR}/compile_commands.json")
    file(COPY "${CMAKE_BINARY_DIR}/compile_commands.json"
            DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
endif ()
