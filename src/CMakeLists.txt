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

set(CS_MODULE_SRCS "")
add_subdirectory(module)

set(CS_COMPUTE_SRCS "")
add_subdirectory(compute)

set(CS_MEMORY_SRCS "")
add_subdirectory(memory)

set(CS_COMMUNICATION_SRCS "")
add_subdirectory(communication)

set(CS_OPTIMIZER_SRCS "")
add_subdirectory(optimizer)

set(CS_THREADING_SRCS "")
add_subdirectory(threading)

set(CS_DATASET_SRCS "")
add_subdirectory(data)

add_library(cs_core SHARED
        logger.cpp
        tensor.cpp
        ${CS_MODULE_SRCS}
        ${CS_COMPUTE_SRCS}
        ${CS_MEMORY_SRCS}
        ${CS_COMMUNICATION_SRCS}
        ${CS_OPTIMIZER_SRCS}
        ${CS_THREADING_SRCS}
        ${CS_DATA_SRCS}
)

target_compile_options(cs_core PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr;--extended-lambda;-O3;-Wextra;-Xcompiler;-Ofast;-Wextra>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;--expt-relaxed-constexpr;--extended-lambda;-O0;-Wextra;-Xcompiler;-Ofast;-Wextra>
        $<$<AND:$<CONFIG:Release>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Ofast;-Wextra>
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-O0;-Wextra>
)

target_include_directories(cs_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_include_directories(cs_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/internal
)

target_link_libraries(cs_core PRIVATE CUDA::cudart CUDA::nvtx3)
target_link_libraries(cs_core PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(cs_core PRIVATE arrow_dataset Arrow::arrow_shared Parquet::parquet_shared)
target_link_libraries(cs_core PRIVATE hwloc CUDA::nvml)
target_link_libraries(cs_core PUBLIC fmt::fmt-header-only spdlog::spdlog_header_only MPI::MPI_C torch)

target_compile_definitions(cs_core PRIVATE -DCS_ENABLE_INTERNAL_BUILD)

add_library(cs INTERFACE)

target_link_libraries(cs INTERFACE cs_core)
