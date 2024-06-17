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

set(DLLM_MODULE_SRCS "")
add_subdirectory(module)

set(DLLM_COMPUTE_SRCS "")
add_subdirectory(compute)

set(DLLM_MEMORY_SRCS "")
add_subdirectory(memory)

set(DLLM_COMMUNICATION_SRCS "")
add_subdirectory(communication)

set(DLLM_OPTIMIZER_SRCS "")
add_subdirectory(optimizer)

set(DLLM_THREADING_SRCS "")
add_subdirectory(threading)

set(DLLM_DATASET_SRCS "")
add_subdirectory(data)

add_library(dllm_core SHARED
        logger.cpp
        tensor.cpp
        ${DLLM_MODULE_SRCS}
        ${DLLM_COMPUTE_SRCS}
        ${DLLM_MEMORY_SRCS}
        ${DLLM_COMMUNICATION_SRCS}
        ${DLLM_OPTIMIZER_SRCS}
        ${DLLM_THREADING_SRCS}
        ${DLLM_DATA_SRCS}
)

target_compile_options(dllm_core PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr;--extended-lambda;-O3;-Wextra;-Xcompiler;-Ofast;-Wextra>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;--expt-relaxed-constexpr;--extended-lambda;-O0;-Wextra;-Xcompiler;-Ofast;-Wextra>
        $<$<AND:$<CONFIG:Release>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Ofast;-Wextra>
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-O0;-Wextra>
)

target_include_directories(dllm_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_include_directories(dllm_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/internal
)

target_link_libraries(dllm_core PRIVATE CUDA::cudart CUDA::nvtx3)
target_link_libraries(dllm_core PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(dllm_core PRIVATE arrow_dataset Arrow::arrow_shared Parquet::parquet_shared)
target_link_libraries(dllm_core PRIVATE hwloc CUDA::nvml)
target_link_libraries(dllm_core PUBLIC fmt::fmt-header-only spdlog::spdlog_header_only MPI::MPI_C torch)

target_compile_definitions(dllm_core PRIVATE -DDLLM_ENABLE_INTERNAL_BUILD)

add_library(dllm INTERFACE)

target_link_libraries(dllm INTERFACE dllm_core)
