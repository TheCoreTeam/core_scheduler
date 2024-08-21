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

set(CORE_SCHEDULER_MODULE_SRCS "")
add_subdirectory(module)

set(CORE_SCHEDULER_COMPUTE_SRCS "")
add_subdirectory(compute)

set(CORE_SCHEDULER_MEMORY_SRCS "")
add_subdirectory(memory)

set(CORE_SCHEDULER_COMMUNICATION_SRCS "")
add_subdirectory(communication)

set(CORE_SCHEDULER_OPTIMIZER_SRCS "")
add_subdirectory(optimizer)

set(CORE_SCHEDULER_THREADING_SRCS "")
add_subdirectory(threading)

set(CORE_SCHEDULER_DATA_SRCS "")
add_subdirectory(data)

add_library(core_scheduler_core SHARED
        autocast.cpp
        logger.cpp
        tensor.cpp
        random.cpp
        ${CORE_SCHEDULER_MODULE_SRCS}
        ${CORE_SCHEDULER_COMPUTE_SRCS}
        ${CORE_SCHEDULER_MEMORY_SRCS}
        ${CORE_SCHEDULER_COMMUNICATION_SRCS}
        ${CORE_SCHEDULER_OPTIMIZER_SRCS}
        ${CORE_SCHEDULER_THREADING_SRCS}
        ${CORE_SCHEDULER_DATA_SRCS}
)

target_compile_options(core_scheduler_core PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:--expt-relaxed-constexpr;--extended-lambda;-O3;-Xcompiler=-Ofast,-Wextra>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;--expt-relaxed-constexpr;--extended-lambda;-O0;-Xcompiler=-Ofast,-Wextra>
        $<$<AND:$<CONFIG:Release>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Ofast;-Wextra>
        $<$<AND:$<CONFIG:Debug>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-O0;-Wextra>
)

target_include_directories(core_scheduler_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_include_directories(core_scheduler_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/internal
)

target_link_libraries(core_scheduler_core PRIVATE CUDA::cudart CUDA::nvtx3 nvidia::cudnn_frontend)
target_link_libraries(core_scheduler_core PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(core_scheduler_core PRIVATE arrow_dataset Arrow::arrow_shared Parquet::parquet_shared)
target_link_libraries(core_scheduler_core PRIVATE hwloc CUDA::nvml fmt::fmt-header-only spdlog::spdlog_header_only MPI::MPI_C)
target_link_libraries(core_scheduler_core PUBLIC torch)
target_compile_options(core_scheduler_core PRIVATE -fvisibility=hidden)
target_link_options(core_scheduler_core PRIVATE -Wl,--as-needed)

target_compile_definitions(core_scheduler_core PRIVATE -DCORE_SCHEDULER_ENABLE_INTERNAL_BUILD)

add_library(core_scheduler INTERFACE)

target_link_libraries(core_scheduler INTERFACE core_scheduler_core)
